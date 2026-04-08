import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline, DDIMInverseScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
import threestudio.utils.mvtm as mvtm


def get_threestudio_cache_root() -> str:
    cache_root_env = os.getenv("THREESTUDIO_CACHE_DIR")
    if cache_root_env:
        return Path(cache_root_env).expanduser().resolve().as_posix()
    repo_root = Path(__file__).resolve().parents[4]
    return (repo_root / "runtime" / ".threestudio_cache").as_posix()


@threestudio.register("stable-diffusion-controlnet-guidance")
class ControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        control_type: str = "depth"  # normal/canny/depth

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 25

        use_sds: bool = False

        video: bool = False

        # Canny threshold
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100

        # vidtome
        chunk_size: int = 2
        chunk_ord: str = "mix-4"
        merge_global: bool = True
        local_merge_ratio: float = 0.9
        global_merge_ratio: float = 0.9
        global_rand: float =  0.5
        seed: int = 142857
        batch_size: int = 2
        align_batch: bool = True

    cfg: Config

    def make_inpaint_condition(self,image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        controlnet_name_or_path: str
        if self.cfg.control_type == "normal":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_normalbae"
        elif self.cfg.control_type == "canny":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_canny"
        elif self.cfg.control_type == "p2p":
            controlnet_name_or_path = "lllyasviel/control_v11e_sd15_ip2p"
        elif self.cfg.control_type == "inpaint":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_inpaint"
        elif self.cfg.control_type == "depth":
            controlnet_name_or_path = "lllyasviel/sd-controlnet-depth"

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        controlnet = ControlNetModel.from_pretrained(
            controlnet_name_or_path,
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet, **pipe_kwargs
        ).to(self.device)
        self.inverse_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet, **pipe_kwargs
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )

        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(
                self.cfg.ddim_scheduler_name_or_path, 
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                cache_dir=self.cfg.cache_dir,
        )

        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        self.ddim_inverser.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.inverse_unet = self.inverse_pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()

        if self.cfg.control_type == "normal":
            self.preprocessor = NormalBaeDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self.preprocessor.model.to(self.device)
        elif self.cfg.control_type == "canny":
            self.preprocessor = CannyDetector()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded ControlNet!")

        if self.cfg.video:
            self.activate_vidtome()
    
    def activate_vidtome(self):
        mvtm.apply_patch(self.pipe, self.cfg.local_merge_ratio, self.cfg.merge_global, self.cfg.global_merge_ratio, 
            seed = self.cfg.seed, batch_size = self.cfg.batch_size, align_batch = self.cfg.align_batch, global_rand = self.cfg.global_rand) 

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def inverse_forward_control_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.inverse_unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    def get_chunks(self, flen):
        x_index = torch.arange(flen)

        # The first chunk has a random length
        rand_first = np.random.randint(0, self.cfg.chunk_size) + 1
        chunks = x_index[rand_first:].split(self.cfg.chunk_size, dim=0)
        chunks = [x_index[:rand_first]] + list(chunks) if len(chunks[0]) > 0 else [x_index[:rand_first]]
        if np.random.rand() > 0.5:
            chunks = chunks[::-1]
        
        # Chunk order only matter when we do global token merging
        if self.cfg.merge_global == False:
            return chunks

        # Chunk order. "seq": sequential order. "rand": full permutation. "mix": partial permutation.
        if self.cfg.chunk_ord == "rand":
            order = torch.randperm(len(chunks))
        elif self.cfg.chunk_ord == "mix":
            randord = torch.randperm(len(chunks)).tolist()
            rand_len = int(len(randord) / self.perm_div)
            seqord = sorted(randord[rand_len:])
            if rand_len > 0:
                randord = randord[:rand_len]
                if abs(seqord[-1] - randord[-1]) < abs(seqord[0] - randord[-1]):
                    seqord = seqord[::-1]
                order = randord + seqord
            else:
                order = seqord
        else:
            order = torch.arange(len(chunks))
        chunks = [chunks[i] for i in order]
        return chunks

    def ddim_inversion(self, latents, image_cond, text_embeddings):
        for t in self.ddim_inverser.timesteps:
            noises = []
            x_index = torch.arange(len(latents))
            batches = x_index.split(self.cfg.batch_size, dim = 0)
            with torch.no_grad():
                for batch in batches:
                # pred noise
                    latent_model_input = torch.cat([latents[batch]] * 2)
                    image_cond_input = torch.cat([image_cond[batch]] * 2)
                    pos, neg = text_embeddings.chunk(2)
                    text_embeddings_chunk = torch.cat([pos[batch], neg[batch]], dim=0)

                    (
                        down_block_res_samples,
                        mid_block_res_sample,
                    ) = self.forward_controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings_chunk,
                        image_cond=image_cond_input,
                        condition_scale=self.cfg.condition_scale,
                    )

                    noise_pred = self.inverse_forward_control_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings_chunk,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    noises += [noise_pred]
            noises = torch.cat(noises)
            # get previous sample, continue loop
            latents = self.ddim_inverser.step(noises, t, latents).prev_sample

        threestudio.debug("DDIM Inversion finished.")
        return latents

    def edit_all_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        origin_text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 3 DH DW"],
        t: Int[Tensor, "B"],
        cams= None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        self.ddim_inverser.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.ddim_inverser.set_timesteps(self.cfg.diffusion_steps)

        print("Start editing images...")

        with torch.no_grad():
            # add noise
            # noise = torch.randn_like(latents)
            # latents = self.scheduler.add_noise(latents, noise, t) 
            latents = self.ddim_inversion(latents, image_cond_latents, origin_text_embeddings)

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for t in self.scheduler.timesteps:
                # pred noise
                chunks = self.get_chunks(len(latents))
                
                noise_preds = torch.zeros_like(latents)
                
                for chunk in chunks:
                    with torch.no_grad():
                        latent_model_input = torch.cat([latents[chunk]] * 2)
                        image_cond_latent = torch.cat([image_cond_latents[chunk]] * 2)
 
                        pos, neg = text_embeddings.chunk(2)
                        text_embeddings_chunk = torch.cat([pos[chunk], neg[chunk]], dim=0)
                    (
                        down_block_res_samples,
                        mid_block_res_sample,
                    ) = self.forward_controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings_chunk,
                        image_cond=image_cond_latent,
                        condition_scale=self.cfg.condition_scale,
                    )
                    noise_pred = self.forward_control_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings_chunk,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    # perform classifier-free guidance
                    noise_pred = (
                        noise_pred_uncond
                        + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    )
                    noise_preds[chunk] = noise_pred
                # get previous sample, continue loop
                latents = self.scheduler.step(noise_preds, t, latents).prev_sample
                # vidtome.update_patch(self.pipe, global_tokens = None)
        print("Editing finished.")
        return latents

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond: Float[Tensor, "B 3 H W"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            threestudio.debug("Start editing...")
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    latent_model_input = torch.cat([latents] * 2)
                    image_cond_input = torch.cat([image_cond] * 2)

                    (
                        down_block_res_samples,
                        mid_block_res_sample,
                    ) = self.forward_controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        image_cond=image_cond_input,
                        condition_scale=self.cfg.condition_scale,
                    )

                    noise_pred = self.forward_control_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )
                # perform classifier-free guidance
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            threestudio.debug("Editing finished.")
        return latents

    def prepare_image_cond(self, cond_rgb: Float[Tensor, "B H W C"]):
        if self.cfg.control_type == "normal":
            cond_rgb = (
                (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            )
            detected_map = self.preprocessor(cond_rgb)
            control = (
                torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            )
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "canny":
            cond_rgb = (
                (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            )
            blurred_img = cv2.blur(cond_rgb, ksize=(5, 5))
            detected_map = self.preprocessor(
                blurred_img, self.cfg.canny_lower_bound, self.cfg.canny_upper_bound
            )
            control = (
                torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            )
            control = control.unsqueeze(-1).repeat(1, 1, 3)
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "depth":
            control = cond_rgb / torch.max(cond_rgb)
            # depth_map = cond_rgb
            control = control.permute(0, 3, 1, 2)
            control = torch.concatenate([control, control, control], dim=1)
        elif self.cfg.control_type == "p2p":
            control = cond_rgb.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "inpaint":
            control = cond_rgb.permute(0, 3, 1, 2)

        return control

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond: Float[Tensor, "B 3 H W"],
        t: Int[Tensor, "B"],
    ):
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        origin_prompt_utils: PromptProcessorOutput,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        # assert batch_size == 1
        assert rgb.shape[:-1] == cond_rgb.shape[:-1]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)

        image_cond = self.prepare_image_cond(cond_rgb)
        image_cond = F.interpolate(
            image_cond, (RH, RW), mode="bilinear", align_corners=False
        )

        temp = torch.zeros(batch_size).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        origin_text_embeddings = origin_prompt_utils.get_text_embeddings(temp, temp, temp, False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if self.cfg.video:
            t = torch.randint(
                self.max_step - 1,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
        else:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

        if self.cfg.use_sds:
            grad = self.compute_grad_sds(text_embeddings, latents, image_cond, t)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
        else:
            if self.cfg.video == False:
                edit_latents = self.edit_latents(text_embeddings, latents, image_cond, t)
            else:
                edit_latents = self.edit_all_latents(text_embeddings, origin_text_embeddings, latents, image_cond, t)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/debugging/controlnet-normal.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )

    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (guidance_out["edit_images"][0].detach().cpu().clip(0, 1).numpy() * 255)
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    cache_root = get_threestudio_cache_root()
    os.makedirs(cache_root, exist_ok=True)
    cv2.imwrite(os.path.join(cache_root, "edit_image.jpg"), edit_image)
