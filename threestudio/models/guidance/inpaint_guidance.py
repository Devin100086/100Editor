from dataclasses import dataclass

import cv2
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
import threestudio.utils.vidtome as vidtome


@threestudio.register("stable-diffusion-inpainting-guidance")
class inpaintingGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        inpaint_name_or_path: str = "stabilityai/stable-diffusion-2-inpainting"

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

        diffusion_steps: int = 20

        use_sds: bool = False

        video: bool = False

        # vidtome
        chunk_size: int = 2
        chunk_ord: str = "mix-4"
        merge_global: bool = True
        local_merge_ratio: float = 0.9
        global_merge_ratio: float = 0.8
        global_rand: float =  0.5
        seed: int = 123456
        batch_size: int = 2
        align_batch: bool = True

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Inpainting model ...")

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

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.cfg.inpaint_name_or_path, **pipe_kwargs
        ).to(self.device)
        # self.scheduler = DDIMScheduler.from_pretrained(
        #     self.cfg.ddim_scheduler_name_or_path,
        #     subfolder="scheduler",
        #     torch_dtype=self.weights_dtype,
        #     cache_dir=self.cfg.cache_dir,
        # )
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

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
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.eval()
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.generator = torch.Generator("cuda").manual_seed(0)

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

        threestudio.info(f"Loaded Inpainting!")

        if self.cfg.video:
            self.activate_vidtome()

    def activate_vidtome(self):
        vidtome.apply_patch(self.pipe, self.cfg.local_merge_ratio, self.cfg.merge_global, self.cfg.global_merge_ratio, 
            seed = self.cfg.seed, batch_size = self.cfg.batch_size, align_batch = self.cfg.align_batch, global_rand = self.cfg.global_rand) 
        
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_prompt(
        self,
        batch_size,
        prompt
    ):
        text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device))
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.repeat(batch_size,1,1)

        uncond_tokens = [""]
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
        negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(self.device),
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size,1,1)

        return prompt_embeds, negative_prompt_embeds

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
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
    def encode_masks(
        self, mask: Float[Tensor, "B 1 H W"],
        masked_image: Float[Tensor, "B 3 H W"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = masked_image.dtype

        mask = mask.to(device=self.device, dtype=self.weights_dtype)
        masked_image = masked_image.to(device=self.device, dtype=self.weights_dtype)

        posterior = self.vae.encode(masked_image.to(self.weights_dtype)).latent_dist
        masked_image_latents = posterior.sample(self.generator) * self.vae.config.scaling_factor

        # masked_image_latents = self.encode_images(masked_image)

        mask = torch.cat([mask] * 2)
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
        )
        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=self.device, dtype=self.weights_dtype)
        return mask.to(input_dtype), masked_image_latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype),generator=self.generator).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    def prepare_latents(self,batch_size, generator, num_channels_latents, height, width, dtype, device):
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = noise * self.scheduler.init_noise_sigma

        return latents, noise

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 1024"],
        latents: Float[Tensor, "B 4 DH DW"],
        mask: Float[Tensor, "B 1 DH DW"],
        masked_image_latents: Float[Tensor, "B 4 DH DW"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        # self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            # add noise
            threestudio.debug("Start editing...")
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                    noise_pred = self.forward_unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )

                # perform classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = (noise_pred_uncond + 
                             self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            )

                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents, eta=1.0).prev_sample
            threestudio.debug("Editing finished.")
        return latents

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

    def edit_all_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        mask: Float[Tensor, "B 1 DH DW"],
        masked_image_latents: Float[Tensor, "B 4 DH DW"],
        cams= None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        print("Start editing images...")

        with torch.no_grad():

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for t in self.scheduler.timesteps:
                # pred noise
                chunks = self.get_chunks(len(latents))
                
                noise_preds = torch.zeros_like(latents)
                
                for chunk in chunks:
                    with torch.no_grad():
                        latent_model_input = torch.cat([latents[chunk]] * 2)
                        latent_model_input = torch.cat(
                        [latent_model_input, mask[chunk],masked_image_latents[chunk]], dim=1
                        )
                        pos,neg = text_embeddings.chunk(2)
                        text_embeddings_chunk = torch.cat([neg[chunk], pos[chunk]], dim=0)
                        eps = self.forward_unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings_chunk
                    )
                    noise_pred_uncond, noise_pred_text = eps.chunk(
                        2
                    )
                    # perform classifier-free guidance
                    noise_pred = (noise_pred_uncond + 
                                  self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
                                )
                    noise_preds[chunk] = noise_pred
                # get previous sample, continue loop
                latents = self.scheduler.step(noise_preds, t, latents).prev_sample
        print("Editing finished.")
        return latents

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        mask: Float[Tensor, "B 1 DH DW"],
        masked_image_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ):
        with torch.no_grad():
            # pred noise
            noise =latents
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            noise_pred = self.forward_unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )

        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
            + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        masks: Float[Tensor, "B H W C"],
        prompt: str,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        masks_BCHW = masks.permute(0, 3, 1, 2)

        latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        # latents = self.encode_images(rgb_BCHW_HW8)
        height, width = rgb_BCHW_HW8.shape[-2:]
        noise_latents, noise = self.prepare_latents(
                                batch_size,
                                self.generator,
                                4,
                                height,
                                width,
                                rgb_BCHW_HW8.dtype,
                                rgb_BCHW_HW8.device,
                            )

        # temp = torch.zeros(batch_size).to(rgb.device)

        positive_text_embeddings, negative_text_embeddings = self.encode_prompt(batch_size, prompt)

        # text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        # positive_text_embeddings, negative_text_embeddings = text_embeddings.chunk(2)
        text_embeddings = torch.cat([negative_text_embeddings, positive_text_embeddings], dim=0) 

        masked_image = (rgb_BCHW_HW8*2-1) * (masks_BCHW < 0.5)
        masks_BCHW_HW8 = F.interpolate(
            masks_BCHW,
            (RH//self.vae_scale_factor, RW//self.vae_scale_factor),
        )
        mask, masked_image_latents = self.encode_masks(masks_BCHW_HW8, masked_image)

        if self.cfg.use_sds:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
            grad = self.compute_grad_sds(text_embeddings, noise_latents, mask, masked_image_latents, t)
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
                edit_latents = self.edit_latents(text_embeddings, noise_latents, mask, masked_image_latents)
            else:
                edit_latents = self.edit_all_latents(text_embeddings, noise_latents, mask, masked_image_latents)
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
