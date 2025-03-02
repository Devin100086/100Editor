import torch
import numpy as np
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from torchvision.transforms.functional import to_pil_image, to_tensor, gaussian_blur
from torchvision.transforms import ToTensor

from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
import torch.nn.functional as F
from transformers import pipeline
# Diffusion model (cached) + prompts + edited_frames + training config

class DelGuidance:
    def __init__(self, guidance, origin_frames, gaussian, text_prompt,
                 lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale, lambda_anchor_opacity,
                 cams):
        self.guidance = guidance # ctn-inpaint guidance
        self.origin_frames = origin_frames
        self.depthPredictor = pipeline(task="depth-estimation", model="depth-anything/depth-anything-V2-Base-hf")
        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.gaussian = gaussian
        self.edit_frames = {}
        self.depth_frames = {}
        self.text_prompt = text_prompt
        self.cams = cams
        self.visible = True

        self.prompt_utils = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )
        self.step = 0
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.to_tensor = ToTensor()

    @torch.no_grad()
    def inpaint_with_mask_ctn(self, image_in, mask_in, view_index) -> None:
        device = get_device()  # Get the device (cpu or cuda)
        generator = torch.Generator(device=device).manual_seed(123)
        image_in = image_in.permute(0, 3, 1, 2)
        image_in = F.interpolate(image_in, (1024, 1024))
        mask_in = mask_in.unsqueeze(0)
        mask_in = F.interpolate(mask_in, (1024, 1024))
        mask_in = gaussian_blur(mask_in, kernel_size=(77, 77))
        out = self.guidance(
                    prompt=self.text_prompt,
                    image=image_in,
                    mask_image=mask_in,
                    height=1024,
                    width=1024,
                    AAS=True, # enable AAS
                    strength=0.8, # inpainting strength
                    rm_guidance_scale=9, # removal guidance scale
                    ss_steps = 9, # similarity suppression steps
                    ss_scale = 0.3, # similarity suppression scale
                    AAS_start_step=0, # AAS start step
                    AAS_start_layer=34, # AAS start layer
                    AAS_end_layer=70, # AAS end layer
                    num_inference_steps=50, # number of inference steps # AAS_end_step = int(strength*num_inference_steps)
                    generator=generator,
                    guidance_scale=1,
                ).images[0]

        self.edit_frames[view_index] = F.interpolate(self.to_tensor(out).to("cuda")[None], (512,512)).permute(0,2,3,1) # 1 C H W to 1 H W C
        self.depth_frames[view_index] = F.interpolate(self.to_tensor(self.depthPredictor(out)["depth"])[None], (512,512)).permute(0,2,3,1)

    def __call__(self, rendering, depth_rendering, image_in, mask_in, view_index, step):
        self.gaussian.update_learning_rate(step)
        if view_index not in self.edit_frames:
            self.inpaint_with_mask_ctn(image_in, mask_in, view_index)

        gt_image = self.edit_frames[view_index]

        loss = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
               self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                                        gt_image.permute(0, 3, 1, 2).contiguous(), ).sum() # 1 H W C to 1 C H W
        # anchor loss
        if (
                self.lambda_anchor_color > 0
                or self.lambda_anchor_geo > 0
                or self.lambda_anchor_scale > 0
                or self.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss += self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
                    self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
                    self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
                    self.lambda_anchor_scale * anchor_out['loss_anchor_scale']

        loss += self.pearson_depth_loss(depth_rendering, self.depth_frames[view_index].to(depth_rendering.device))

        return loss
    
    def pearson_depth_loss(self, depth_src, depth_target):
        #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

        src = depth_src - depth_src.mean()
        target = depth_target - depth_target.mean()

        src = src / (src.std() + 1e-6)
        target = target / (target.std() + 1e-6)

        co = (src * target).mean()
        assert not torch.any(torch.isnan(co))
        return 1 - co
    