import torch
import numpy as np
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from torchvision.transforms.functional import to_pil_image, to_tensor, gaussian_blur
from torchvision.transforms import ToTensor
from EditorGS.lama.saicinpainting.evaluation.refinement import refine_predict

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
        image_in = image_in.permute(0, 3, 1, 2)
        mask_in = mask_in.unsqueeze(0)
        mask_in = gaussian_blur(mask_in, kernel_size=(77, 77))
        mask_in[mask_in < 0.1] = 0
        mask_in[mask_in >= 0.1] = 1
        batch = {'image': image_in, 'mask': mask_in,  "unpad_to_size": [torch.tensor([image_in.shape[2]]), torch.tensor([image_in.shape[3]])]}
        out = refine_predict(batch, self.guidance, gpu_ids="0, ", 
                            modulo=8,
                            n_iters=15, # number of iterations of refinement for each scale
                            lr=0.002, # learning rate
                            min_side=512, # all sides of image on all scales should be >= min_side / sqrt(2)
                            max_scales=3, # max number of downscaling scales for the image-mask pyramid
                            px_budget=1800000)

        self.edit_frames[view_index] = out.permute(0,2,3,1) # 1 C H W to 1 H W C
        self.depth_frames[view_index] = self.to_tensor(self.depthPredictor(to_pil_image(out.squeeze(0)))["depth"])[None].permute(0,2,3,1)

    def __call__(self, rendering, depth_rendering, image_in, mask_in, view_index, step):
        self.gaussian.update_learning_rate(step)
        if view_index not in self.edit_frames:
            self.inpaint_with_mask_ctn(image_in, mask_in, view_index)

        gt_image = self.edit_frames[view_index]

        gt_image = gt_image.to(get_device())

        loss = self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                                        gt_image.permute(0, 3, 1, 2).contiguous(), ).sum() # 1 H W C to 1 C H W

        # loss = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
        #        self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
        #                                                 gt_image.permute(0, 3, 1, 2).contiguous(), ).sum() # 1 H W C to 1 C H W
        # anchor loss
        # if (
        #         self.lambda_anchor_color > 0
        #         or self.lambda_anchor_geo > 0
        #         or self.lambda_anchor_scale > 0
        #         or self.lambda_anchor_opacity > 0
        # ):
        #     anchor_out = self.gaussian.anchor_loss()
        #     loss = self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
        #             self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
        #             self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
        #             self.lambda_anchor_scale * anchor_out['loss_anchor_scale']

        loss = self.pearson_depth_loss(depth_rendering, self.depth_frames[view_index].to(depth_rendering.device))

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
    