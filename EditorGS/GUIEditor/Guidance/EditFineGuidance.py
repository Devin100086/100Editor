import os
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from diffusers.image_processor  import VaeImageProcessor
import torch

from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from torchvision.transforms.functional import to_pil_image

from threestudio.utils.sam import LangSAMTextSegmentor

def BrushNetGuidance(pipe, 
                    prompts,
                    mask_np,
                    original_image, 
                    generator,
                    num_inference_steps,
                    guidance_scale,
                    control_strength,
                    negative_prompt,
                    blending):
    if mask_np.ndim != 3:
        mask_np = mask_np[:, :, np.newaxis]

    mask_np = mask_np / 255
    height, width = mask_np.shape[0], mask_np.shape[1]
    ## resize the mask and original image to the same size which is divisible by vae_scale_factor
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor, do_convert_rgb=True)
    height_new, width_new = image_processor.get_default_height_width(original_image, height, width)
    mask_np = cv2.resize(mask_np, (width_new, height_new))[:,:,np.newaxis]
    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:, :, np.newaxis]

    original_image = cv2.resize(original_image, (width_new, height_new))

    init_image = original_image * (1 - mask_np)
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray((mask_np.repeat(3, -1) * 255).astype(np.uint8)).convert("RGB")

    brushnet_conditioning_scale = float(control_strength)
    
    images = pipe(
        prompts, 
        init_image, 
        mask_image, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale,
        negative_prompt=negative_prompt,
        height=height_new,
        width=width_new,
        output_type = 'pt'
    ).images
    return images


class EditFineGuidance:
    def __init__(self, guidance, gaussian, masks, text_prompt, per_editing_step, edit_begin_step,
                 edit_until_step, lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale,
                 lambda_anchor_opacity, cams, colmap_cameras):
        self.guidance = guidance
        self.gaussian = gaussian
        self.per_editing_step = per_editing_step
        self.edit_begin_step = edit_begin_step
        self.edit_until_step = edit_until_step
        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.masks = masks
        self.cams = cams

        self.cameras = colmap_cameras
        self.use_masked_image = False

        self.edit_frames = {}
        self.visible = True
        self.prompt_utils = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )()
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.lang_sam = LangSAMTextSegmentor().to(get_device())
    
    def __call__(self, rendering, view_index, step, edit_text, negative_prompt):

        self.gaussian.update_learning_rate(step)

        # nerf2nerf loss
        if view_index not in self.edit_frames or (
                self.per_editing_step > 0
                and self.edit_begin_step
                < step
                < self.edit_until_step
                and step % self.per_editing_step == 0
        ):
            image_pil = to_pil_image(rendering.squeeze(0).permute(2, 0, 1))
            # result = self.guidance(
            #     rendering,
            #     self.origin_frames[view_index],
            #     self.prompt_utils,
            # )
            generator = torch.Generator("cuda").manual_seed(1)
            result = BrushNetGuidance(self.guidance, 
                                        edit_text+", high quality, extremely detailed",
                                        self.masks[view_index],
                                        np.array(image_pil), 
                                        generator,
                                        5,
                                        7.5,
                                        1,
                                        negative_prompt,
                                        blending = True)
            edit_images = result.permute(0, 2, 3, 1)
            # self.edit_frames[view_index] = result["edit_images"].detach().clone() # 1 H W C
            self.edit_frames[view_index] = edit_images.detach().clone()
            # print("edited image index", cur_index)

        gt_image = self.edit_frames[view_index]

        loss = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
               self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                                    gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()
        
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

        return loss
        