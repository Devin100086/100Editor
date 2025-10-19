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

class EditFineGuidance:
    def __init__(self, guidance, gaussian, masks, text_prompt, per_editing_step, edit_begin_step,
                 edit_until_step, lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale,
                 lambda_anchor_opacity, cams):
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
    
    def __call__(self, rendering, view_index, step):

        self.gaussian.update_learning_rate(step)

        # nerf2nerf loss
        if view_index not in self.edit_frames or (
                self.per_editing_step > 0
                and self.edit_begin_step
                < step
                < self.edit_until_step
                and step % self.per_editing_step == 0
        ):
            
            mask = (self.masks[view_index]/255)[:,:,:,np.newaxis]
            mask = mask.repeat(1,1,1,3).float().to(rendering.device)
            rgb = rendering * (1-mask)
            result = self.guidance(
                rgb,
                mask,
                self.prompt_utils,
            )
            self.edit_frames[view_index] = result["edit_images"].detach().clone() # 1 H W C
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
    
    def edit_all(self, frames, masks):

        # nerf2nerf loss
            
        # masks = (masks / 255)
        masks = torch.from_numpy(masks).permute(0,2,3,1).repeat(1,1,1,3).float().to(frames.device)
        rgb = frames * (1-masks)
        result = self.guidance(
            rgb,
            masks,
            self.prompt_utils,
        )

        gt_image = result["edit_images"].detach().clone()

        return gt_image
    
    # def get_loss(self, rendering, view_index, step):

    #     gt_image = self.edit_frames[view_index]

    #     loss = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
    #            self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
    #                                                 gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()
        
    #     # anchor loss
    #     if (
    #             self.lambda_anchor_color > 0
    #             or self.lambda_anchor_geo > 0
    #             or self.lambda_anchor_scale > 0
    #             or self.lambda_anchor_opacity > 0
    #     ):
    #         anchor_out = self.gaussian.anchor_loss()
    #         loss += self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
    #                 self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
    #                 self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
    #                 self.lambda_anchor_scale * anchor_out['loss_anchor_scale']

    #     return loss