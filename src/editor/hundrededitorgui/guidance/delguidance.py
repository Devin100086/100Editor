import torch
import numpy as np
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from torchvision.transforms.functional import to_pil_image,to_tensor
from torchvision.transforms import ToTensor
from torch.nn import functional as F

from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from transformers import pipeline
from PIL import Image
from src.utils.path_utils import resolve_runtime_subdir

try:
    from leftrefill import predict
except ModuleNotFoundError:
    try:
        from extern.LeftRefill.run import predict
    except ModuleNotFoundError:
        from LeftRefill.run import predict
# Diffusion model (cached) + prompts + edited_frames + training config

class DelGuidance:
    def __init__(self, guidance, gaussian, per_editing_step, edit_begin_step, edit_until_step,
                 lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale, lambda_anchor_opacity,
                 cams):
        self.guidance = guidance # ctn-inpaint guidance
        self.depthPredictor = pipeline(task="depth-estimation", model="depth-anything/depth-anything-V2-Base-hf")
        self.lambda_l1 = lambda_l1
        self.per_editing_step = per_editing_step
        self.edit_begin_step = edit_begin_step
        self.edit_until_step = edit_until_step
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.gaussian = gaussian
        self.edit_frames = {}
        self.depth_frames = {}
        self.cams = cams
        self.visible = True

        # self.prompt_utils = StableDiffusionPromptProcessor(
        #     {
        #         "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        #         "prompt": text_prompt,
        #     }
        # )()
        self.step = 0
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.to_tensor = ToTensor()
        self.reference_image_path = (
            resolve_runtime_subdir(__file__, "cache", "delete", create=True) / "Reference.png"
        )


    @torch.no_grad()
    def inpaint_with_mask_ctn(self, image_in, mask_in, view_index) -> None:
        image_in_pil = to_pil_image(image_in[0].permute(2, 0, 1)) # 1, H, W, C to C, H, W
        mask_in_pil = to_pil_image(torch.stack([mask_in[0].to(torch.uint8) * 255] * 3)) # C, H, W 255

        def make_inpaint_condition(image, image_mask):
            image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

            assert image.shape[0:1] == image_mask.shape[
                                       0:1], "image and image_mask must have the same image size"
            image[image_mask > 0.5] = -1.0  # set as masked pixel
            image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return image
        
        source = {"image":image_in_pil, "mask":mask_in_pil}
        reference = Image.open(self.reference_image_path.as_posix())
        out = predict(source, reference, 25, 1, 2.5, 124241)[0]
        # control_image = make_inpaint_condition(image_in_pil, mask_in_pil).to("cuda")
        # generator = torch.Generator(device="cuda").manual_seed(0)
        # out = self.guidance(
        #     self.text_prompt,
        #     num_inference_steps=20,
        #     generator=generator,
        #     eta=1.0,
        #     image=image_in_pil,
        #     mask_image=mask_in_pil,
        #     control_image=control_image,
        # ).images[0]
        # out.save(f"image_{view_index}.png")
        self.edit_frames[view_index] = self.to_tensor(out).to("cuda")[None].permute(0,2,3,1) # 1 C H W to 1 H W C
        # self.depth_frames[view_index] = self.to_tensor(self.depthPredictor(out)["depth"]).unsqueeze(0).permute(0,2,3,1)

    def __call__(self, rendering, image_in, mask_in, view_index, step):
        torch.cuda.empty_cache()
        self.gaussian.update_learning_rate(step)

        if view_index not in self.edit_frames:
            # self.inpaint_with_mask_ctn(image_in, mask_in, view_index)
            # mask = mask_in.unsqueeze(0).permute(0, 2, 3, 1).repeat(1,1,1,3).float().to(rendering.device)
            # rgb = image_in * (1-mask)
            self.inpaint_with_mask_ctn(image_in, mask_in, view_index)

        gt_image = self.edit_frames[view_index].permute(0,3,1,2)
        gt_image = F.interpolate(gt_image, size=(rendering.shape[1], rendering.shape[2]), mode='bilinear', align_corners=False).permute(0,2,3,1)
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

        # loss += self.pearson_depth_loss(depth_rendering, self.depth_frames[view_index].to(depth_rendering.device))

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
    
    def edit_all(self, frames, masks):
        
        # nerf2nerf loss
        mask = masks.unsqueeze(-1).repeat(1,1,1,3).float().to(frames.device)
        rgb = frames * (1-mask)

        result = self.guidance(
            rgb,
            mask,
            self.prompt_utils,
        )
        # result = self.guidance(
        #     frames,
        #     masks.unsqueeze(-1).float().to(frames.device),
        #     self.text_prompt,
        # )
        gt_image = result["edit_images"].detach().clone()

        return gt_image
    
