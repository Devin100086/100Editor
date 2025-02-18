from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import BaseTrainer
from EditorGS.GUIEditor.Guidance.DelGuidance import DelGuidance
from torchvision.transforms.functional import to_pil_image, to_tensor
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.gaussiansplatting.scene.camera_scene import CamScene
from PIL import Image
from EditorGS.GUIEditor.utils import *
from EditorGS.GUIEditor.Network import EditorNetwork
from threestudio.utils.misc import (
    dilate_mask,
    fill_closed_areas,
)

class DeleteTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.edit_cam_num = cfg.edit_cam_num
        self.lambda_l1 = cfg.lambda_l1
        self.lambda_p = cfg.lambda_p
        self.lambda_anchor_color = cfg.lambda_anchor_color
        self.lambda_anchor_geo = cfg.lambda_anchor_geo
        self.lambda_anchor_scale = cfg.lambda_anchor_scale
        self.lambda_anchor_opacity = cfg.lambda_anchor_opacity
        self.per_editing_step = cfg.per_editing_step
        self.edit_begin_step = cfg.edit_begin_step
        self.edit_until_step = cfg.edit_until_step
        self.inpaint_prompt = cfg.inpaint_prompt
        self.delete_prompt = cfg.delete_prompt
        self.inpaint_scale = cfg.inpaint_scale
        self.colmap_dir = cfg.colmap_dir
        self.mask_dilate  = cfg.mask_dilate
        self.fix_holes = True
        self.lang_sam = LangSAMTextSegmentor().to(get_device())

        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras


    def delete(self):
        edit_cameras = sample_train_camera(self.colmap_cameras,
                                           self.edit_cam_num,
                                          )

        from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                DDIMScheduler,
            )

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        pipe.enable_model_cpu_offload()

        self.ctn_inpaint = pipe
        self.ctn_inpaint.set_progress_bar_config(disable=True)
        self.ctn_inpaint.safety_checker = None

        self.update_mask(self.colmap_cameras, text_prompt=self.delete_prompt)

        # origin_frames = self.render_cameras_list(edit_cameras)
        num_channels_latents = self.ctn_inpaint.vae.config.latent_channels
        shape = (
            1,
            num_channels_latents,
            edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
            edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
        )

        latents = torch.zeros(shape, dtype=torch.float16, device="cuda")

        dist_thres = (
            self.inpaint_scale * self.cameras_extent * self.gaussian.percent_dense
        )
        valid_remaining_idx = self.gaussian.get_near_gaussians_by_mask(
            self.gaussian.mask, dist_thres
        )
        # Prune and update mask to valid_remaining_idx
        self.gaussian.prune_with_mask(new_mask=valid_remaining_idx)

        inpaint_2D_mask, origin_frames = self.render_all_view_with_mask(
            edit_cameras
        )

        self.guidance = DelGuidance(
            guidance=self.ctn_inpaint,
            latents=latents,
            gaussian=self.gaussian,
            text_prompt=self.inpaint_prompt,
            lambda_l1=self.lambda_l1,
            lambda_p=self.lambda_p,
            lambda_anchor_color=self.lambda_anchor_color,
            lambda_anchor_geo=self.lambda_anchor_geo,
            lambda_anchor_scale=self.lambda_anchor_scale,
            lambda_anchor_opacity=self.lambda_anchor_opacity,
            cams=edit_cameras,
        )

        view_index_stack = list(range(len(edit_cameras)))
        ema_loss_for_log = 0.0
        network = EditorNetwork(host="127.0.0.1",port=8084)
        for step in tqdm(range(self.edit_train_steps)):
            network.render(self.pipe,self.gaussian,ema_loss_for_log,render,self.background_tensor,step,self.opt)
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

            loss = self.guidance(
                rendering,
                origin_frames[view_index],
                inpaint_2D_mask[view_index],
                view_index,
                step,
            )
            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return
            
            ema_loss_for_log = self.alpha * ema_loss_for_log + (1-self.alpha) * loss.item()
        
        os.makedirs("save", exist_ok=True)
        self.gaussian.save_ply("save/result1.ply")

    @torch.no_grad()
    def render_all_view_with_mask(self, edit_cameras):
        inpaint_2D_mask = []
        origin_frames = []

        for _, cam in enumerate(edit_cameras):
            res = self.render(cam)
            rgb, mask = res["comp_rgb"], res["masks"]
            mask = dilate_mask(mask.to(torch.float32), self.mask_dilate)
            if self.fix_holes:
                mask = fill_closed_areas(mask)
            inpaint_2D_mask.append(mask)
            origin_frames.append(rgb)

        return inpaint_2D_mask, origin_frames


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--edit_cam_num", type=int, default=0, help="Camera number.")
    parser.add_argument("--delete_prompt", type=str, default="man", help="Delete Prompt.")
    parser.add_argument("--inpaint_prompt", type=str, default="wall", help="Inpaint Prompt.")
    parser.add_argument("--text_prompt", type=str, default="", help="Lambda anchor color.")
    parser.add_argument("--edit_train_steps", type=int, default=1500, help="Edit train steps.")
    parser.add_argument("--per_train_step", type=int, default=1, help="Per train step.")
    parser.add_argument("--per_editing_step", type=int, default=1, help="Per editing step.")
    parser.add_argument("--edit_begin_step", type=int, default=0, help="Edit begin step.")
    parser.add_argument("--edit_until_step", type=int, default=100, help="Edit until step.")
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="Lambda L1.")
    parser.add_argument("--lambda_p", type=int, default=2, help="Lambda P.")
    parser.add_argument("--lambda_anchor_color", type=float, default=1.0, help="Lambda anchor color.")
    parser.add_argument("--lambda_anchor_geo", type=float, default=1.0, help="Lambda anchor geo.")
    parser.add_argument("--lambda_anchor_scale", type=float, default=1.0, help="Lambda anchor scale.")
    parser.add_argument("--lambda_anchor_opacity", type=float, default=1.0, help="Lambda anchor opacity.")
    parser.add_argument("--inpaint_scale", type=float, default=1.0, help="Inpaint scale.")
    parser.add_argument("--mask_dilate", type=int, default=15, help="Mask dilate.")

    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = DeleteTrainer(args)
        trainer.gaussian.update_anchor_term(
            anchor_weight_init_g0=0.05,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=1.3,
        )
        trainer.configure_optimizers()
        trainer.delete()