import random
from omegaconf import OmegaConf
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"gaussiansplatting"))
from EditorGS.gaussiansplatting.scene import GaussianModel
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from threestudio.utils.typing import *
from threestudio.utils.transform import rotate_gaussians
from threestudio.utils.dpt import DPT
from argparse import ArgumentParser
from threestudio.utils.misc import (
    get_device,
    step_check,
    dilate_mask,
    erode_mask,
    fill_closed_areas,
)
from threestudio.utils.sam import LangSAMTextSegmentor
from threestudio.utils.camera import camera_ray_sample_points, project, unproject

from argparse import ArgumentParser
from EditorGS.gaussiansplatting.scene.camera_scene import CamScene


class BaseTrainer:
    def __init__(self,cfg):
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.edit_cam_num = cfg.edit_cam_num
        self.guidance_type = cfg.guidance_type
        self.edit_text = cfg.text_prompt
        self.edit_train_steps = cfg.edit_train_steps
        self.per_editing_step = cfg.per_editing_step
        self.edit_begin_step = cfg.edit_begin_step
        self.edit_until_step = cfg.edit_until_step
        self.lambda_l1 = cfg.lambda_l1
        self.lambda_p = cfg.lambda_p
        self.lambda_anchor_color = cfg.lambda_anchor_color
        self.lambda_anchor_geo = cfg.lambda_anchor_geo
        self.lambda_anchor_scale = cfg.lambda_anchor_scale
        self.lambda_anchor_opacity = cfg.lambda_anchor_opacity

        self.gs_lr_scaler = 3.0
        self.lr_final_scaler = 2.0
        self.color_lr_scaler = 3.0
        self.opacity_lr_scaler = 2.0
        self.scaling_lr_scaler = 2.0
        self.rotation_lr_scaler = 2.0
        self.gs_lr_end_scaler = 2.0
        self.densify_until_step = 1300
        self.densification_interval = 100
        self.max_densify_percent = 0.01
        self.min_opacity = 0.005
        self.alpha = 0.99
        # training cfg

        self.use_sam = False
        self.guidance = None
        self.stop_training = False
        self.inpaint_end_flag = False
        self.scale_depth = True
        self.depth_end_flag = False
        self.seg_scale = True
        self.seg_scale_end = False
        # from original system
        self.points3d = []
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        # load
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        # front end related
        self.colmap_cameras = None
        self.render_cameras = None

        # diffusion model
        self.ip2p = None
        self.ctn_ip2p = None

        self.ctn_inpaint = None
        self.ctn_ip2p = None
        self.training = False
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.masks_2D = {}
        # self.text_segmentor = LangSAMTextSegmentor().to(get_device())
        # self.sam_predictor = self.text_segmentor.model.sam
        # self.sam_predictor.is_image_set = True
        # self.sam_features = {}
        # self.semantic_gauassian_masks = {}
        # self.semantic_gauassian_masks["ALL"] = torch.ones_like(self.gaussian._opacity)

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        # status
        self.display_semantic_mask = False
        self.display_point_prompt = False

        self.viewer_need_update = False
        self.system_need_update = False
        self.inpaint_again = True
        self.scale_depth = True

    @torch.no_grad()
    def render_cameras_list(self, edit_cameras):
        origin_frames = []
        for cam in edit_cameras:
            out = self.render(cam)["comp_rgb"]
            origin_frames.append(out)

        return origin_frames

    def render(
        self,
        cam,
        local=False,
        sam=False,
        train=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if train:
            self.viewspace_point_tensor = viewspace_point_tensor
            self.radii = radii
            self.visibility_filter = self.radii > 0.0

        semantic_map = render(
            cam,
            self.gaussian,
            self.pipe,
            self.background_tensor,
            override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
        )["render"]
        semantic_map = torch.norm(semantic_map, dim=0)
        semantic_map = semantic_map > 0.0  # 1, H, W
        semantic_map_viz = image.detach().clone()  # C, H, W
        semantic_map_viz = semantic_map_viz.permute(1, 2, 0)  # 3 512 512 to 512 512 3
        semantic_map_viz[semantic_map] = 0.50 * semantic_map_viz[
            semantic_map
        ] + 0.50 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
        semantic_map_viz = semantic_map_viz.permute(2, 0, 1)  # 512 512 3 to 3 512 512

        render_pkg["sam_masks"] = []
        render_pkg["point2ds"] = []
        if sam:
            if hasattr(self, "points3d") and len(self.points3d) > 0:
                sam_output = self.sam_predict(image, cam)
                if sam_output is not None:
                    render_pkg["sam_masks"].append(sam_output[0])
                    render_pkg["point2ds"].append(sam_output[1])

        self.gaussian.localize = False  # reverse

        render_pkg["semantic"] = semantic_map_viz[None]
        render_pkg["masks"] = semantic_map[None]  # 1, 1, H, W

        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image  # 1 H W C

        depth = render_pkg["depth_3dgs"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        return {
            **render_pkg,
        }

    def densify_and_prune(self, step):
        if step <= self.densify_until_step.value:
            self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian.max_radii2D[self.visibility_filter],
                self.radii[self.visibility_filter],
            )
            self.gaussian.add_densification_stats(
                self.viewspace_point_tensor.grad, self.visibility_filter
            )

            if step > 0 and step % self.densification_interval.value == 0:
                self.gaussian.densify_and_prune(
                    max_grad=1e-7,
                    max_densify_percent=self.max_densify_percent.value,
                    min_opacity=self.min_opacity.value,
                    extent=self.cameras_extent,
                    max_screen_size=5,
                )
    
    def configure_optimizers(self):
        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= self.edit_train_steps,
            lr_scaler = self.gs_lr_scaler,
            lr_final_scaler = self.gs_lr_end_scaler,
            color_lr_scaler = self.color_lr_scaler,
            opacity_lr_scaler = self.opacity_lr_scaler,
            scaling_lr_scaler = self.scaling_lr_scaler,
            rotation_lr_scaler = self.rotation_lr_scaler,
        )
        opt = OmegaConf.create(vars(opt))
        # opt.update(self.training_args)
        self.gaussian.spatial_lr_scale = self.cameras_extent
        self.gaussian.training_setup(opt)
        self.opt = opt

    def densify_and_prune(self, step):
        if step <= self.densify_until_step:
            self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian.max_radii2D[self.visibility_filter],
                self.radii[self.visibility_filter],
            )
            self.gaussian.add_densification_stats(
                self.viewspace_point_tensor.grad, self.visibility_filter
            )

            if step > 0 and step % self.densification_interval == 0:
                self.gaussian.densify_and_prune(
                    max_grad=1e-7,
                    max_densify_percent=self.max_densify_percent,
                    min_opacity=self.min_opacity,
                    extent=self.cameras_extent,
                    max_screen_size=5,
                )