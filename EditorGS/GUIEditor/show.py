import pickle
from pathlib import Path
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.gaussiansplatting.scene.vanilla_gaussian_model import (
    GaussianModel as VanillaGaussianModel,
)
from EditorGS.gaussiansplatting.utils.graphics_utils import fov2focal
from EditorGS.GUIEditor.utils import *
from EditorGS.GUIEditor.Network import EditorNetwork
from torchvision.transforms.functional import to_tensor
from torchvision.ops import masks_to_boxes
from utils import *
import numpy as np
import torch


class ShowGaussian():
    def __init__(self,gs_source,colmap_dir):
        self.gs_source = gs_source
        self.colmap_dir = colmap_dir
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.scale_depth = True
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.gs_lr_scaler = 3.0
        self.lr_final_scaler = 2.0
        self.color_lr_scaler = 3.0
        self.opacity_lr_scaler = 2.0
        self.scaling_lr_scaler = 2.0
        self.rotation_lr_scaler = 2.0
        self.gs_lr_end_scaler = 2.0
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

    def configure_optimizers(self):
        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= 0,
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

    def show(self,depth,cam):
        network = EditorNetwork(host="127.0.0.1",port=8084)
        cache_dir = Path("tmp_add").absolute().as_posix()
        os.makedirs(cache_dir, exist_ok=True)
        mv_image_dir = os.path.join(cache_dir, "multiview_pred_images")
        os.makedirs(mv_image_dir, exist_ok=True)
        inpaint_path = os.path.join(cache_dir, "inpainted.png")
        removed_bg_path = os.path.join(cache_dir, "removed_bg.png")
        gs_path = os.path.join(cache_dir, "inpaint_gs.obj")

        removed_bg = Image.open(removed_bg_path)
        inpainted_image = to_tensor(Image.open(inpaint_path))[:3,...].to("cuda")

        object_mask = np.array(removed_bg)
        object_mask = object_mask[:, :, 3] > 0
        object_mask = torch.from_numpy(object_mask)
        bbox = masks_to_boxes(object_mask[None])[0].to("cuda")

        depth_estimator = DPT(get_device(), mode="depth")

        estimated_depth = depth_estimator(
            inpainted_image.moveaxis(0, -1)[None, ...]
        ).squeeze()
        # ui_utils.vis_depth(estimated_depth.cpu())
        object_center = (bbox[:2] + bbox[2:]) / 2

        fx = fov2focal(cam.FoVx, cam.image_width)
        fy = fov2focal(cam.FoVy, cam.image_height)

        object_center = (
            object_center
            - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
        ) / torch.tensor([fx, fy]).to("cuda")

        with torch.no_grad():
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        rendered_depth = render_pkg["depth_3dgs"][..., ~object_mask]

        inpainted_depth = estimated_depth[~object_mask]
        object_depth = estimated_depth[..., object_mask]

        min_object_depth = torch.quantile(object_depth, 0.05)
        max_object_depth = torch.quantile(object_depth, 0.95)
        obj_depth_scale = (max_object_depth - min_object_depth) * 1

        min_valid_depth_mask = (min_object_depth - obj_depth_scale) < inpainted_depth
        max_valid_depth_mask = inpainted_depth < (max_object_depth + obj_depth_scale)
        valid_depth_mask = torch.logical_and(min_valid_depth_mask, max_valid_depth_mask)
        valid_percent = valid_depth_mask.sum() / min_valid_depth_mask.shape[0]
        print("depth valid percent: ", valid_percent)

        rendered_depth = rendered_depth[0, valid_depth_mask]
        inpainted_depth = inpainted_depth[valid_depth_mask.squeeze()]

        ## assuming rendered_depth = a * estimated_depth + b
        y = rendered_depth
        x = inpainted_depth
        a = (torch.sum(x * y) - torch.sum(x) * torch.sum(y)) / (
            torch.sum(x**2) - torch.sum(x) ** 2
        )
        b = torch.sum(y) - a * torch.sum(x)

        z_in_cam = object_depth.min() * a + b

        new_object_gaussian = None

        if self.scale_depth:
            if new_object_gaussian is not None:
                self.gaussian.prune_with_mask()
            scaled_z_in_cam = z_in_cam * depth
            x_in_cam, y_in_cam = (object_center.cuda()) * scaled_z_in_cam
            T_in_cam = torch.stack([x_in_cam, y_in_cam, scaled_z_in_cam], dim=-1)

            bbox = bbox.cuda()
            real_scale = (
                (bbox[2:] - bbox[:2])
                / torch.tensor([fx, fy], device="cuda")
                * scaled_z_in_cam
            )

            new_object_gaussian = VanillaGaussianModel(self.gaussian.max_sh_degree)
            new_object_gaussian.load_ply(gs_path)
            new_object_gaussian._opacity.data = (
                torch.ones_like(new_object_gaussian._opacity.data) * 99.99
            )

            new_object_gaussian._xyz.data -= new_object_gaussian._xyz.data.mean(
                dim=0, keepdim=True
            )
            rotate_gaussians(new_object_gaussian, default_model_mtx.T)

            object_scale = (
                new_object_gaussian._xyz.data.max(dim=0)[0]
                - new_object_gaussian._xyz.data.min(dim=0)[0]
            )[:2]

            relative_scale = (real_scale / object_scale).mean()
            print(relative_scale)

            scale_gaussians(new_object_gaussian, relative_scale)

            new_object_gaussian._xyz.data += T_in_cam

            R = torch.from_numpy(cam.R).float().cuda()
            T = -R @ torch.from_numpy(cam.T).float().cuda()

            rotate_gaussians(new_object_gaussian, R)
            translate_gaussians(new_object_gaussian, T)

            self.gaussian.concat_gaussians(new_object_gaussian)
            self.scale_depth = False
            self.gaussian.save_ply("tmp_add/merge.ply")
            network.render(self.pipe,self.gaussian,0, render,self.background_tensor,0,self.opt,show=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--cam_dir", type=str,required=True)
    parser.add_argument("--depth", type=float,required=False)

    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        shower = ShowGaussian(args.gs_source,args.colmap_dir)
        shower.configure_optimizers()
        with open(args.cam_dir, 'rb') as f:
            cam = pickle.load(f)
        
        shower.show(args.depth, cam)