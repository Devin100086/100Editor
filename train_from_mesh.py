import os
import numpy as np
import torch
from random import randint
from EditorGS.gaussiansplatting.utils.loss_utils import l1_loss, ssim
from EditorGS.gaussiansplatting.gaussian_renderer import render
import sys
from threestudio.utils.perceptual import PerceptualLoss

# from gaussiansplatting.scene import Scene, GaussianModel
from EditorGS.gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel as Vanilla_GaussianModel
from EditorGS.gaussiansplatting.scene.gaussian_model import GaussianModel

from EditorGS.gaussiansplatting.utils.general_utils import safe_state
import uuid
from tqdm import tqdm, trange
from EditorGS.gaussiansplatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from EditorGS.gaussiansplatting.arguments import ModelParams, PipelineParams, OptimizationParams
from EditorGS.gaussiansplatting.utils.graphics_utils import BasicPointCloud
from mediapy import write_image

from threestudio.utils.render import render_multiview_images_from_mesh
from threestudio.utils.mesh import load_mesh_as_pcd_trimesh
from omegaconf import OmegaConf
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor


def replace_filename(path, new_filename):
    dir_name = os.path.dirname(path)
    new_path = os.path.join(dir_name, new_filename)
    return new_path


def get_scene_radius(c2ws, scale: float = 1.1):
    camera_centers = c2ws[..., :3, 3]
    camera_centers = np.linalg.norm(
        camera_centers - np.mean(camera_centers, axis=0, keepdims=True), axis=-1
    )
    return np.max(camera_centers) * scale


def training(
        dataset,
        opt,
        pipe,
        den_percent,
        prompt,
        per_editing_step,
        coarse_iteration,
        coarse_path,
        save_path,
        hori_cams,
        cams,
        gt_images,
        mesh_file,
        camera_extent,
        args
):
    # poses = None
    # cams = []
    # gt_images = []
    val_dir = replace_filename(save_path, "val")
    os.makedirs(val_dir, exist_ok=True)
    gt_images = torch.from_numpy(gt_images)
    gt_images = gt_images.moveaxis(-1, 1)

    opt.position_lr_init = 0
    opt.position_lr_final = 0
    opt.scaling_lr = 0
    opt.rotation_lr = 0
    opt.opacity_lr = 0
    opt.feature_lr = 0.00625

    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(3, 1, 1, 1.5)

    xyz, color = load_mesh_as_pcd_trimesh(mesh_file, 1200000)
    pcd = BasicPointCloud(xyz, color, None)

    gaussians.create_from_pcd(pcd, camera_extent)
    # scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gaussians.save_ply(save_path)


if __name__ == "__main__":
    max_steps = 0
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser, max_steps= max_steps)
    pp = PipelineParams(parser)


    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--guide", type=str, default="n2n")

    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--per_editing_step", type=int, default=50000000)
    parser.add_argument("--den_percent", type=float, default=0.0)
    parser.add_argument("--coarse_iteration", type=int, default=0)


    args = parser.parse_args(sys.argv[1:])
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    coarse_path = replace_filename(args.save_path, "coarse_gs.obj")
    # TODO: add more arguments
    cams, images, camera_extent = render_multiview_images_from_mesh(args.mesh,
                                                                    save_path=replace_filename(args.save_path,
                                                                                               "mesh_rendering.mp4"))
    hori_cams, _, _ = render_multiview_images_from_mesh(args.mesh, horizontal=True,
                                                        save_path=replace_filename(args.save_path,
                                                                                   "horizontal_mesh_rendering.mp4"))
    # hard code
    args.iterations = max_steps
    args.densify_grad_threshold = 0.00005
    args.opacity_reset_interval = 100000000

    # if not os.path.exists(coarse_path):w
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.den_percent,
        args.prompt,
        args.per_editing_step,
        args.coarse_iteration,
        coarse_path,
        args.save_path,
        hori_cams,
        cams,
        images,
        args.mesh,
        camera_extent,
        args
    )