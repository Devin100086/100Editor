import torch
from tqdm import tqdm
import datetime
import sys
import os
from argparse import ArgumentParser
from editor.gaussiansplatting.scene import GaussianModel
from editor.gaussiansplatting.gaussian_renderer import render
from editor.gaussiansplatting.arguments import PipelineParams
from editor.gaussiansplatting.gaussian_renderer import render
from editor.gaussiansplatting.scene.camera_scene import CamScene

from torchvision.utils import save_image
import numpy as np

def render_cameras_list(gaussian, colmap_dir, save_dir, use_original_resolution):
    parser = ArgumentParser(description="Training script parameters")
    pipe = PipelineParams(parser)
    background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
    if use_original_resolution != 0:
        scene = CamScene(colmap_dir, h=-1, w=-1)
    else:
        scene = CamScene(colmap_dir, h=512, w=512)
    colmap_cameras = scene.cameras

    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_dir = f"{save_dir}_{current_time}"
    if use_original_resolution:
        save_dir = os.path.join(save_dir,"rendered_origin")
    else:
        save_dir = os.path.join(save_dir,"rendered_no_origin")
    os.makedirs(save_dir, exist_ok=True)

    for cam in tqdm(colmap_cameras):
        render_pkg = render(cam, gaussian, pipe, background_tensor, separate_sh=True)
        out = render_pkg["render"]
        image = out[None]
        save_image(image, f"{save_dir}/{cam.image_name}.png")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="save/")
    parser.add_argument("--use_original_resolution", type=int, default=0)

    args = parser.parse_args()
    gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
    gaussian.load_ply(args.gs_source)
    gaussian.max_radii2D = torch.zeros(
        (gaussian.get_xyz.shape[0]), device="cuda"
    )
    print("🚀Start rendering...")
    render_cameras_list(gaussian, args.colmap_dir, args.save_dir, args.use_original_resolution)
    print(f"🌟Finish rendering! Renderings are saved in {args.save_dir}")
    
    