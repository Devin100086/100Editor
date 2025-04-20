import argparse
import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.GUIEditor.utils import *
from EditorGS.gaussiansplatting.scene.camera_scene import CamScene
from threestudio.utils.clip_metrics import *


def metric(orign_gaussian, edited_gaussian, colmap_dir, use_original_resolution, clip_prompt_origin, clip_prompt_target):

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

    clip_metrics = ClipSimilarity().to(orign_gaussian.get_xyz.device)
    total_sim_direction = 0
    total_sim = 0
    
    with torch.no_grad():
        for cam in tqdm(colmap_cameras):
            origin_render_pkg = render(cam, origin_gaussian, pipe, background_tensor)
            origin_out = origin_render_pkg["render"].unsqueeze(0)

            edited_render_pkg = render(cam, edited_gaussian, pipe, background_tensor)
            edited_out = edited_render_pkg["render"].unsqueeze(0)

            _, sim, cos_sim, _ = clip_metrics(origin_out, edited_out,
                                            clip_prompt_origin, clip_prompt_target)
            total_sim_direction += abs(cos_sim.item())
            total_sim += abs(sim.item())
    print(clip_prompt_origin, clip_prompt_target, "sim", total_sim / len(colmap_cameras), "sim_direction", total_sim_direction / len(colmap_cameras))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_prompt_origin", type=str, required=True)
    parser.add_argument("--clip_prompt_target", type=str, required=True)
    parser.add_argument("--origin_gs_source", type=str, required=True)
    parser.add_argument("--edited_gs_source", type=str, required=True)
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--use_original_resolution", type=int, default=0)
    
    args = parser.parse_args()
    origin_gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
    origin_gaussian.load_ply(args.origin_gs_source)
    origin_gaussian.max_radii2D = torch.zeros(
        (origin_gaussian.get_xyz.shape[0]), device="cuda"
    )

    edited_gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
    edited_gaussian.load_ply(args.edited_gs_source)
    edited_gaussian.max_radii2D = torch.zeros(
        (edited_gaussian.get_xyz.shape[0]), device="cuda"
    )

    print("🚀Start Evaluating...")
    metric(origin_gaussian, edited_gaussian, args.colmap_dir, args.use_original_resolution, args.clip_prompt_origin, args.clip_prompt_target)
    print("🌟Finish Evaluating!")



