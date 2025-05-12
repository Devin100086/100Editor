import os
import copy
from argparse import ArgumentParser
from os import makedirs
from typing import Tuple
import cv2
from scipy.spatial.transform import Rotation as R

import imageio
import open3d as o3d
import torch
import torchvision
from natsort import natsorted
from torchvision.io import write_video
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from EditorGS.gaussiansplatting.scene import GaussianModel
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.gaussiansplatting.arguments import PipelineParams
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.gaussiansplatting.scene.camera_scene import CamScene
from EditorGS.gaussiansplatting.scene.cameras import CustomCam

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def slerp(q0, q1, t):
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    s0 = np.cos(theta_t) - dot * sin_theta_t / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * q0 + s1 * q1

def generate(cameras, num_frames):
    
    video_cams = []

    for i in range(len(cameras) - 1):
      cam0 = cameras[i]
      cam1 = cameras[i + 1]

      R0 = cam0.R
      R1 = cam1.R
      t0 = cam0.T
      t1 = cam1.T
      q0 = R.from_matrix(R0).as_quat()
      q1 = R.from_matrix(R1).as_quat()
      for j in range(num_frames + 1):
          t = j / (num_frames + 1)
          q_interp = slerp(q0, q1, t)
          t_interp = (1 - t) * t0 + t * t1
          R_interp = R.from_quat(q_interp).as_matrix()
          extr_interp = np.eye(4)
          extr_interp[:3, :3] = R_interp
          extr_interp[:3, 3] = t_interp

          c2w = torch.from_numpy(extr_interp).to(torch.float32)
          new_cam = copy.deepcopy(cameras[0])
          new_cam.image_height = int(new_cam.image_height / 2) * 2
          new_cam.image_width = int(new_cam.image_width / 2) * 2
          new_cam.world_view_transform = torch.tensor(getWorld2View2(R_interp, t_interp, new_cam.trans, new_cam.scale)).transpose(0, 1).cuda()
          new_cam.full_proj_transform = (new_cam.world_view_transform.unsqueeze(0).bmm(new_cam.projection_matrix.unsqueeze(0))).squeeze(0)
          new_cam.camera_center = new_cam.world_view_transform.inverse()[3, :3]

          video_cams.append(new_cam)
    
    return video_cams


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="save/")
    parser.add_argument("--use_original_resolution", type=int, default=0)
    parser.add_argument("--render_path", type=bool, default=True)
    parser.add_argument("--save_gif", type=bool, default=True)

    args = parser.parse_args()

    parser = ArgumentParser(description="Training script parameters")
    pipe = PipelineParams(parser)

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
    if args.use_original_resolution != 0:
        scene = CamScene(args.colmap_dir, h=-1, w=-1)
    else:
        scene = CamScene(args.colmap_dir, h=512, w=512)
    background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
    
    if args.render_path:
        print("render videos ...")
        n_fames = 2
        cam_traj = generate(scene.cameras, num_frames=n_fames)
        
        filename = args.save_dir
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        gif = imageio.get_writer("save/output.gif", mode='I', duration=0.1) if args.save_gif else None
        for render_cam in tqdm(cam_traj):
            img = render(render_cam, gaussian, pipe, background_tensor)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
            gif.append_data(img)
        video.close()
        if args.save_gif:
            gif.close()
        print(f"Video saved in {filename}.")
        
    # if args.render_spiral:
    #     print("render spiral videos ...")
    #     traj_dir = os.path.join(args.model_path, 'spiral', "ours_{}".format(scene.loaded_iter))
    #     os.makedirs(traj_dir, exist_ok=True)
    #     n_fames = 120
    #     cam_traj = generate_path_spiral(scene.getTrainCameras(), n_frames=n_fames, args=args)
    #     create_videos(base_dir=traj_dir,
    #                 input_dir=traj_dir, 
    #                 out_name='render_spiral', 
    #                 num_frames=n_fames)
    