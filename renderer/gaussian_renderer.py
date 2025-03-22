import copy
import os
import traceback
from typing import List
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from pathlib import Path

from compression.compression_exp import run_single_decompression
from gaussiansplatting.gaussian_renderer import render_simple,render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from lumina3D_utils.dict_utils import EasyDict


class GaussianRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0

    def pixel_to_ray(self, pixel, intrinsic, extrinsic):
        extrinsic = extrinsic.cpu().numpy()
        u, v = pixel
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        x = (u - cx) / fx
        y = (v - cy) / fy
        z = 1.0

        ray_camera = np.array([x, y, z, 1.0])

        ray_origin = extrinsic[:3, 3]
        ray_direction = extrinsic[:3, :3] @ ray_camera[:3]
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        return ray_origin, ray_direction

    def find_intersection(self, ray_origin, ray_direction, point_cloud):
        points = point_cloud.cpu().numpy()
        vectors = points - ray_origin
        projections = np.dot(vectors, ray_direction)
        distances = np.linalg.norm(vectors - np.outer(projections, ray_direction), axis=1)
        nearest_index = np.argmin(distances)
        return points[nearest_index]

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        resolution,
        ply_file_paths,
        cam_params,
        current_ply_names,
        background_color,
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        use_splitscreen=False,
        highlight_border=False,
        save_ply_path=None,
        slider={},
        roate_point = None,
        **other_args,
    ):
        cam_params = cam_params.to("cuda")
        slider = EasyDict(slider)
        if len(ply_file_paths) == 0:
            res.error = "Select a .ply file"
            return

        # Remove old scenes
        if len(ply_file_paths) < self._last_num_scenes:
            for i in range(ply_file_paths, self.num_parallel_scenes):
                self.gaussian_models[i] = None
            self._last_num_scenes = len(ply_file_paths)

        images = []
        for scene_index, ply_file_path in enumerate(ply_file_paths):
            # Load
            if ply_file_path != self._current_ply_file_paths[scene_index]:
                self.gaussian_models[scene_index] = self._load_model(ply_file_path)
                center = self.gaussian_models[scene_index]._xyz.mean(dim=0)
                self.gaussian_models[scene_index]._xyz = self.gaussian_models[scene_index]._xyz - center
                self._current_ply_file_paths[scene_index] = ply_file_path

            # Edit
            gs: GaussianModel = copy.deepcopy(self.gaussian_models[scene_index])

            try:
                exec(self.sanitize_command(edit_text))
            except Exception as e:
                error = traceback.format_exc()
                error += str(e)
                res.error = error

            # Render video
            if len(video_cams) > 0:
                self.render_video("./_videos", video_cams, gs)

            # Render current view
            fov_rad = fov / 360 * 2 * np.pi

            fx = (resolution / 2) / np.tan(fov_rad / 2)
            fy = (resolution / 2) / np.tan(fov_rad / 2)
            cx = resolution / 2
            cy = resolution / 2

            intrinsic = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ])

            if roate_point is not None:
                ray_origin, ray_direction = self.pixel_to_ray(roate_point, intrinsic, cam_params)
                intersection_point = self.find_intersection(ray_origin, ray_direction, gs._xyz)
                gs._xyz = self.gaussian_models[scene_index]._xyz = self.gaussian_models[scene_index]._xyz - torch.from_numpy(intersection_point).to("cuda")

            render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
            render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))
            if render_alpha:
                images.append(render["alpha"])
            elif render_depth:
                images.append(render["depth"] / render["depth"].max())
            else:
                images.append(render["render"])

            # Save ply
            if save_ply_path is not None:
                self.save_ply(gs, save_ply_path)

        self._return_image(
            images,
            res,
            normalize=img_normalize,
            use_splitscreen=use_splitscreen,
            highlight_border=highlight_border,
        )

        res.mean_xyz = torch.mean(gs.get_xyz, dim=0)
        res.std_xyz = torch.std(gs.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def _load_model(self, ply_file_path):
        if ply_file_path.endswith(".ply"):
            model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
            model.load_ply(ply_file_path)
        elif ply_file_path.endswith("compression_config.yml"):
            model = run_single_decompression(Path(ply_file_path).parent.absolute())
        else:
            raise NotImplementedError("Only .ply or .yml files are supported.")
        return model

    def render_video(self, save_path, video_cams, gaussian):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/rotate_{len(os.listdir(save_path))}.mp4"
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        for render_cam in tqdm(video_cams):
            img = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")

    @staticmethod
    def save_ply(gaussian, save_ply_path):
        os.makedirs(save_ply_path, exist_ok=True)
        save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
        print("Model saved in", save_path)
        gaussian.save_ply(save_path)
    
    @staticmethod
    def close():
        pass
