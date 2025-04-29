import copy
import math
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
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class GaussianRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0

        checkpoint = ".cache/models/sam2/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        self.positive_point3d = []
        self.negative_point3d = []
        self.center_point = np.array([0,0,0]).astype('float64')
    
    def project_3d_to_2d(self, points_3d, intrinsic, extrinsic):
        extrinsic = extrinsic.cpu().numpy() if hasattr(extrinsic, 'cpu') else extrinsic
        intrinsic = intrinsic.cpu().numpy() if hasattr(intrinsic, 'cpu') else intrinsic

        points_3d = np.asarray(points_3d)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
        
        if points_3d.shape[1] == 3:
            homogeneous = np.ones((points_3d.shape[0], 1))
            points_homogeneous = np.hstack([points_3d, homogeneous])
        else:
            points_homogeneous = points_3d

        if hasattr(extrinsic, 'cpu'):
            extrinsic = extrinsic.cpu().numpy()
        
        if extrinsic.shape == (4, 4):
            extrinsic_4x4 = extrinsic
        else:
            extrinsic_4x4 = np.eye(4)
            extrinsic_4x4[:3, :] = extrinsic[:3]

        world_to_cam = np.linalg.inv(extrinsic_4x4)

        points_camera = (world_to_cam @ points_homogeneous.T).T

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        z = points_camera[:, 2]
        z = np.where(z == 0, 1e-10, z)
        x = (points_camera[:, 0] / z) * fx + cx
        y = (points_camera[:, 1] / z) * fy + cy

        return np.column_stack((x, y))

    def pixel_to_3d(self, pixel, intrinsic, extrinsic, depth):
        intrinsic = intrinsic.cpu().numpy() if hasattr(intrinsic, 'cpu') else intrinsic
        extrinsic = extrinsic.cpu().numpy() if hasattr(extrinsic, 'cpu') else extrinsic
        
        u, v = pixel
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        x_cam = (u - cx) / fx * depth
        y_cam = (v - cy) / fy * depth
        z_cam = depth
        point_camera = np.array([x_cam, y_cam, z_cam, 1.0]) 
        
        if extrinsic.shape == (4, 4):
            point_world = extrinsic @ point_camera
        else:
            point_world = np.vstack([extrinsic, [0, 0, 0, 1]]) @ point_camera 
        
        return point_world[:3]
    
    def add_star(self, image, center, size=20, alpha=0.8, type="positive"):
        image = image.clone().float()
        
        mask = torch.zeros(image.shape[1], image.shape[2], dtype=torch.float32)
        
        x0, y0 = center
        r = size / 2  
        inner_r = r * 0.382  
        
        points = []
        for i in range(5):
            outer_angle = i * 72 * np.pi / 180
            x_outer = int(x0 + r * np.cos(outer_angle))
            y_outer = int(y0 - r * np.sin(outer_angle))
            
            inner_angle = (i * 72 + 36) * np.pi / 180
            x_inner = int(x0 + inner_r * np.cos(inner_angle))
            y_inner = int(y0 - inner_r * np.sin(inner_angle))
            
            points.append((x_outer, y_outer))
            points.append((x_inner, y_inner))
        
        def fill_polygon(mask, points):
            min_x = max(0, min(p[0] for p in points))
            max_x = min(image.shape[1], max(p[0] for p in points))
            min_y = max(0, min(p[1] for p in points))
            max_y = min(image.shape[2], max(p[1] for p in points))
            
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if point_in_polygon(x, y, points):
                        mask[y, x] = 1.0
        
        def point_in_polygon(x, y, points):
            inside = False
            for i in range(len(points)):
                j = (i - 1) % len(points)
                xi, yi = points[i]
                xj, yj = points[j]
                if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                    inside = not inside
            return inside
        
        fill_polygon(mask, points)
        
        star = torch.zeros_like(image)
        if type == "positive":
            star[1] = mask
        elif type == "negative":
            star[0] = mask 
        
        output = image + alpha * star
        output = torch.clamp(output, 0, 1)
        
        return output

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
        sam_positive_points = [],
        sam_negative_points = [],
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
                render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
                render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))
                intersection_point = self.pixel_to_3d(roate_point, intrinsic, cam_params, render["depth"].cpu().numpy()[0][int(roate_point[1]),int(roate_point[0])])
                gs._xyz = self.gaussian_models[scene_index]._xyz = self.gaussian_models[scene_index]._xyz - torch.from_numpy(intersection_point).to("cuda").to(torch.float32)
                self.center_point += intersection_point

            render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
            render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))
            if render_alpha:
                images.append(render["alpha"])
            elif render_depth:
                images.append(render["depth"] / render["depth"].max())
            else:
                if sam_positive_points != [] or sam_negative_points != []:
                    self.sam_predictor.set_image(to_pil_image(render["render"]))
                    sam_positive_points = np.array(sam_positive_points)
                    sam_negative_points = np.array(sam_negative_points)

                    # positive points
                    if len(self.positive_point3d) > 0:
                        sam_positive_points[:len(self.positive_point3d)] = self.project_3d_to_2d(self.positive_point3d, intrinsic, cam_params)
                    for i in range(len(self.positive_point3d),len(sam_positive_points)):
                        sam_positive_points[i] *= np.array([render_cam.image_height, render_cam.image_width])
                        self.positive_point3d.append(self.pixel_to_3d(sam_positive_points[i], intrinsic, cam_params,
                                                                      render["depth"].cpu().numpy()[0][int(sam_positive_points[i][1]),int(sam_positive_points[i][0])]))
                    
                    # negetive points
                    if len(self.negative_point3d) > 0:
                        sam_negative_points[:len(self.negative_point3d)] = self.project_3d_to_2d(self.negative_point3d, intrinsic, cam_params)
                    for i in range(len(self.negative_point3d),len(sam_negative_points)):
                        sam_negative_points[i] *= np.array([render_cam.image_height, render_cam.image_width])
                        self.negative_point3d.append(self.pixel_to_3d(sam_negative_points[i], intrinsic, cam_params,
                                                                      render["depth"].cpu().numpy()[0][int(sam_negative_points[i][1]),int(sam_negative_points[i][0])]))
                    sam_positive_points = np.empty((0,2)) if sam_positive_points.shape[0] == 0 else sam_positive_points
                    sam_negative_points = np.empty((0,2)) if sam_negative_points.shape[0] == 0 else sam_negative_points
                    positive_label = np.empty((0), dtype=np.int64) if sam_positive_points.shape[0] == 0 else np.array([1] * sam_positive_points.shape[0], dtype=np.int64) 
                    negative_label = np.empty((0), dtype=np.int64) if sam_negative_points.shape[0] == 0 else np.array([0] * sam_negative_points.shape[0], dtype=np.int64)
                    
                    point_coords = np.concatenate((sam_positive_points, sam_negative_points), axis=0)
                    point_labels = np.concatenate((positive_label, negative_label), axis=0) 

                    masks, scores, _ = self.sam_predictor.predict(point_coords=point_coords, 
                                                                  point_labels=point_labels)
                    max_index = np.argmax(scores)
                    best_mask = masks[max_index]
                    image = render["render"]
                    red_mask = torch.zeros_like(image)
                    red_mask[0] = torch.from_numpy(best_mask)

                    # Add green stars
                    for sam_point in sam_positive_points:
                        image = self.add_star(image, sam_point, type="positive")
                    # Add red stars
                    for sam_point in sam_negative_points:
                        image = self.add_star(image, sam_point, type="negative")

                    image = image + 0.5 * red_mask
                    images.append(image)
                else:
                    self.positive_point3d = []
                    self.negative_point3d = []
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
        res.center_point = self.center_point.tolist()
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
