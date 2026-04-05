import copy
import math
import os
import traceback
from typing import List
import imageio
import numpy as np
import torch
import torch.nn
from torchvision.transforms.functional import to_tensor
from torchvision.ops import masks_to_boxes
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from compression.compression_exp import run_single_decompression
from third_party.gaussiansplatting.gaussian_renderer import render_simple,render, render_drag
from third_party.gaussiansplatting.utils.graphics_utils import fov2focal
from third_party.gaussiansplatting.scene import GaussianModel
from third_party.gaussiansplatting.scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from utils.dict_utils import EasyDict
from utils.path_utils import resolve_runtime_subdir, resolve_sam2_paths
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from threestudio.utils.dpt import DPT
from threestudio.utils.transform import rotate_gaussians, scale_gaussians, translate_gaussians
from threestudio.utils.transform import default_model_mtx
from threestudio.utils.misc import get_device
from skimage.draw import line_aa

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class GaussianRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self.concat_gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0

        self._sam2_model_cfg, self._sam2_checkpoint = resolve_sam2_paths(__file__)
        self._runtime_video_dir = resolve_runtime_subdir(
            __file__, "videos", create=True
        )
        self._runtime_add_cache_dir = resolve_runtime_subdir(
            __file__, "cache", "add", create=True
        )
        self.sam_predictor = None
        self.positive_point3d = []
        self.negative_point3d = []
        self.center = torch.tensor((0.0, 0.0, 0.0))

    def _ensure_sam_predictor(self):
        if self.sam_predictor is not None:
            return self.sam_predictor

        checkpoint = Path(self._sam2_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint}\n"
                "Please place `sam2.1_hiera_large.pt` there or set "
                "`SAM2_CHECKPOINT=/abs/path/sam2.1_hiera_large.pt`."
            )

        self.sam_predictor = SAM2ImagePredictor(
            build_sam2(self._sam2_model_cfg, str(checkpoint))
        )
        return self.sam_predictor
    
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

    def add_drag_point(self, image, source_points, target_points, intrinsic, cam_params, selective_keypoints_idx_list):
        if len(source_points) > 0:
            try:
                buffer_overlay = np.zeros_like(image.cpu().numpy()).transpose(1, 2, 0)
                H = image.shape[1]
                W = image.shape[2]

                source_points = np.array(source_points)
                target_points = np.array(target_points)
                
                points_indices = np.arange(len(source_points))

                source_points_2d = self.project_3d_to_2d(source_points, intrinsic, cam_params).round().astype(np.int32)
                target_points_2d = self.project_3d_to_2d(target_points, intrinsic, cam_params).round().astype(np.int32)

                radius = int((H + W) / 2 * 0.005)
                keypoint_idxs_to_drag = selective_keypoints_idx_list
                for i in range(len(source_points_2d)):
                    point_idx = points_indices[i]
                    # draw source point
                    if source_points_2d[i, 0] >= radius and source_points_2d[i, 0] < W - radius and source_points_2d[i, 1] >= radius and source_points_2d[i, 1] < H - radius:
                        buffer_overlay[source_points_2d[i, 1]-radius:source_points_2d[i, 1]+radius, source_points_2d[i, 0]-radius:source_points_2d[i, 0]+radius] += np.array([1,0,0]) if not point_idx in keypoint_idxs_to_drag else np.array([1,0.87,0])
                        # draw target point
                        if target_points_2d[i, 0] >= radius and target_points_2d[i, 0] < W - radius and target_points_2d[i, 1] >= radius and target_points_2d[i, 1] < H - radius:
                            buffer_overlay[target_points_2d[i, 1]-radius:target_points_2d[i, 1]+radius, target_points_2d[i, 0]-radius:target_points_2d[i, 0]+radius] += np.array([0,0,1]) if not point_idx in keypoint_idxs_to_drag else np.array([0.5,0.5,1])
                        # draw line
                        rr, cc, val = line_aa(source_points_2d[i, 1], source_points_2d[i, 0], target_points_2d[i, 1], target_points_2d[i, 0])
                        in_canvas_mask = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                        buffer_overlay[rr[in_canvas_mask], cc[in_canvas_mask]] += val[in_canvas_mask, None] * np.array([0,1,0]) if not point_idx in keypoint_idxs_to_drag else np.array([0.5,1,0])
                overlay_mask = buffer_overlay.sum(axis=-1, keepdims=True) == 0
                try:
                    overlay_mask = torch.tensor(overlay_mask, dtype=torch.float32, device=image.device).permute(2, 0, 1)
                    buffer_overlay = torch.tensor(buffer_overlay, dtype=torch.float32, device=image.device).permute(2, 0, 1)
                    image = image * overlay_mask + buffer_overlay
                except:
                    image = image
            except:
                print('Async Fault in Overlay Drawing!')
                buffer_overlay = None
        
        return image

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
        show_image,
        video_cams=[],
        circle_video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        use_splitscreen=False,
        highlight_border=False,
        save_ply_path=None,
        save_concat_ply_path=None,
        slider={},
        roate_point = None,
        drag_point = None,
        showing_overlay = False,
        sam_positive_points = [],
        sam_negative_points = [],
        source_drag_points = [],
        target_drag_points = [],
        selective_keypoints_idx_list = [],
        drag = {},
        concat = False,
        stop_concat = False,
        depth = None,
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
                self._current_ply_file_paths[scene_index] = ply_file_path
                self.center = torch.tensor((0.0, 0.0, 0.0))

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
                self.render_video(self._runtime_video_dir.as_posix(), video_cams, gs)
            
            if len(circle_video_cams) > 0:
                self.render_video(
                    self._runtime_video_dir.as_posix(), circle_video_cams, gs
                )

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
                self.center = torch.tensor(intersection_point).to(torch.float32)
            if drag_point is not None:
                render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
                render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))
                intersection_point = self.pixel_to_3d(drag_point, intrinsic, cam_params, render["depth"].cpu().numpy()[0][int(drag_point[1]),int(drag_point[0])])
                self.p3d = torch.tensor(intersection_point).to(torch.float32)

            R = cam_params.inverse()[:3, :3].T.cpu().numpy()
            T = cam_params.inverse()[:3, 3].cpu().numpy()
            render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, R=R, T=T, extr=cam_params)
            if concat:
                gs_concat = copy.deepcopy(gs)
                self.concat_gaussian_models[scene_index] = self.concat_gaussian(render_cam, depth, background_color, gs_concat)
                del gs_concat
                torch.cuda.empty_cache()
            
            if stop_concat:
                self.concat_gaussian_models[scene_index] = None
                torch.cuda.empty_cache()
            if self.concat_gaussian_models[scene_index] is not None:
                if drag == {}:
                    render = render_simple(viewpoint_camera=render_cam, pc=self.concat_gaussian_models[scene_index], bg_color=background_color.to("cuda"))
                else:
                    render = render_drag(viewpoint_camera=render_cam, pc=self.concat_gaussian_models[scene_index], bg_color=background_color.to("cuda"),
                                         d_xyz=drag.get("xyz"), d_rotation=drag.get("rotation"), d_scaling=drag.get("scaling"), d_opacity=drag.get("opacity"), d_color=drag.get("color"),d_rotation_bias=drag.get("rotation_bias"))
            else:
                if drag == {}:
                    render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))
                else:
                    render = render_drag(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"),
                                         d_xyz=drag.get("xyz"), d_rotation=drag.get("rotation"), d_scaling=drag.get("scaling"), d_opacity=drag.get("opacity"), d_color=drag.get("color"), d_rotation_bias=drag.get("rotation_bias"))
            if render_alpha:
                images.append(render["alpha"])
            elif render_depth:
                images.append(render["depth"] / render["depth"].max())
            elif showing_overlay and source_drag_points != [] and target_drag_points.tolist() != []:
                image = self.add_drag_point(render["render"], source_drag_points, target_drag_points, intrinsic, cam_params, selective_keypoints_idx_list)
                images.append(image)
            else:
                if sam_positive_points != [] or sam_negative_points != []:
                    try:
                        sam_predictor = self._ensure_sam_predictor()
                    except FileNotFoundError as e:
                        res.error = str(e)
                        images.append(render["render"])
                        continue
                    sam_predictor.set_image(to_pil_image(render["render"]))
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

                    masks, scores, _ = sam_predictor.predict(point_coords=point_coords, 
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
            if save_concat_ply_path is not None:
                self.save_concat_ply(self.concat_gaussian_models[scene_index], save_concat_ply_path)
                self.concat_gaussian_models[scene_index] = None
                torch.cuda.empty_cache()

        self._return_image(
            images,
            res,
            normalize=img_normalize,
            use_splitscreen=use_splitscreen,
            highlight_border=highlight_border,
        )

        if show_image == False:
            del res["image"]

        res.mean_xyz = torch.mean(gs.get_xyz, dim=0)
        res.center = self.center
        res.p3d = self.p3d if hasattr(self, 'p3d') else None
        res.std_xyz = torch.std(gs.get_xyz)
        res.cam_params = cam_params
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
    def save_concat_ply(gaussian, save_ply_path):
        print("Model saved in", save_ply_path)
        gaussian.save_ply(save_ply_path)
    
    def concat_gaussian(self, cam, depth, background_color, gaussian):
        cache_dir = self._runtime_add_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # mv_image_dir = os.path.join(cache_dir, "multiview_pred_images")
        # os.makedirs(mv_image_dir, exist_ok=True)
        inpaint_path = (cache_dir / "inpainted.png").as_posix()
        removed_bg_path = (cache_dir / "removed_bg.png").as_posix()
        gs_path = (cache_dir / "inpaint_gs.ply").as_posix()

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
        del depth_estimator
        # ui_utils.vis_depth(estimated_depth.cpu())
        object_center = (bbox[:2] + bbox[2:]) / 2

        fx = fov2focal(cam.FoVx, cam.image_width)
        fy = fov2focal(cam.FoVy, cam.image_height)

        object_center = (
            object_center
            - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
        ) / torch.tensor([fx, fy]).to("cuda")

        with torch.no_grad():
            render_pkg = render_simple(viewpoint_camera=cam, pc=gaussian, bg_color=background_color.to("cuda"))
        rendered_depth = render_pkg["depth"][..., ~object_mask]
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

        if new_object_gaussian is not None:
            gaussian.prune_with_mask()
        scaled_z_in_cam = z_in_cam * depth
        x_in_cam, y_in_cam = (object_center.cuda()) * scaled_z_in_cam
        T_in_cam = torch.stack([x_in_cam, y_in_cam, scaled_z_in_cam], dim=-1)

        bbox = bbox.cuda()
        real_scale = (
            (bbox[2:] - bbox[:2])
            / torch.tensor([fx, fy], device="cuda")
            * scaled_z_in_cam
        )

        new_object_gaussian = GaussianModel(sh_degree=gaussian.max_sh_degree, disable_xyz_log_activation=True)
        new_object_gaussian.load_ply(gs_path)
        # new_object_gaussian._opacity.data = (
        #     torch.ones_like(new_object_gaussian._opacity.data) * 99.99
        # )

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

        np.save(
            (cache_dir / "center_3D.npy").as_posix(),
            torch.mean(new_object_gaussian.get_xyz, dim=0)
            .detach()
            .cpu()
            .float()
            .numpy(),
        )
        gaussian.concat_gaussians(new_object_gaussian)
        del new_object_gaussian
        torch.cuda.empty_cache()
        return gaussian
    
    @staticmethod
    def close():
        pass
