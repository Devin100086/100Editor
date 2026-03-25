import random
from omegaconf import OmegaConf
import torch
import numpy as np
import os
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
import sys
import gc
import shutil

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"gaussiansplatting"))
from threestudio.utils.sam import LangSAMTextSegmentor
from EditorGS.gaussiansplatting.scene import GaussianModel
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from threestudio.utils.typing import *
from threestudio.utils.clip_metrics import *
from argparse import ArgumentParser
from threestudio.utils.misc import (
    get_device,
)
from threestudio.utils.camera import project_3d_to_2d

from EditorGS.gaussiansplatting.scene.camera_scene import CamScene
from transformers import pipeline
from tqdm import tqdm
from torchvision.utils import save_image

try:
    from acc_diff_gaussian_rasterization_editor import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


class BaseTrainer:
    def __init__(self,cfg):
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.use_sparse_adam = cfg.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
        self.edit_text = None if cfg.text_prompt=="" else cfg.text_prompt
        self.edit_train_steps = None if cfg.edit_train_steps==-1 else cfg.edit_train_steps

        # Camera
        self.gs_lr_scaler = cfg.gs_lr_scaler 
        self.color_lr_scaler = cfg.color_lr_scaler
        self.opacity_lr_scaler = cfg.opacity_lr_scaler
        self.scaling_lr_scaler = cfg.scaling_lr_scaler
        self.rotation_lr_scaler = cfg.rotation_lr_scaler
        self.gs_lr_end_scaler = cfg.gs_lr_end_scaler
        self.densify_until_step = 5000
        self.densification_interval = 100
        self.max_densify_percent = 0.01
        self.min_opacity = 0.005
        self.alpha = 0.99
        self.mask_thres = cfg.mask_thres
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
        self.positive_points3d = []
        self.negative_points3d = []
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
            optimizer_type=cfg.optimizer_type,
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
        self.pds = None

        self.ctn_inpaint = None
        self.training = False
        use_original_resolution = eval(cfg.use_original_resolution) 
        if self.colmap_dir is not None:
            if use_original_resolution:
                scene = CamScene(self.colmap_dir, h=-1, w=-1)
            else:
                scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.masks_2D = {}

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        # status
        self.display_semantic_mask = False
        self.display_point_prompt = False

        self.viewer_need_update = False
        self.system_need_update = False
        self.inpaint_again = True
        self.scale_depth = True
        self.clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)

    def _clear_cuda_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @torch.no_grad()
    def render_cameras_list(self, edit_cameras, separate_sh=False):
        origin_frames = []
        depths = []
        for cam in edit_cameras:
            render_pkg = self.render(cam, separate_sh=separate_sh)
            out, depth = render_pkg["comp_rgb"], render_pkg["depth"]
            origin_frames.append(out)
            depths.append(depth)

        return origin_frames, depths

    def render(
        self,
        cam,
        local=False,
        sam=False,
        train=False,
        mask = False,
        separate_sh = False
    ) -> Dict[str, Any]:
        self.gaussian.localize = local
        if mask:
            render_pkg = render(cam, self.gaussian2, self.pipe, self.background_tensor)
        else:
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor, separate_sh=separate_sh)
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
    
    def update_mask(self, edit_cameras, text_prompt = "hat", type = "default") -> None:
        from threestudio.utils.sam import LangSAMTextSegmentor
        lang_sam = LangSAMTextSegmentor().to(get_device())

        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        with torch.inference_mode():
            for cam in tqdm(edit_cameras):
                cur_cam = cam
                if type == "default":
                    this_frame = render(
                        cur_cam, self.gaussian, self.pipe, self.background_tensor, separate_sh=self.use_sparse_adam
                    )["render"]
                else:
                    this_frame = render(
                        cur_cam, self.gaussian2, self.pipe, self.background_tensor, separate_sh=self.use_sparse_adam
                    )["render"]

                mask = lang_sam(this_frame.unsqueeze(0).permute(0, 2, 3, 1), text_prompt)[0].to(get_device())
                masks.append((mask.detach().to("cpu").numpy().astype(np.uint8) * 255))
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask.to(torch.float32))
                del this_frame, mask

        weights /= weights_cnt + 1e-7
        selected_mask = weights > self.mask_thres
        selected_mask = selected_mask[:, 0]
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)
        del lang_sam
        self._clear_cuda_memory()

        return masks, selected_mask
    
    def update_sam_mask_with_point_prompt(
        self, edit_cameras, positive_points3d=None, negative_points3d=None, type = "default"
    ):
        os.makedirs(self.save_mask_tmp, exist_ok=True)
        os.system(f"rm -rf {self.save_mask_tmp}/*")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_checkpoint = "./.cache/models/sam2/sam2.1_hiera_large.pt"
        sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_predictor = SAM2ImagePredictor(build_sam2(sam2_model_cfg, sam2_checkpoint))
        positive_points3d = positive_points3d if positive_points3d is not None else self.positive_points3d
        negative_points3d = negative_points3d if negative_points3d is not None else self.negative_points3d
        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
        with torch.inference_mode():
            for cam in tqdm(edit_cameras):
                cur_cam = cam
                assert len(positive_points3d) > 0
                positive_points2ds = project_3d_to_2d(positive_points3d, cur_cam) if len(positive_points3d) > 0 else np.empty((0,2))
                negative_points2ds = project_3d_to_2d(negative_points3d, cur_cam) if len(negative_points3d) > 0 else np.empty((0,2))
                if type == "default":
                    img = render(cur_cam, self.gaussian, self.pipe, self.background_tensor, separate_sh=self.use_sparse_adam)[
                        "render"
                    ]
                else:
                    img = render(cur_cam, self.gaussian2, self.pipe, self.background_tensor, separate_sh=self.use_sparse_adam)[
                        "render"
                    ]
                sam2_predictor.set_image(
                    np.asarray(to_pil_image(img.detach().to("cpu"))),
                )

                positive_points2ds = np.empty((0,2)) if positive_points2ds.shape[0] == 0 else positive_points2ds
                negative_points2ds = np.empty((0,2)) if negative_points2ds.shape[0] == 0 else negative_points2ds
                positive_label = np.empty((0), dtype=np.int64) if positive_points2ds.shape[0] == 0 else np.array([1] * positive_points2ds.shape[0], dtype=np.int64)
                negative_label = np.empty((0), dtype=np.int64) if negative_points2ds.shape[0] == 0 else np.array([0] * negative_points2ds.shape[0], dtype=np.int64)

                point_coords = np.concatenate((positive_points2ds, negative_points2ds), axis=0)
                point_labels = np.concatenate((positive_label, negative_label), axis=0)

                mask, _, _ = sam2_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=False,
                )
                mask = torch.from_numpy(mask).to(get_device())
                mask_float = mask.to(torch.float32)
                torchvision.utils.save_image(mask_float.unsqueeze(0).to(torch.float16), f"{self.save_mask_tmp}/mask_{cam.image_name}" + ".png")
                self.gaussian.apply_weights(
                    cur_cam, weights, weights_cnt, mask_float
                )
                masks.append((mask.detach().to("cpu").numpy().astype(np.uint8) * 255))
                del img, mask, mask_float

        weights /= weights_cnt + 1e-7
        selected_mask = weights > self.mask_thres
        selected_mask = selected_mask[:, 0]
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)
        del sam2_predictor
        self._clear_cuda_memory()

        return masks, selected_mask
    
    def update_sam2_mask_with_point_prompt(
        self, edit_cameras, positive_sam_points=None, negative_sam_points=None, type = "default"
    ):
        os.makedirs(self.save_mask_tmp, exist_ok=True)
        os.system(f"rm -rf {self.save_mask_tmp}/*")
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        sam2_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
        render_folder = os.path.join(os.path.dirname(self.save_mask_tmp), "render")
        os.makedirs(render_folder, exist_ok=True)
        for name in os.listdir(render_folder):
            file_path = os.path.join(render_folder, name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)
        with torch.inference_mode():
            for i, cam in tqdm(enumerate(edit_cameras)):
                cur_cam = cam
                if type == "default":
                    img = render(cur_cam, self.gaussian, self.pipe, self.background_tensor, separate_sh=self.use_sparse_adam)["render"]
                else:
                    img = render(cur_cam, self.gaussian2, self.pipe, self.background_tensor, separate_sh=self.use_sparse_adam)["render"]
                save_image(img[None], f"{render_folder}/{i+1:05d}" + ".jpg")
                del img

        state = sam2_predictor.init_state(video_path=render_folder)
        sam2_predictor.reset_state(state)

        ann_frame_idx = 0  
        ann_obj_id = 1  

        positive_points2ds = np.empty((0,2)) if positive_sam_points.shape[0] == 0 else positive_sam_points
        negative_points2ds = np.empty((0,2)) if negative_sam_points.shape[0] == 0 else negative_sam_points
        positive_label = np.empty((0), dtype=np.int64) if positive_sam_points.shape[0] == 0 else np.array([1] * positive_sam_points.shape[0], dtype=np.int64) 
        negative_label = np.empty((0), dtype=np.int64) if negative_sam_points.shape[0] == 0 else np.array([0] * negative_sam_points.shape[0], dtype=np.int64)
        
        point_coords = np.concatenate((positive_points2ds, negative_points2ds), axis=0)
        point_labels = np.concatenate((positive_label, negative_label), axis=0) 

        sam2_predictor.add_new_points(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=point_coords,
            labels=point_labels,
        )

        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        with torch.inference_mode():
            for out_frame_idx, _, out_mask_logits in sam2_predictor.propagate_in_video(state):
                if out_frame_idx == 0:
                    continue
                mask = out_mask_logits[0] > 0.0
                mask_float = mask.to(torch.float32)
                cur_cam = edit_cameras[out_frame_idx - 1]
                save_image(mask_float.unsqueeze(0).to(torch.float16), f"{self.save_mask_tmp}/mask_{cur_cam.image_name}" + ".png")
                self.gaussian.apply_weights(
                    cur_cam, weights, weights_cnt, mask_float
                )
                masks.append((mask.detach().to("cpu").numpy().astype(np.uint8) * 255))
                del out_mask_logits, mask, mask_float

        weights /= weights_cnt + 1e-7
        selected_mask = weights > self.mask_thres
        selected_mask = selected_mask[:, 0]
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)
        del sam2_predictor,state
        self._clear_cuda_memory()

        return masks, selected_mask

    def obtain_depth(self,edit_camears):
        pipe = pipeline(task="depth-estimation", model="depth-anything/depth-anything-V2-Base-hf")
        depths = []
        for i, cam in enumerate(edit_camears):
            cur_cam = cam
            this_frame = render(
                cur_cam, self.gaussian2, self.pipe, self.background_tensor
            )["render"]
            this_frame_pil = to_pil_image(this_frame.cpu())
            depth = pipe(this_frame_pil)["depth"]
            depths.append(to_tensor(depth))
        return depths
    
    def sort_the_cameras_idx(self, cams):
        foward_vectos = [cam.R[:, 2] for cam in cams]
        foward_vectos = np.array(foward_vectos)
        cams_center_x = np.array([cam.camera_center[0].item() for cam in cams])
        most_left_vecotr = foward_vectos[np.argmin(cams_center_x)]
        distances = [np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), -1, 1)) for cam in cams]
        sorted_cams = [cam for _, cam in sorted(zip(distances, cams), key=lambda pair: pair[0])]
        reference_axis = np.cross(most_left_vecotr, sorted_cams[1].R[:, 2])
        distances_with_sign = [np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), -1, 1)) if np.dot(reference_axis,  np.cross(most_left_vecotr, cam.R[:, 2])) >= 0 else 2 * np.pi - np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), -1, 1)) for cam in cams]
        
        sorted_cam_idx = [idx for _, idx in sorted(zip(distances_with_sign, range(len(cams))), key=lambda pair: pair[0])]

        return sorted_cam_idx
    
    def update_cameras(self, random_seed=0):
        random.seed(random_seed)
        self.n2n_view_index = random.sample(
            range(0, len(self.train_cameras)),
            min(len(self.train_cameras), self.edit_cam_num),
        )

    def gaussian_blur(self, mask, kernel_size=21, sigma=8.0):
    
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        x = torch.exp(-x**2 / (2 * sigma**2))
        kernel1d = x / x.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.expand(mask.size(1), 1, kernel_size, kernel_size)
        kernel2d = kernel2d.to(mask.device)

  
        blurred_mask = F.conv2d(mask, kernel2d, padding=kernel_size // 2, groups=mask.size(1))
        return blurred_mask

    def compute_clip(self, step, clip_prompt_origin = "a photo of a face of a man", clip_prompt_target = "a photo of a face of vampire"):
        total_cos = 0
        total_sim = 0
        with torch.no_grad():
            for id, cam in enumerate(self.colmap_cameras):
                cur_cam = cam
                out = self.render(cur_cam, train=False, separate_sh=self.use_sparse_adam)["comp_rgb"]
                _, sim, cos_sim, _ = self.clip_metrics(self.origin_frames_eval[id].permute(0, 3, 1, 2), out.permute(0, 3, 1, 2),
                                                clip_prompt_origin, clip_prompt_target)
                total_cos += abs(cos_sim.item())
                total_sim += abs(sim.item())
        print(clip_prompt_origin, clip_prompt_target, "cos:", total_cos / len(self.colmap_cameras), "sim:", total_sim / len(self.colmap_cameras))
        with open("Ours(CLIP_DIRECTION).txt", "a") as f:
            if step == 0:
                f.write(str(total_cos / len(self.colmap_cameras)))
            else:
                f.write(","+str(total_cos / len(self.colmap_cameras)))
        with open("Ours(CLIP).txt", "a") as f:
            if step == 0:
                f.write(str(total_cos / len(self.colmap_cameras)))
            else:
                f.write(","+str(total_sim / len(self.colmap_cameras)))
        return total_sim / len(self.colmap_cameras), total_cos / len(self.colmap_cameras)

    def compute_metric(self, clip_prompt_origin = "a photo of a face of a man", clip_prompt_target = "a photo of a face of Harry Potter"):
        # clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        total_cos = 0
        total_sim = 0
        with torch.no_grad():
            for id, cam in enumerate(self.test_cameras):
                cur_cam = cam
                out = self.render(cur_cam, train=False, separate_sh=self.use_sparse_adam)["comp_rgb"]
                _, sim, cos_sim, _ = self.clip_metrics(self.test_origin_frames[id].permute(0, 3, 1, 2), out.permute(0, 3, 1, 2),
                                                clip_prompt_origin, clip_prompt_target)
                total_cos += abs(cos_sim.item())
                total_sim += abs(sim.item())
        print(clip_prompt_origin, clip_prompt_target, "cos:", total_cos / len(self.test_cameras), "sim:", total_sim / len(self.test_cameras))
        return total_sim / len(self.test_cameras), total_cos / len(self.test_cameras)
