from argparse import ArgumentParser
import copy
import pickle
import time
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os

from editor.hundrededitor_gui.base import *
from editor.hundrededitor_gui.guidance.edit_guidance import EditGuidance
from editor.gaussiansplatting.gaussian_renderer import render
from editor.hundrededitor_gui.utils import *
from editor.hundrededitor_gui.Network import EditorNetwork
from threestudio.utils.camera import pixel_to_3d
from threestudio.models.guidance.brushnet_guidance import BrushNetGuidance
from torchvision.utils import save_image
import datetime
from src.utils.path_utils import resolve_runtime_subdir

class EditTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.origin_prompt = cfg.origin_prompt
        self.edit_cam_num = cfg.edit_cam_num
        self.guidance_type = cfg.guidance_type
        self.lambda_l1 = cfg.lambda_l1
        self.lambda_p = cfg.lambda_p
        self.lambda_anchor_color = cfg.lambda_anchor_color
        self.lambda_anchor_geo = cfg.lambda_anchor_geo
        self.lambda_anchor_scale = cfg.lambda_anchor_scale
        self.lambda_anchor_opacity = cfg.lambda_anchor_opacity
        self.per_editing_step = cfg.per_editing_step
        self.edit_begin_step = cfg.edit_begin_step
        self.edit_until_step = cfg.edit_until_step
        self.output_dir = cfg.output_dir
        self.earlystop = eval(cfg.earlystop)
        self.cps_patience_counter = cfg.cps_patience_counter
        self.cps_patience = cfg.cps_patience
        self.cps_batch_count = cfg.cps_batch_count
        self.clip_origin_prompt = cfg.clip_origin_prompt
        self.clip_target_prompt = cfg.clip_target_prompt
        self.eval = cfg.eval

        self.t_max_step = [999, 300, 300, 21]
        self.cameara_update_step = 500
        self.use_masked_image = False

        self.positive_sam_points = np.load(cfg.positive_sam_points)
        self.negative_sam_points = np.load(cfg.negative_sam_points)

        self.hard_segmentation = eval(cfg.hard_segmentation)

        self.save_mask_tmp = resolve_runtime_subdir(
            __file__, "cache", "edit", "mask", create=True
        ).as_posix()

        with open(cfg.camera, 'rb') as f:
            self.cam  = pickle.load(f)

    @staticmethod
    def _supports_ansi() -> bool:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @classmethod
    def _ansi(cls, text: str, code: str) -> str:
        if not cls._supports_ansi():
            return text
        return f"\033[{code}m{text}\033[0m"

    def _log_kv(self, label: str, value: str, color: str = "36") -> None:
        print(self._ansi(f"  {label:<16}:", color), value)

    def _log_stage(self, message: str, color: str = "1;34") -> None:
        print(self._ansi(f"[Semantic Edit] {message}", color))

    def _log_start_banner(self, sam_option: int, seg_prompt: str, video: bool) -> None:
        sam_map = {
            0: "No Sam",
            1: "Lang-sam",
            2: "SAM2(image)",
            3: "SAM2(video)",
        }
        print("\n" + self._ansi("[Semantic Edit] Start", "1;36"))
        self._log_kv("Guidance", str(self.guidance_type))
        self._log_kv("Prompt", str(self.edit_text))
        self._log_kv("Train Steps", str(self.edit_train_steps))
        self._log_kv("Batch Mode", str(video))
        self._log_kv("SAM Mode", sam_map.get(sam_option, f"Unknown({sam_option})"))
        if sam_option == 1:
            self._log_kv("Seg Prompt", seg_prompt)
        self._log_kv("Output Dir", self.output_dir)

    def _log_finish_banner(self, wall_time_s: float, result_ply_path: str) -> None:
        print(self._ansi("[Semantic Edit] Done", "1;32"))
        self._log_kv("Elapsed", f"{wall_time_s:.2f}s", color="32")
        self._log_kv("Result PLY", result_ply_path, color="32")
        print("")

    def _log_batch_mode_config(self) -> None:
        self._log_stage("Batch Mode Enabled", color="1;35")
        self._log_kv("Camera Update", f"every {self.cameara_update_step} steps", color="35")
        self._log_kv(
            "Batch Views",
            f"{len(self.n2n_view_index)} / {len(self.train_cameras)} train views",
            color="35",
        )

    def _wait_for_user_finalize(self, network, ema_loss_for_log, step):
        network.render(
            self.pipe,
            self.gaussian,
            ema_loss_for_log,
            render,
            self.background_tensor,
            step,
            self.opt,
            self.use_sparse_adam,
            force_pause=True,
        )

    def edit(self, sam_option, seg_prompt, video):
        wall_start = time.perf_counter()
        now = datetime.datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.output_dir = os.path.join(self.output_dir, now)
        os.makedirs(self.output_dir, exist_ok=True)
        self._log_start_banner(sam_option=sam_option, seg_prompt=seg_prompt, video=video)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()

        # edit_cameras = sample_train_camera(self.colmap_cameras,
        #                                    self.edit_cam_num,
        #                                   )
        if self.guidance_type == "InstructPix2Pix":
            if not self.ip2p:
                from threestudio.models.guidance.instructpix2pix_guidance import (
                    InstructPix2PixGuidance,
                )

                self.ip2p = InstructPix2PixGuidance(
                    OmegaConf.create({"min_step_percent": 0.02,
                                      "max_step_percent": 0.98,
                                      "video":video})
                )

            cur_2D_guidance = self.ip2p
            # cur_2D_guidance = self.dreambooth
            self.origin_prompt = None
            self._log_stage("Guidance Loaded: InstructPix2Pix")
        elif self.guidance_type == "ControlNet-Depth":
            if not self.ctn_ip2p:
                from threestudio.models.guidance.controlnet_guidance import (
                    ControlNetGuidance,
                )

                self.ctn_controlnet = ControlNetGuidance(
                    OmegaConf.create({"min_step_percent": 0.02,
                                      "max_step_percent": 0.98,
                                      "video":video,
                                      "control_type": "depth"})
                )
            cur_2D_guidance = self.ctn_controlnet
            self._log_stage("Guidance Loaded: ControlNet-Depth")
        elif self.guidance_type == "BrushNet":
            self.brushnet = BrushNetGuidance(
                OmegaConf.create({"min_step_percent": 0.02,
                                    "max_step_percent": 0.98,
                                    "video": video})
            )
            self.origin_prompt = None
            cur_2D_guidance = self.brushnet
            self._log_stage("Guidance Loaded: BrushNet")
        
        if self.eval:
            self.origin_frames_eval, self.depths_eval = self.render_cameras_list(self.colmap_cameras, separate_sh=self.use_sparse_adam)

        if self.earlystop:
            num_test_views = min(6, len(self.colmap_cameras))
            self.test_view_indices = random.sample(range(len(self.colmap_cameras)), num_test_views)
            # self.train_view_indices = [i for i in range(len(self.colmap_cameras)) if i not in self.test_view_indices]
            self.test_cameras = [self.colmap_cameras[i] for i in self.test_view_indices]
            self.test_origin_frames, self.test_depths = self.render_cameras_list(self.test_cameras, separate_sh=self.use_sparse_adam)
            self.train_cameras = [cam for i, cam in enumerate(self.colmap_cameras) if i not in self.test_view_indices]
        else:
            self.train_cameras = self.colmap_cameras
        
        self.origin_frames, self.depths = self.render_cameras_list(self.train_cameras, separate_sh=self.use_sparse_adam)

        random.seed(0)  # make sure same views
        self.n2n_view_index = random.sample(
            range(0, len(self.train_cameras)),
            min(len(self.train_cameras), self.edit_cam_num),
        )

        self.view_list = self.n2n_view_index
        if video:
            self._log_batch_mode_config()

        if sam_option == 0:
            self._log_stage("SAM disabled")
        elif sam_option == 1:
            self._log_stage(f"SAM segmentation with prompt: {seg_prompt}")
            self.masks, _ = self.update_mask(self.train_cameras, text_prompt=seg_prompt)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        elif sam_option == 2:
            self._log_stage(
                f"SAM2(image) with points: +{len(self.positive_sam_points)} / -{len(self.negative_sam_points)}"
            )

            positive_points3d = []
            negative_points3d = []
            with torch.inference_mode():
                depth = render(self.cam, self.gaussian, self.pipe, self.background_tensor, separate_sh=self.use_sparse_adam)[
                    "depth_3dgs"
                ]
                depth_np = (1 / depth).detach().cpu().numpy()

            # positive
            for sam_point in self.positive_sam_points:
                sam_point = sam_point * np.array([self.cam.image_width, self.cam.image_height])
                unprojected_points3d = pixel_to_3d(
                    sam_point,
                    self.cam,
                    depth_np[0][int(sam_point[1]), int(sam_point[0])],
                )
                positive_points3d.append(unprojected_points3d)

            # negative
            for sam_point in self.negative_sam_points:
                sam_point = sam_point * np.array([self.cam.image_width, self.cam.image_height])
                unprojected_points3d = pixel_to_3d(
                    sam_point,
                    self.cam,
                    depth_np[0][int(sam_point[1]), int(sam_point[0])],
                )
                negative_points3d.append(unprojected_points3d)
            
            positive_points3d = np.array(positive_points3d)
            negative_points3d = np.array(negative_points3d)
            self.masks, _ = self.update_sam_mask_with_point_prompt(self.train_cameras, positive_points3d, negative_points3d)

            del depth, depth_np, positive_points3d, negative_points3d
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._clear_cuda_memory()

        elif sam_option == 3:
            self._log_stage(
                f"SAM2(video) with points: +{len(self.positive_sam_points)} / -{len(self.negative_sam_points)}"
            )

            positive_sam_points = np.empty((0,2)) if self.positive_sam_points.shape[0] == 0 else self.positive_sam_points * np.array([self.cam.image_width, self.cam.image_height])
            negative_sam_points = np.empty((0,2)) if self.negative_sam_points.shape[0] == 0 else self.negative_sam_points * np.array([self.cam.image_width, self.cam.image_height])
            self.masks, _ = self.update_sam2_mask_with_point_prompt(self.train_cameras, 
                                                                    positive_sam_points ,
                                                                   negative_sam_points,
                                                                   prompt_camera=self.cam)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._clear_cuda_memory()
        
        self.guidance = EditGuidance(
            guidance=cur_2D_guidance,
            guidance_type = self.guidance_type,
            gaussian=self.gaussian,
            origin_frames=self.origin_frames,
            depths = self.depths,
            text_prompt=self.edit_text,
            per_editing_step=self.per_editing_step,
            edit_begin_step=self.edit_begin_step,
            edit_until_step=self.edit_until_step,
            lambda_l1=self.lambda_l1,
            lambda_p=self.lambda_p,
            lambda_anchor_color=self.lambda_anchor_color,
            lambda_anchor_geo=self.lambda_anchor_geo,
            lambda_anchor_scale=self.lambda_anchor_scale,
            lambda_anchor_opacity=self.lambda_anchor_opacity,
            cams=self.colmap_cameras,
            origin_text_prompt=self.origin_prompt,
        )
        view_index_stack = self.n2n_view_index.copy()
        ema_loss_for_log = 0.0

        # Renderings1 = []
        # Renderings2 = []
        best_metric = float('-inf')
        max_delta = 0.006
        initial_patience_counter = max(0, int(self.cps_patience_counter))
        patience_counter = initial_patience_counter
        patience = max(1, int(self.cps_patience))
        max_batch_count = max(1, int(self.cps_batch_count))
        Batch_flag = False
        Batch_count = 0

        network = EditorNetwork(host="127.0.0.1",port=8084)
        start_event.record()
        self._log_stage("Training loop started")
        
        # 创建时间记录文件
        time_log_path = os.path.join(self.output_dir, "time_log.txt")
        memory_log_path = os.path.join(self.output_dir, "memory_log.txt")
        earlystop_log_path = os.path.join(self.output_dir, "earlystop_log.txt")
        training_start_time = torch.cuda.Event(enable_timing=True)
        step_end_time = torch.cuda.Event(enable_timing=True)
        training_start_time.record()
        
        # 初始化显存监控
        max_memory_allocated = 0.0
        max_memory_reserved = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # 创建保存相机45渲染结果的目录
        # camera_45_dir = os.path.join(self.output_dir, "camera_45_renders")
        # os.makedirs(camera_45_dir, exist_ok=True)
        
        progress_bar = tqdm(
            range(self.edit_train_steps),
            desc="Semantic Train (Batch)" if video else "Semantic Train",
            dynamic_ncols=True,
        )
        for step in progress_bar:
            network.render(self.pipe,self.gaussian,ema_loss_for_log, render,self.background_tensor,step,self.opt, self.use_sparse_adam)
            
            # 保存相机索引45的渲染结果
            # if len(self.colmap_cameras2) > 45:
            #     with torch.no_grad():
            #         camera_45_render = self.render(self.colmap_cameras2[45], train=False, separate_sh=self.use_sparse_adam)["comp_rgb"]
            #         save_image(camera_45_render.permute(0,3,1,2), os.path.join(camera_45_dir, f"step_{step:05d}.png"))
            
            # 更新显存峰值
            if torch.cuda.is_available():
                current_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为GB
                current_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)  # 转换为GB
                max_memory_allocated = max(max_memory_allocated, current_memory_allocated)
                max_memory_reserved = max(max_memory_reserved, current_memory_reserved)
            
            # if step % 50 == 0 :
            #     if step == 0:
            #         elapsed_time_s = 0.0
            #     else:
            #         step_end_time.record()
            #         torch.cuda.synchronize()
            #         elapsed_time_ms = training_start_time.elapsed_time(step_end_time)
            #         elapsed_time_s = elapsed_time_ms / 1000
                
            #     with open(time_log_path, 'a') as f:
            #         f.write(f"{elapsed_time_s:.4f},")
            
            if self.earlystop:
                if step % 20 == 0:
                    _, metric = self.compute_metric(clip_prompt_origin=self.clip_origin_prompt, clip_prompt_target=self.clip_target_prompt)
                    if metric > best_metric + max_delta:
                        best_metric = metric
                        patience_counter = initial_patience_counter
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        patience_counter = initial_patience_counter
                        with open(earlystop_log_path, "a") as f:
                            f.write(f"Early stopping at step {step} with best metric {best_metric:.4f}\n")
                        self._log_stage(
                            f"Early-stop trigger at step {step} (best metric={best_metric:.4f})",
                            color="33",
                        )
                        Batch_flag = True
                        continue
                if ((step == 0 or Batch_flag == True) and video):
                    if Batch_count >= max_batch_count:
                        with open(earlystop_log_path, "a") as f:
                            f.write(f"Finish all batches at step {step}.\n")
                        self._log_stage("Early-stop batches complete, waiting for user finalize", color="33")
                        self._wait_for_user_finalize(network, ema_loss_for_log, step)
                        break
                    batch_idx = Batch_count + 1
                    self._log_stage(
                        f"Batch {batch_idx}/{max_batch_count} full-view update at step {step}",
                        color="1;35",
                    )
                    self.edit_all_view(sam_option, update_camera= step >= self.cameara_update_step, global_step=step)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    Batch_flag = False
                    Batch_count += 1
                    self._log_stage(
                        f"Batch {batch_idx}/{max_batch_count} update complete",
                        color="35",
                    )
            else:
                if (step % self.cameara_update_step == 0 and video):
                    batch_round = step // self.cameara_update_step + 1
                    self._log_stage(
                        f"Batch refresh #{batch_round} at step {step}",
                        color="1;35",
                    )
                    self.edit_all_view(sam_option, update_camera= step >= self.cameara_update_step, global_step=step)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._log_stage(
                        f"Batch refresh #{batch_round} complete",
                        color="35",
                    )

            # if (step == 999):
            #     self.edit_all_view(sam_option, update_camera= False, global_step=step)

            if not view_index_stack:
                view_index_stack = self.n2n_view_index.copy()
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(self.train_cameras[view_index], train=True, separate_sh=self.use_sparse_adam)["comp_rgb"]
            # import torchvision.utils as vutils
            # vutils.save_image(rendering.permute(0,3,1,2), "output_images.png", nrow=1)
            # if (step+1) % 250 == 0:
            #     image1 = self.render(self.colmap_cameras[5],separate_sh=self.use_sparse_adam)["comp_rgb"]
            #     image2 = self.render(self.colmap_cameras[10],separate_sh=self.use_sparse_adam)["comp_rgb"]
            #     Renderings1.append(image1.permute(0,3,1,2).cpu())
            #     Renderings2.append(image2.permute(0,3,1,2).cpu())
            
            loss = self.guidance(rendering, view_index, step)

            loss.backward()

            self.densify_and_prune(step)

            if self.use_sparse_adam:
                    visible = self.radii > 0
                    self.gaussian.optimizer.step(visible, self.radii.shape[0])
                    self.gaussian.optimizer.zero_grad(set_to_none=True)
            else:
                self.gaussian.optimizer.step()
                self.gaussian.optimizer.zero_grad(set_to_none=True)

            if self.stop_training:
                self.stop_training = False
                return
            if ema_loss_for_log == 0:
                ema_loss_for_log = loss.item()
            else:
                ema_loss_for_log = self.alpha * ema_loss_for_log + (1-self.alpha) * loss.item()
            if step % 10 == 0:
                postfix_data = {
                    "loss": f"{ema_loss_for_log:.4f}",
                    "mem_gb": f"{max_memory_allocated:.2f}",
                }
                if video and self.earlystop:
                    postfix_data["batch"] = f"{Batch_count}/{max_batch_count}"
                progress_bar.set_postfix(postfix_data)
        
        # Renderings1 = torch.cat(Renderings1, dim=0)
        # Renderings2 = torch.cat(Renderings2, dim=0)
        # save_image(Renderings1, f"batch_image1_{self.edit_train_steps}.png", nrow=Renderings1.shape[0])
        # save_image(Renderings2, f"batch_image2_{self.edit_train_steps}.png", nrow=Renderings2.shape[0])

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)

        elapsed_time_s = elapsed_time_ms / 1000

        hours, remainder = divmod(elapsed_time_s, 3600)
        minutes, seconds = divmod(remainder, 60)
        with open(time_log_path, "a") as f:
            f.write(f"Time: {hours} h {minutes} min {seconds:.2f} s\n")
        
        # 保存显存信息到文件
        with open(memory_log_path, "a") as f:
            f.write(f"Max Memory Allocated: {max_memory_allocated:.4f} GB\n")
            f.write(f"Max Memory Reserved: {max_memory_reserved:.4f} GB\n")

        if self.eval:
            self.compute_clip(self.edit_train_steps, clip_prompt_origin=self.clip_origin_prompt, clip_prompt_target=self.clip_target_prompt)

        result_ply_path = f"{self.output_dir}/result.ply"
        self.gaussian.save_ply(result_ply_path)
        wall_elapsed = time.perf_counter() - wall_start
        self._log_finish_banner(wall_time_s=wall_elapsed, result_ply_path=result_ply_path)
    
    def edit_all_view(self, sam_option, update_camera=False, global_step=0):
    
        self.edited_cams = []
        if update_camera:
            self.update_cameras(random_seed = global_step + 1)
            self.view_list = self.n2n_view_index

        cameras = []
        images = []
        origin_frames = []
        depths = []
        masks = []
        brushnet_masks = []

        self.guidance.guidance.max_step = self.t_max_step[min(len(self.t_max_step)-1, global_step// self.cameara_update_step)]
        with torch.no_grad():
            for id in self.view_list:
                cameras.append(self.train_cameras[id])
            if len(cameras) == 1:
                sorted_cam_idx = [0]
            else:
                sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            # sorted_cam_idx = range(len(cameras))
            view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]  
                
            for id in view_sorted:
                cur_cam = self.train_cameras[id]
                out_pkg = self.render(cur_cam, separate_sh=self.use_sparse_adam)
                out = out_pkg["comp_rgb"]
                if self.use_masked_image:
                    out = out * out_pkg["masks"].unsqueeze(-1)
                images.append(out)
                if sam_option != 0:
                    if isinstance(self.masks[id], np.ndarray):
                        mask = torch.from_numpy(self.masks[id] / 255.0)
                    else:
                        mask = self.masks[id].to(torch.float32)
                        if torch.max(mask) > 1:
                            mask = mask / 255.0
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0).unsqueeze(0)
                    elif mask.ndim == 3:
                        mask = mask.unsqueeze(0)
                    mask = mask.to(torch.float32).to(get_device())
                    mask_blur = self.gaussian_blur(mask)
                    masks.append(mask_blur)
                    if self.guidance_type == "BrushNet":
                        import torch.nn.functional as F
                        kernel_size = 25 
                        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device) / (kernel_size * kernel_size)
                        dilated_mask = F.conv2d(mask.float(), kernel, padding=kernel_size//2)
                        dilated_mask = (dilated_mask > 0.02).float() 
                        brushnet_masks.append(dilated_mask)
                origin_frames.append(self.origin_frames[id])
                depths.append(self.depths[id])

            images = torch.cat(images, dim=0)
            if sam_option != 0:
                masks = torch.cat(masks, dim=0)
                masks = masks.permute(0, 2, 3, 1) # B H W C
                if self.guidance_type == "BrushNet":
                    brushnet_masks = torch.cat(brushnet_masks, dim=0)
                    brushnet_masks = brushnet_masks.permute(0, 2, 3, 1) # B H W C
            else:
                masks = torch.ones_like(images)
                brushnet_masks = torch.ones_like(images)
            origin_frames = torch.cat(origin_frames, dim=0)
            depths = torch.cat(depths, dim=0)
            
            if self.guidance_type == "InstructPix2Pix":
                edited_images = self.guidance.edit_all(images, origin_frames)
            elif self.guidance_type == "ControlNet-Depth":
                edited_images = self.guidance.edit_all(images, depths)
            elif self.guidance_type == "BrushNet":
                brushnet_masks = brushnet_masks.repeat(1, 1, 1, 3)
                images = images * (1 - brushnet_masks)
                edited_images = self.guidance.edit_all(images, brushnet_masks)
                edited_images = edited_images * brushnet_masks + (1-brushnet_masks) * origin_frames
            if self.hard_segmentation and self.guidance_type != "BrushNet":
                edited_images = edited_images * masks + (1-masks) * origin_frames

            # save_image(edited_images.permute(0, 3, 1, 2), f'{self.output_dir}/batch_image_{global_step}.png', nrow=10)
            for view_index_tmp in range(len(self.view_list)):
                self.guidance.edit_frames[view_sorted[view_index_tmp]] = edited_images[view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C


if __name__ == "__main__":
    default_delete_cache_dir = resolve_runtime_subdir(__file__, "cache", "delete", create=True)
    default_positive = (default_delete_cache_dir / "sam2_positive_points.npy").as_posix()
    default_negative = (default_delete_cache_dir / "sam2_negative_points.npy").as_posix()
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--edit_cam_num", type=int, default=0, help="Camera number.")
    parser.add_argument("--optimizer_type", type=str, default="sparse_adam", help="default or sparse_adam")
    parser.add_argument("--guidance_type", type=str, default="InstructPix2Pix")
    parser.add_argument("--text_prompt", default="default_text", help="Text prompt.")
    parser.add_argument("--origin_prompt", default="default_origin_text", help="Origin text prompt.")
    parser.add_argument("--edit_train_steps", type=int, default=1500, help="Edit train steps.")
    parser.add_argument("--per_train_step", type=int, default=1, help="Per train step.")
    parser.add_argument("--per_editing_step", type=int, default=1, help="Per editing step.")
    parser.add_argument("--edit_begin_step", type=int, default=0, help="Edit begin step.")
    parser.add_argument("--edit_until_step", type=int, default=100, help="Edit until step.")
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="Lambda L1.")
    parser.add_argument("--lambda_p", type=int, default=2, help="Lambda P.")
    parser.add_argument("--lambda_anchor_color", type=float, default=1.0, help="Lambda anchor color.")
    parser.add_argument("--lambda_anchor_geo", type=float, default=1.0, help="Lambda anchor geo.")
    parser.add_argument("--lambda_anchor_scale", type=float, default=1.0, help="Lambda anchor scale.")
    parser.add_argument("--lambda_anchor_opacity", type=float, default=1.0, help="Lambda anchor opacity.")
    parser.add_argument("--sam_option", type=int, default=-1, help="Sam Option.")
    parser.add_argument("--seg_prompt", type=str, default="face", help="seg Prompt.")
    parser.add_argument("--gs_lr_scaler", type=float, default=1.0, help="Initial learning rate scaler for GS.")
    parser.add_argument("--gs_lr_end_scaler", type=float, default=1.0, help="Final learning rate scaler for GS.")
    parser.add_argument("--color_lr_scaler", type=float, default=3.0, help="Learning rate scaler for color.")
    parser.add_argument("--opacity_lr_scaler", type=float, default=2.0, help="Learning rate scaler for opacity.")
    parser.add_argument("--scaling_lr_scaler", type=float, default=2.0, help="Learning rate scaler for scaling.")
    parser.add_argument("--rotation_lr_scaler", type=float, default=2.0, help="Learning rate scaler for rotation.")
    parser.add_argument("--video", type=str, default="False", help="video.")
    parser.add_argument("--use_original_resolution", type=str, default="False", help="use original resolution.")
    parser.add_argument("--positive_sam_points", type=str, default=default_positive, help="the path of the positive sam points.")
    parser.add_argument("--negative_sam_points", type=str, default=default_negative, help="the path of the negative sam points.")
    parser.add_argument(
        "--camera",
        type=str,
        default=(default_delete_cache_dir / "camera.pkl").as_posix(),
        help="camera.",
    )
    parser.add_argument("--output_dir", type=str, default="save/", help="output dir.")
    parser.add_argument("--hard_segmentation", type=str, default="True", help="hard segmentation.")
    parser.add_argument("--mask_thres", type=float, default=0.8, help="mask threshold.")
    parser.add_argument("--earlystop", type=str, default="True", help="Early stopping.")
    parser.add_argument("--cps_patience_counter", type=int, default=0, help="CPS patience counter.")
    parser.add_argument("--cps_patience", type=int, default=4, help="CPS patience.")
    parser.add_argument("--cps_batch_count", type=int, default=3, help="CPS batch count.")
    parser.add_argument("--clip_origin_prompt", type=str, default="a photo of a face of a man", help="Clip origin prompt.")
    parser.add_argument("--clip_target_prompt", type=str, default="a photo of a face of a Kevin Durant", help="Clip target prompt.")
    parser.add_argument("--eval", type=bool, default=False, help="Eval or not.")

    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = EditTrainer(args)
        trainer.gaussian.update_anchor_term(
            anchor_weight_init_g0=0.05,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=1.3,
        )
        trainer.configure_optimizers()
        trainer.edit(sam_option=args.sam_option, seg_prompt=args.seg_prompt, video=eval(args.video))
