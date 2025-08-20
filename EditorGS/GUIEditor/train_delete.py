from argparse import ArgumentParser
import pickle
from tqdm import tqdm
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from gaussiansplatting.gaussian_renderer import render_simple
from EditorGS.GUIEditor.train_base import BaseTrainer
from EditorGS.GUIEditor.Guidance.DelGuidance import DelGuidance
from torchvision.utils import save_image
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from PIL import Image
from EditorGS.GUIEditor.utils import *
from EditorGS.GUIEditor.Network import EditorNetwork
from threestudio.utils.misc import (
    dilate_mask,
    fill_closed_areas,
)
from threestudio.utils.camera import pixel_to_3d
import datetime

class DeleteTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.edit_cam_num = cfg.edit_cam_num
        self.lambda_l1 = cfg.lambda_l1
        self.lambda_p = cfg.lambda_p
        self.lambda_anchor_color = cfg.lambda_anchor_color
        self.lambda_anchor_geo = cfg.lambda_anchor_geo
        self.lambda_anchor_scale = cfg.lambda_anchor_scale
        self.lambda_anchor_opacity = cfg.lambda_anchor_opacity
        self.per_editing_step = cfg.per_editing_step
        self.edit_begin_step = cfg.edit_begin_step
        self.edit_until_step = cfg.edit_until_step
        self.delete_prompt = cfg.delete_prompt
        self.inpaint_scale = cfg.inpaint_scale
        self.colmap_dir = cfg.colmap_dir
        self.mask_dilate  = cfg.mask_dilate
        self.sam_type = cfg.sam_type
        self.fix_holes = True
        self.output_dir = cfg.output_dir

        self.cameara_update_step = 500
        self.t_max_step = [999, 300, 300, 21]

        self.positive_sam_points = np.load(cfg.positive_sam_points)
        self.negative_sam_points = np.load(cfg.negative_sam_points)

        self.save_mask_tmp =  os.path.join(os.path.dirname(cfg.positive_sam_points),"mask")

        with open(args.camera, 'rb') as f:
            self.cam  = pickle.load(f)
    
    def sample_train_camera(self, colmap_cameras, edit_cam_num):
        total_view_num = len(colmap_cameras)
        random.seed(0)  # make sure same views
        view_index = random.sample(
            range(0, total_view_num),
            min(total_view_num, edit_cam_num),
        )
        edit_cameras = [colmap_cameras[idx] for idx in view_index]

        return edit_cameras

    def delete(self,video):
        now = datetime.datetime.now()
        now = f"Delete@{now.strftime('%Y_%m_%d_%H_%M')}"
        self.output_dir = os.path.join(self.output_dir, now)
        os.makedirs(self.output_dir, exist_ok=True)

        edit_cameras = self.sample_train_camera(self.colmap_cameras,
                                           self.edit_cam_num,
                                          )
        # for cam in self.colmap_cameras:
        #     rgb = self.render(cam, train=True)["comp_rgb"]
        #     rgb[rgb != 0] = 1
        #     print(rgb)


        # from diffusers import (
        #         StableDiffusionControlNetInpaintPipeline,
        #         ControlNetModel,
        #         DDIMScheduler,
        #         StableDiffusionInpaintPipeline
        #     )

        # controlnet = ControlNetModel.from_pretrained(
        #     "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        # )
        # pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-v1-5",
        #     controlnet=controlnet,
        #     torch_dtype=torch.float16,
        # )
        # from threestudio.models.guidance.brushnet_guidance import (
        #             BrushNetGuidance
        #         )
        # from threestudio.models.guidance.inpaint_guidance import (
        #             inpaintingGuidance
        #         )
        
        # self.inpainting = inpaintingGuidance(
        #             OmegaConf.create({"min_step_percent": 0.02,
        #                               "max_step_percent": 0.98,
        #                               "video":video})
        #         )
        # self.inpainting = BrushNetGuidance(
        #             OmegaConf.create({"min_step_percent": 0.02,
        #                               "max_step_percent": 0.98,
        #                               "video":video,})
        #         )
        cur_2D_guidance = None
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-2-inpainting",
        #     torch_dtype=torch.float16,
        # )
        # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        # pipe.enable_model_cpu_offload()

        # self.ctn_inpaint = pipe
        # self.ctn_inpaint.set_progress_bar_config(disable=True)
        # self.ctn_inpaint.safety_checker = None

        random.seed(0)  # make sure same views
        self.n2n_view_index = random.sample(
            range(0, len(self.colmap_cameras)),
            min(len(self.colmap_cameras), self.edit_cam_num),
        )
        self.view_list = self.n2n_view_index
        if self.sam_type == 1:
            self.masks, _ = self.update_mask(self.colmap_cameras, text_prompt=self.delete_prompt)
        elif self.sam_type == 2:
            # self.cam.FoVx = fov / 360 * 2 * np.pi

            positive_points3d = []
            negative_points3d = []

            # positive
            for i, sam_point in enumerate(self.positive_sam_points):
                depth = render(self.cam, self.gaussian, self.pipe ,self.background_tensor)[
                    "depth_3dgs"
                ]
                # depth = render_simple(self.cam[i], gaussian_copy, self.background_tensor)["depth"]
                depth = (1/depth).detach().cpu().numpy()
                sam_point = sam_point * np.array([self.cam.image_width, self.cam.image_height])
                unprojected_points3d = pixel_to_3d(sam_point, self.cam, depth[0][int(sam_point[1]), int(sam_point[0])])
                # point2d = project_3d_to_2d(unprojected_points3d, self.cam[i])
                positive_points3d.append(unprojected_points3d)
            
            # negative
            for i, sam_point in enumerate(self.negative_sam_points):
                depth = render(self.cam, self.gaussian, self.pipe ,self.background_tensor)[
                    "depth_3dgs"
                ]
                # depth = render_simple(self.cam[i], gaussian_copy, self.background_tensor)["depth"]
                depth = (1/depth).detach().cpu().numpy()
                sam_point = sam_point * np.array([self.cam.image_width, self.cam.image_height])
                unprojected_points3d = pixel_to_3d(sam_point, self.cam, depth[0][int(sam_point[1]), int(sam_point[0])])
                # point2d = project_3d_to_2d(unprojected_points3d, self.cam[i])
                negative_points3d.append(unprojected_points3d)

            positive_points3d = np.array(positive_points3d)
            negative_points3d = np.array(negative_points3d)
            self.update_sam_mask_with_point_prompt(self.colmap_cameras, positive_points3d, negative_points3d)

        elif self.sam_type == 3:
            
            self.positive_sam_points = np.empty((0,2)) if self.positive_sam_points.shape[0] == 0 else self.positive_sam_points * np.array([self.cam.image_width, self.cam.image_height])
            self.negative_sam_points = np.empty((0,2)) if self.negative_sam_points.shape[0] == 0 else self.negative_sam_points * np.array([self.cam.image_width, self.cam.image_height])
            render_folder = os.path.join(os.path.dirname(self.save_mask_tmp), "render")
            init_render = render(self.cam, self.gaussian, self.pipe ,self.background_tensor)["render"]
            save_image(init_render[None], f"{render_folder}/{0:05d}" + ".jpg")
            self.update_sam2_mask_with_point_prompt(edit_cameras, 
                                                                    self.positive_sam_points ,
                                                                    self.negative_sam_points)

        # origin_frames = self.render_cameras_list(self.colmap_cameras)
        # num_channels_latents = self.ctn_inpaint.vae.config.latent_channels
        # shape = (
        #     1,
        #     num_channels_latents,
        #     edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
        #     edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
        # )

        # latents = torch.zeros(shape, dtype=torch.float16, device="cuda")

        dist_thres = (
            self.inpaint_scale * self.cameras_extent * self.gaussian.percent_dense
        )
        valid_remaining_idx = self.gaussian.get_near_gaussians_by_mask(
            self.gaussian.mask, dist_thres
        )
        # Prune and update mask to valid_remaining_idx
        self.gaussian.prune_with_mask(new_mask=valid_remaining_idx)

        self.inpaint_2D_mask, origin_frames = self.render_all_view_with_mask(
            self.colmap_cameras
        )

        self.guidance = DelGuidance(
            guidance=cur_2D_guidance,
            gaussian=self.gaussian,
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
        )

        view_index_stack = self.n2n_view_index.copy()
        ema_loss_for_log = 0.0
        network = EditorNetwork(host="127.0.0.1",port=8084)
        for step in tqdm(range(self.edit_train_steps)):
            network.render(self.pipe,self.gaussian,ema_loss_for_log,render,self.background_tensor,step,self.opt)
            if step % self.cameara_update_step == 0 and video:
                self.edit_all_view(update_camera= step >= self.cameara_update_step, global_step=step)

            if not view_index_stack:
                view_index_stack = self.n2n_view_index.copy()
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(self.colmap_cameras[view_index], train=True)["comp_rgb"]
            # depth_rendering = render_pkg["depth"]
            loss = self.guidance(
                rendering,
                origin_frames[view_index],
                self.inpaint_2D_mask[view_index],
                view_index,
                step,
            )
            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return
            
            ema_loss_for_log = self.alpha * ema_loss_for_log + (1-self.alpha) * loss.item()
        
        self.gaussian.save_ply(f"{self.output_dir}/result.ply")

    @torch.no_grad()
    def render_all_view_with_mask(self, edit_cameras):
        inpaint_2D_mask = []
        origin_frames = []

        for _, cam in enumerate(edit_cameras):
            res = self.render(cam)
            rgb, mask = res["comp_rgb"], res["masks"]
            mask = dilate_mask(mask.to(torch.float32), self.mask_dilate)
            if self.fix_holes:
                mask = fill_closed_areas(mask)
            inpaint_2D_mask.append(mask.to(torch.float32))
            origin_frames.append(rgb)

        return inpaint_2D_mask, origin_frames
    
    def edit_all_view(self, update_camera=False, global_step=0):
    
        self.edited_cams = []
        if update_camera:
            self.update_cameras(random_seed = global_step + 1)
            self.view_list = self.n2n_view_index

        cameras = []
        images = []
        masks = []

        self.guidance.guidance.max_step = self.t_max_step[min(len(self.t_max_step)-1, global_step// self.cameara_update_step)]
        with torch.no_grad():
            for id in self.view_list:
                cameras.append(self.colmap_cameras[id])
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]  
                
            for id in view_sorted:
                cur_cam = self.colmap_cameras[id]
                out_pkg = self.render(cur_cam)
                out = out_pkg["comp_rgb"]
                images.append(out)
                mask = self.inpaint_2D_mask[id].to(torch.float32)
                masks.append(mask)

            images = torch.cat(images, dim=0)
            masks = torch.cat(masks, dim=0)

            edited_images = self.guidance.edit_all(images, masks)

            # save_image(images.permute(0,3,1,2), f'batch_image_{global_step}.png', nrow=4)
            save_image(edited_images.permute(0, 3, 1, 2), f'batch_image_{global_step}.png', nrow=4)
            for view_index_tmp in range(len(self.view_list)):
                self.guidance.edit_frames[view_sorted[view_index_tmp]] = edited_images[view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--edit_cam_num", type=int, default=0, help="Camera number.")
    parser.add_argument("--delete_prompt", type=str, default="man", help="Delete Prompt.")
    parser.add_argument("--text_prompt", type=str, default="", help="text prompt.")
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
    parser.add_argument("--gs_lr_scaler", type=float, default=1.0, help="Initial learning rate scaler for GS.")
    parser.add_argument("--gs_lr_end_scaler", type=float, default=1.0, help="Final learning rate scaler for GS.")
    parser.add_argument("--color_lr_scaler", type=float, default=3.0, help="Learning rate scaler for color.")
    parser.add_argument("--opacity_lr_scaler", type=float, default=2.0, help="Learning rate scaler for opacity.")
    parser.add_argument("--scaling_lr_scaler", type=float, default=2.0, help="Learning rate scaler for scaling.")
    parser.add_argument("--rotation_lr_scaler", type=float, default=2.0, help="Learning rate scaler for rotation.")
    parser.add_argument("--inpaint_scale", type=float, default=1.0, help="Inpaint scale.")
    parser.add_argument("--mask_dilate", type=int, default=15, help="Mask dilate.")
    parser.add_argument("--video", type=str, default="False", help="video.")
    parser.add_argument("--sam_type", type=int, default=0, help="sam type.")
    parser.add_argument("--positive_sam_points", type=str, default="/", help="the path of the positive sam points.")
    parser.add_argument("--negative_sam_points", type=str, default="/", help="the path of the negative sam points.")
    parser.add_argument("--camera", type=str, default="tmd_delete/camera.pkl", help="camera.")
    parser.add_argument("--use_original_resolution", type=str, default="False", help="use original resolution.")
    parser.add_argument("--output_dir", type=str, default="save/", help="output dir.")
    parser.add_argument("--mask_thres", type=float, default=0.5, help="mask threshold.")

    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = DeleteTrainer(args)
        trainer.gaussian.update_anchor_term(
            anchor_weight_init_g0=0.05,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=1.3,
        )

        trainer.configure_optimizers()
        trainer.delete(video=eval(args.video))