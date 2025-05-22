from argparse import ArgumentParser
import copy
import pickle
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import *
from EditorGS.GUIEditor.Guidance.EditGuidance import EditGuidance
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.GUIEditor.utils import *
from EditorGS.GUIEditor.Network import EditorNetwork
from threestudio.utils.camera import pixel_to_3d

from torchvision.utils import save_image

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
        self.lang_sam = LangSAMTextSegmentor().to(get_device())
        self.edit_begin_step = cfg.edit_begin_step
        self.edit_until_step = cfg.edit_until_step

        self.t_max_step = [999, 300, 300, 21]
        self.cameara_update_step = 500
        self.use_masked_image = False

        self.positive_sam_points = np.load(cfg.positive_sam_points)
        self.negative_sam_points = np.load(cfg.negative_sam_points)

        self.save_mask_tmp =  os.path.join(os.path.dirname(cfg.positive_sam_points),"mask")

        with open(args.camera, 'rb') as f:
            self.cam  = pickle.load(f)

    def edit(self, sam_option, seg_prompt, video):
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
            self.origin_prompt = None
            print("using InstructPix2Pix!")
        elif self.guidance_type == "ControlNet-Depth":
            if not self.ctn_ip2p:
                from threestudio.models.guidance.controlnet_guidance import (
                    ControlNetGuidance,
                )

                self.ctn_ip2p = ControlNetGuidance(
                    OmegaConf.create({"min_step_percent": 0.02,
                                      "max_step_percent": 0.98,
                                      "video":video,
                                      "control_type": "depth"})
                )
            cur_2D_guidance = self.ctn_ip2p
            print("using ControlNet-Depth!")
        
        self.origin_frames, self.depths = self.render_cameras_list(self.colmap_cameras)

        random.seed(0)  # make sure same views
        self.n2n_view_index = random.sample(
            range(0, len(self.colmap_cameras)),
            min(len(self.colmap_cameras), self.edit_cam_num),
        )
        self.view_list = self.n2n_view_index

        if sam_option == 0:
            pass
        elif sam_option == 1:
            self.masks, _ = self.update_mask(self.colmap_cameras, text_prompt=seg_prompt)
        elif sam_option == 2:

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
            self.masks, _ = self.update_sam_mask_with_point_prompt(self.colmap_cameras, positive_points3d, negative_points3d)

        elif sam_option == 3:

            self.positive_sam_points = np.empty((0,2)) if self.positive_sam_points.shape[0] == 0 else self.positive_sam_points * np.array([self.cam.image_width, self.cam.image_height])
            self.negative_sam_points = np.empty((0,2)) if self.negative_sam_points.shape[0] == 0 else self.negative_sam_points * np.array([self.cam.image_width, self.cam.image_height])
            render_folder = os.path.join(os.path.dirname(self.save_mask_tmp), "render")
            init_render = render(self.cam, self.gaussian, self.pipe ,self.background_tensor)["render"]
            save_image(init_render[None], f"{render_folder}/{0:05d}" + ".jpg")
            self.masks, _ = self.update_sam2_mask_with_point_prompt(self.colmap_cameras, 
                                                                    self.positive_sam_points ,
                                                                    self.negative_sam_points)
        
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

        Renderings1 = []
        Renderings2 = []

        network = EditorNetwork(host="127.0.0.1",port=8084)
        for step in tqdm(range(self.edit_train_steps)):
            network.render(self.pipe,self.gaussian,ema_loss_for_log,render,self.background_tensor,step,self.opt)
            if step % self.cameara_update_step == 0 and video:
                self.edit_all_view(sam_option, update_camera= step >= self.cameara_update_step, global_step=step)

            if not view_index_stack:
                view_index_stack = self.n2n_view_index.copy()
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(self.colmap_cameras[view_index], train=True)["comp_rgb"]
            # if (step+1) % 250 == 0:
            #     image1 = self.render(self.colmap_cameras[37])["comp_rgb"]
            #     image2 = self.render(self.colmap_cameras[29])["comp_rgb"]
            #     Renderings1.append(image1.permute(0,3,1,2).cpu())
            #     Renderings2.append(image2.permute(0,3,1,2).cpu())
            
            loss = self.guidance(rendering, view_index, step)

            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return
            if ema_loss_for_log == 0:
                ema_loss_for_log = loss.item()
            else:
                ema_loss_for_log = self.alpha * ema_loss_for_log + (1-self.alpha) * loss.item()
        
        # Renderings1 = torch.cat(Renderings1, dim=0)
        # Renderings2 = torch.cat(Renderings2, dim=0)
        # save_image(Renderings1, f"batch_image1_{self.edit_train_steps}.png", nrow=Renderings1.shape[0])
        # save_image(Renderings2, f"batch_image2_{self.edit_train_steps}.png", nrow=Renderings2.shape[0])
        os.makedirs("save", exist_ok=True)
        self.gaussian.save_ply("save/result1.ply")
    
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

        self.guidance.guidance.max_step = self.t_max_step[min(len(self.t_max_step)-1, global_step// self.cameara_update_step)]
        with torch.no_grad():
            for id in self.view_list:
                cameras.append(self.colmap_cameras[id])
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            # sorted_cam_idx = range(len(cameras))
            view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]  
                
            for id in view_sorted:
                cur_cam = self.colmap_cameras[id]
                out_pkg = self.render(cur_cam)
                out = out_pkg["comp_rgb"]
                if self.use_masked_image:
                    out = out * out_pkg["masks"].unsqueeze(-1)
                images.append(out)
                if sam_option != 0:
                    if isinstance(self.masks[id], np.ndarray):
                        mask = torch.from_numpy(self.masks[id]/255).unsqueeze(0)
                        mask = mask.to(torch.float32).to(get_device())
                    else:
                        mask = self.masks[id].unsqueeze(0)
                    mask = self.gaussian_blur(mask)
                    masks.append(mask)
                origin_frames.append(self.origin_frames[id])
                depths.append(self.depths[id])

            images = torch.cat(images, dim=0)
            if sam_option != 0:
                masks = torch.cat(masks, dim=0)
                masks = masks.permute(0, 2, 3, 1) # B H W C
            else:
                masks = torch.ones_like(images)
            origin_frames = torch.cat(origin_frames, dim=0)
            depths = torch.cat(depths, dim=0)

            if self.guidance_type == "InstructPix2Pix":
                edited_images = self.guidance.edit_all(images, origin_frames)
            elif self.guidance_type == "ControlNet-Depth":
                edited_images = self.guidance.edit_all(images, depths)

            edited_images = edited_images * masks + (1-masks) * origin_frames

            # save_image(images.permute(0,3,1,2), f'batch_image_{global_step}.png', nrow=4)
            save_image(edited_images.permute(0, 3, 1, 2), f'batch_image_{global_step}.png', nrow=4)
            for view_index_tmp in range(len(self.view_list)):
                self.guidance.edit_frames[view_sorted[view_index_tmp]] = edited_images[view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--edit_cam_num", type=int, default=0, help="Camera number.")
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
    parser.add_argument("--positive_sam_points", type=str, default="/", help="the path of the positive sam points.")
    parser.add_argument("--negative_sam_points", type=str, default="/", help="the path of the negative sam points.")
    parser.add_argument("--camera", type=str, default="tmd_delete/camera.pkl", help="camera.")

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