from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.GUIEditor.utils import *
from EditorGS.GUIEditor.Network import EditorNetwork
from EditorGS.GUIEditor.Guidance.EditFineGuidance import EditFineGuidance
from utils import *
import torch.nn.functional as F
import numpy as np
import torch

from torchvision.utils import save_image

class TrainFineeAdd(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mask_dir = cfg.mask_dir
        self.inpaint_seed = 1
        self.depth_scaler = 1
        self.refine_text = ""
        self.negative_prompt = cfg.negative_prompt

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
        self.cameara_update_step = cfg.cameara_update_step
        self.seg_prompt = cfg.seg_prompt
        self.lang_sam = LangSAMTextSegmentor().to(get_device())

        self.mask_frames = {}

        self.use_masked_image = False

    def edit(self, one_time = True):
        from threestudio.models.guidance.brushnet_guidance import (
                    BrushNetGuidance,
                )
        self.brushnet = BrushNetGuidance(
                    OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98, "video": one_time})
                )
        cur_2D_guidance = self.brushnet
        print("using BrushNet!")

        # self.edit_cameras = sample_train_camera(self.colmap_cameras,
        #                                    self.edit_cam_num,
        #                                   )
        self.origin_frames = self.render_cameras_list(self.colmap_cameras)

        random.seed(0)  # make sure same views
        self.n2n_view_index = random.sample(
            range(0, len(self.colmap_cameras)),
            min(len(self.colmap_cameras), self.edit_cam_num),
        )
        self.view_list = self.n2n_view_index

        self.masks = self.get_mask(self.colmap_cameras, text_prompt=self.seg_prompt)
        self.update_mask(self.colmap_cameras, text_prompt=self.seg_prompt)
        self.guidance = EditFineGuidance(
            guidance=cur_2D_guidance,
            gaussian=self.gaussian,
            masks=self.masks,
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
        )
        view_index_stack = self.n2n_view_index.copy()
        ema_loss_for_log = 0.0
        network = EditorNetwork(host="127.0.0.1",port=8084)
        
        for step in tqdm(range(self.edit_train_steps)):
            if step % self.cameara_update_step == 0 and one_time:
                print("start editing")
                self.edit_all_view(update_camera= step >= self.cameara_update_step, global_step=step)
                print("end editing")
            network.render(self.pipe,self.gaussian,ema_loss_for_log,render,self.background_tensor,step,self.opt)
            
            if not view_index_stack:
                view_index_stack = self.n2n_view_index.copy()
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(self.colmap_cameras[view_index], train=True)["comp_rgb"]

            if not one_time:
                loss = self.guidance(rendering, view_index, step)
            else:
                loss = self.guidance.get_loss(rendering, view_index, step)

            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return
            
            ema_loss_for_log = self.alpha * ema_loss_for_log + (1-self.alpha) * loss.item()

        os.makedirs("save", exist_ok=True)
        self.gaussian.save_ply("save/result1.ply")

    def sort_the_cameras_idx(self, cams):
        foward_vectos = [cam.R[:, 2] for cam in cams]
        foward_vectos = np.array(foward_vectos)
        cams_center_x = np.array([cam.camera_center[0].item() for cam in cams])
        most_left_vecotr = foward_vectos[np.argmin(cams_center_x)]
        distances = [np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), 0, 1)) for cam in cams]
        sorted_cams = [cam for _, cam in sorted(zip(distances, cams), key=lambda pair: pair[0])]
        reference_axis = np.cross(most_left_vecotr, sorted_cams[1].R[:, 2])
        distances_with_sign = [np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) if np.dot(reference_axis,  np.cross(most_left_vecotr, cam.R[:, 2])) >= 0 else 2 * np.pi - np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) for cam in cams]
        
        sorted_cam_idx = [idx for _, idx in sorted(zip(distances_with_sign, range(len(cams))), key=lambda pair: pair[0])]

        return sorted_cam_idx

    def update_cameras(self, random_seed=0):
        random.seed(random_seed)
        self.n2n_view_index = random.sample(
            range(0, len(self.colmap_cameras)),
            min(len(self.colmap_cameras), 16),
        )

    def edit_all_view(self, update_camera=False, global_step=0):
        
        self.edited_cams = []
        if update_camera:
            self.update_cameras(random_seed = global_step + 1)
            self.view_list = self.n2n_view_index

        cameras = []
        images = []
        masked_frames = []
        t_max_step = [999, 300, 300, 21]
        self.guidance.guidance.max_step = t_max_step[min(len(t_max_step)-1, self.edit_train_steps// self.cameara_update_step)]
        with torch.no_grad():
            for id in self.view_list:
                cameras.append(self.colmap_cameras[id])
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]  
                   
            for id in view_sorted:
                cur_cam = self.colmap_cameras[id]

                # out_pkg = self(cur_batch)
                out_pkg = self.render(cur_cam)
                out = out_pkg["comp_rgb"]
                # if self.cfg.use_masked_image:
                #     out = out * out_pkg["masks"].unsqueeze(-1)
                images.append(out)
                cached_image = self.masks[id]
                self.mask_frames[id] = torch.tensor(
                    cached_image , device="cuda", dtype=torch.float32
                )
                masked_frames.append(self.mask_frames[id])
            images = torch.cat(images, dim=0)
            masked_frames = torch.cat(masked_frames, dim=0).unsqueeze(1).cpu().numpy()

            edited_images = self.guidance.edit_all(
                images,
                masked_frames,
                global_step,
            )

            # save_image(images.permute(0,3,1,2), f'batch_image_{global_step}.png', nrow=4)
            save_image(edited_images.permute(0, 3, 1, 2), f'batch_image_{global_step}.png', nrow=4)
            for view_index_tmp in range(len(self.view_list)):
                self.guidance.edit_frames[view_sorted[view_index_tmp]] = edited_images[view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C
    
    def get_mask(self, edit_cameras, text_prompt="hat"):
        masks = []
        depths = []
        kernel =  np.ones((5,5),np.uint8)
        for cam in edit_cameras:
            out = self.render(cam,mask=True)["comp_rgb"]
            sam_results = self.lang_sam(out, text_prompt)[
                    0
                ]
            mask_np = sam_results.numpy().astype(np.uint8) * 255
            # mask_np = cv2.dilate(sam_results.numpy().astype(np.uint8), kernel, iterations=5) * 255

            masks.append(mask_np)
        
        return masks


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str,required=True)
    parser.add_argument("--negative_prompt", type=str ,default="ugly, low quality")
    parser.add_argument("--seg_prompt", type=str ,default="hat", help="Seg Prompt.")
    parser.add_argument("--text_prompt", type=str ,default="turn him a clown", help="Text prompt.")
    parser.add_argument("--edit_train_steps", type=int, default=1500, help="Edit train steps.")
    parser.add_argument("--cameara_update_step", type=int, default=500, help="Cameara Update Step.")

    parser.add_argument("--guidance_type", type=str, default="InstructPix2Pix")
    parser.add_argument("--edit_cam_num", type=int, default=0, help="Camera number.")
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
    parser.add_argument("--video", type=str, default=True, help="video editing pattern.")


    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = TrainFineeAdd(args)
        trainer.gaussian.update_anchor_term(
            anchor_weight_init_g0=0.05,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=1.3,
        )
        trainer.configure_optimizers()
        
        trainer.edit(one_time=eval(args.video))
    
    
    
