from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import BaseTrainer
from EditorGS.GUIEditor.Guidance.EditGuidance import EditGuidance
from torchvision.transforms.functional import to_pil_image, to_tensor
from EditorGS.gaussiansplatting.gaussian_renderer import render
from lang_sam import LangSAM
from PIL import Image
from EditorGS.GUIEditor.utils import *
from EditorGS.GUIEditor.Network import EditorNetwork
class EditTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.edit_cam_num = cfg.edit_cam_num
        self.guidance_type = cfg.guidance_type
        self.lambda_l1 = cfg.lambda_l1
        self.lambda_p = cfg.lambda_p
        self.lambda_anchor_color = cfg.lambda_anchor_color
        self.lambda_anchor_geo = cfg.lambda_anchor_geo
        self.lambda_anchor_scale = cfg.lambda_anchor_scale
        self.lambda_anchor_opacity = cfg.lambda_anchor_opacity
        self.per_editing_step = cfg.per_editing_step
        self.lang_sam = LangSAM()
        self.edit_begin_step = cfg.edit_begin_step
        self.edit_until_step = cfg.edit_until_step

    def edit(self):
        edit_cameras = sample_train_camera(self.colmap_cameras,
                                           self.edit_cam_num,
                                          )
        if self.guidance_type == "InstructPix2Pix":
            if not self.ip2p:
                from threestudio.models.guidance.instructpix2pix_guidance import (
                    InstructPix2PixGuidance,
                )

                self.ip2p = InstructPix2PixGuidance(
                    OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98})
                )
            cur_2D_guidance = self.ip2p
            print("using InstructPix2Pix!")
        elif self.guidance_type == "ControlNet-Pix2Pix":
            if not self.ctn_ip2p:
                from threestudio.models.guidance.controlnet_guidance import (
                    ControlNetGuidance,
                )

                self.ctn_ip2p = ControlNetGuidance(
                    OmegaConf.create({"min_step_percent": 0.05,
                                      "max_step_percent": 0.8,
                                        "control_type": "p2p"})
                )
            cur_2D_guidance = self.ctn_ip2p
            print("using ControlNet-InstructPix2Pix!")
        
        origin_frames = self.render_cameras_list(edit_cameras)

        self.guidance = EditGuidance(
            guidance=cur_2D_guidance,
            gaussian=self.gaussian,
            origin_frames=origin_frames,
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
            cams=edit_cameras,
        )
        view_index_stack = list(range(len(edit_cameras)))
        ema_loss_for_log = 0.0
        network = EditorNetwork(host="127.0.0.1",port=8084)
        for step in tqdm(range(self.edit_train_steps)):
            network.render(self.pipe,self.gaussian,ema_loss_for_log,render,self.background_tensor,step,self.opt)
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

            loss = self.guidance(rendering, view_index, step)
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--edit_cam_num", type=int, default=0, help="Camera number.")
    parser.add_argument("--guidance_type", type=str, default="InstructPix2Pix")
    parser.add_argument("--text_prompt", default="default_text", help="Text prompt.")
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

    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = EditTrainer(args)
        trainer.gaussian.update_anchor_term(
            anchor_weight_init_g0=0.05,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=1.3,
        )
        trainer.configure_optimizers()
        trainer.edit()