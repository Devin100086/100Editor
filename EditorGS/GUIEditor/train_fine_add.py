from argparse import ArgumentParser
from pathlib import Path
import pickle
import subprocess
from omegaconf import OmegaConf
import rembg
from tqdm import tqdm
from PIL import Image
from lang_sam import LangSAM
from transformers import pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.gaussiansplatting.utils.graphics_utils import fov2focal
from EditorGS.GUIEditor.utils import *
from EditorGS.GUIEditor.Network import EditorNetwork
from EditorGS.GUIEditor.Guidance.EditFineGuidance import BrushNetGuidance, EditFineGuidance
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.ops import masks_to_boxes
from utils import *
import torch.nn.functional as F
import numpy as np
import torch

from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler

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
        self.lang_sam = LangSAMTextSegmentor().to(get_device())



        self.use_masked_image = False

    def edit(self, cam):
        BrushEdit_path = ".cache/models"
        base_model_path = os.path.join(BrushEdit_path, "base_model/realisticVisionV60B1_v51VAE")
        torch_dtype = torch.float16
        brushnet_path = os.path.join(BrushEdit_path, "brushnetX")
        brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype)
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path, brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
    )
        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # remove following line if xformers is not installed or when using Torch 2.0.
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        self.ctn_brushnetx = pipe
        self.ctn_brushnetx.set_progress_bar_config(disable=True)
        self.ctn_brushnetx.safety_checker = None
        
        edit_cameras = sample_train_camera(self.colmap_cameras,
                                           self.edit_cam_num,
                                          )
        self.origin_frames = self.render_cameras_list(edit_cameras)
        masks = self.get_mask(edit_cameras)
        self.update_mask(self.colmap_cameras)
        self.guidance = EditFineGuidance(
            guidance=self.ctn_brushnetx,
            gaussian=self.gaussian,
            masks=masks,
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
            origin_camera = self.colmap_cameras
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

            loss = self.guidance(rendering, view_index, step, self.edit_text, self.negative_prompt)
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

    def get_mask(self, edit_cameras):
        masks = []
        depths = []
        kernel =  np.ones((5,5),np.uint8)
        for cam in edit_cameras:
            out = self.render(cam,mask=True)["comp_rgb"]
            text_prompt = "hat"
            sam_results = self.lang_sam(out, text_prompt)[
                    0
                ]
            # mask_np = sam_results.numpy().astype(np.uint8) * 255
            mask_np = cv2.dilate(sam_results.numpy().astype(np.uint8), kernel, iterations=5) * 255
            mask_np = mask_np.squeeze(0)
            masks.append(mask_np)
        
        return masks
    
    def update_mask(self,edit_cameras) -> None:

        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
        kernel =  np.ones((5,5),np.uint8)

        for i,cam in enumerate(edit_cameras):
            cur_cam = cam
            this_frame = render(
                cur_cam, self.gaussian2, self.pipe, self.background_tensor
            )["render"]
            text_prompt = "hat"

            mask = self.lang_sam(this_frame.unsqueeze(0).permute(0,2,3,1), text_prompt)[
                    0
                ].to(get_device())
        
            masks.append(mask)
            self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

        weights /= weights_cnt + 1e-7
        selected_mask = weights > 0.3
        selected_mask = selected_mask[:, 0]
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)

        return masks, selected_mask


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--cam_dir", type=str,required=True)
    parser.add_argument("--mask_dir", type=str,required=True)
    parser.add_argument("--negative_prompt", type=str ,default="ugly, low quality")
    parser.add_argument("--text_prompt", type=str ,default="turn him a clown", help="Text prompt.")
    parser.add_argument("--edit_train_steps", type=int, default=1500, help="Edit train steps.")

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


    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = TrainFineeAdd(args)
        trainer.gaussian.update_anchor_term(
            anchor_weight_init_g0=0.05,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=1.3,
        )
        trainer.configure_optimizers()
        with open(args.cam_dir, 'rb') as f:
            cam = pickle.load(f)
        trainer.edit(cam)
    
