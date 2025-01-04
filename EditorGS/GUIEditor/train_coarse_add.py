from argparse import ArgumentParser
from pathlib import Path
import pickle
import subprocess
from omegaconf import OmegaConf
import rembg
from tqdm import tqdm
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.gaussiansplatting.scene.vanilla_gaussian_model import (
    GaussianModel as VanillaGaussianModel,
)
from EditorGS.gaussiansplatting.utils.graphics_utils import fov2focal
from EditorGS.GUIEditor.utils import *
from EditorGS.GUIEditor.Network import EditorNetwork
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.ops import masks_to_boxes
from utils import *
import numpy as np
import torch
from lang_sam import LangSAM

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

class TrainCoarseAdd(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.mask_dir = cfg.mask_dir
        self.inpaint_seed = 1
        self.refine_text = ""
        self.zoom = None if cfg.zoom == -1 else cfg.zoom
        self.left_up = None if cfg.left_up[0] == -1 else [int(x/self.zoom) for x in cfg.left_up]
        self.right_down = None if cfg.left_up[0] == -1 else [int(x/self.zoom) for x in cfg.right_down]
        self.langsam = LangSAM()

    def add(self, cam):
        self.cam = cam
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        pipe.enable_model_cpu_offload()

        self.ctn_inpaint = pipe
        self.ctn_inpaint.set_progress_bar_config(disable=True)
        self.ctn_inpaint.safety_checker = None

        with torch.no_grad():
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        
        image_in = to_pil_image(torch.clip(render_pkg["render"], 0.0, 1.0))
        origin_size = image_in.size

        if self.inpaint_again:
            mask_in = torch.zeros(
                (origin_size[1], origin_size[0]),
                dtype=torch.float32,
                device=get_device(),
            )  # H, W
            mask_in[
                self.left_up[1] : self.right_down[1],
                self.left_up[0] : self.right_down[0],
            ] = 1.0

            # mask_in = Image.open(self.mask_dir).convert("L")

            mask_in = np.array(mask_in.cpu())
            mask_in = torch.from_numpy(mask_in).to(dtype=torch.float32).to(get_device())

            image_in_pil = to_pil_image(
                resize_image_ctn(np.asarray(image_in), 512)
            )

            mask_in = to_pil_image(mask_in)  # .resize((1024, 1024))

            mask_in_pil = to_pil_image(
                resize_image_ctn(np.asarray(mask_in)[..., None], 512)
            )

            image = np.array(image_in_pil.convert("RGB")).astype(np.float32) / 255.0
            image_mask = (
                np.array(mask_in_pil.convert("L")).astype(np.float32) / 255.0
            )

            image[image_mask > 0.5] = -1.0  # set as masked pixel
            image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            control_image = torch.from_numpy(image).to("cuda")
            generator = torch.Generator(device="cuda").manual_seed(
                self.inpaint_seed
            )
            out = self.ctn_inpaint(
                self.edit_text+", high quality, extremely detailed",
                num_inference_steps=25,
                generator=generator,
                eta=1.0,
                image=image_in_pil,
                mask_image=mask_in_pil,
                control_image=control_image,
            ).images[0]
            out = cv2.resize(
                np.asarray(out),
                origin_size,
                interpolation=cv2.INTER_LANCZOS4
                if out.width / origin_size[0] > 1
                else cv2.INTER_AREA,
            )
            out = to_pil_image(out)
            self.inpaint_again = False
        
        removed_bg = rembg.remove(out)
        inpainted_image = to_tensor(out)[:3,...].to("cuda")

        cache_dir = Path("tmp_add").absolute().as_posix()
        os.makedirs(cache_dir, exist_ok=True)
        mv_image_dir = os.path.join(cache_dir, "multiview_pred_images")
        os.makedirs(mv_image_dir, exist_ok=True)
        inpaint_path = os.path.join(cache_dir, "inpainted.png")
        removed_bg_path = os.path.join(cache_dir, "removed_bg.png")
        mesh_path = os.path.join(cache_dir, "inpaint_mesh.obj")
        gs_path = os.path.join(cache_dir, "inpaint_gs.obj")
        out.save(inpaint_path)
        removed_bg.save(removed_bg_path)

        p1 = subprocess.Popen(
            f"{sys.prefix}/bin/accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py "
            f"--save_dir {mv_image_dir} --config configs/mvdiffusion-joint-ortho-6views.yaml"
            f" validation_dataset.root_dir={cache_dir} validation_dataset.filepaths=[removed_bg.png]".split(
                " "
            ),
            cwd="threestudio/utils/wonder3D",
        )
        p1.wait()

        print(
            f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml --save_dir {cache_dir} --gpu 0 --train dataset.root_dir={os.path.dirname(mv_image_dir)} dataset.scene={os.path.basename(mv_image_dir)}"
        )
        cmd = f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml --save_dir {cache_dir} --gpu 0 --train dataset.root_dir={os.path.dirname(mv_image_dir)} dataset.scene={os.path.basename(mv_image_dir)}".split(
            " "
        )
        p2 = subprocess.Popen(
            cmd,
            cwd="threestudio/utils/wonder3D/instant-nsr-pl",
        )
        p2.wait()

        p3 = subprocess.Popen(
            [
                f"{sys.prefix}/bin/python",
                "train_from_mesh.py",
                "--mesh",
                mesh_path,
                "--save_path",
                gs_path,
                "--prompt",
                self.refine_text,
            ]
        )
        p3.wait()
    
    def add_sketch(self, image_pil, text_prompt):
        results = self.langsam.predict([image_pil], [text_prompt])
        mask = results[0]['masks'].astype(np.uint8) * 255
        mask = mask.squeeze()
        original_image = np.array(image_pil)
        bgra_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
        bgra_image[:, :, 3] = mask
        out = Image.fromarray(bgra_image)

        removed_bg = rembg.remove(out)

        cache_dir = Path("tmp_add").absolute().as_posix()
        os.makedirs(cache_dir, exist_ok=True)
        mv_image_dir = os.path.join(cache_dir, "multiview_pred_images")
        os.makedirs(mv_image_dir, exist_ok=True)
        inpaint_path = os.path.join(cache_dir, "inpainted.png")
        removed_bg_path = os.path.join(cache_dir, "removed_bg.png")
        mesh_path = os.path.join(cache_dir, "inpaint_mesh.obj")
        gs_path = os.path.join(cache_dir, "inpaint_gs.obj")
        out.save(inpaint_path)
        removed_bg.save(removed_bg_path)

        p1 = subprocess.Popen(
            f"{sys.prefix}/bin/accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py "
            f"--save_dir {mv_image_dir} --config configs/mvdiffusion-joint-ortho-6views.yaml"
            f" validation_dataset.root_dir={cache_dir} validation_dataset.filepaths=[removed_bg.png]".split(
                " "
            ),
            cwd="threestudio/utils/wonder3D",
        )
        p1.wait()

        print(
            f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml --save_dir {cache_dir} --gpu 0 --train dataset.root_dir={os.path.dirname(mv_image_dir)} dataset.scene={os.path.basename(mv_image_dir)}"
        )
        cmd = f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml --save_dir {cache_dir} --gpu 0 --train dataset.root_dir={os.path.dirname(mv_image_dir)} dataset.scene={os.path.basename(mv_image_dir)}".split(
            " "
        )
        p2 = subprocess.Popen(
            cmd,
            cwd="threestudio/utils/wonder3D/instant-nsr-pl",
        )
        p2.wait()
        p3 = subprocess.Popen(
            [
                f"{sys.prefix}/bin/python",
                "train_from_mesh.py",
                "--mesh",
                mesh_path,
                "--save_path",
                gs_path,
                "--prompt",
                "",
            ]
        )
        p3.wait()

    def show(self,depth,cam):
        
        network = EditorNetwork(host="127.0.0.1",port=8084)
        cache_dir = Path("tmp_add").absolute().as_posix()
        os.makedirs(cache_dir, exist_ok=True)
        mv_image_dir = os.path.join(cache_dir, "multiview_pred_images")
        os.makedirs(mv_image_dir, exist_ok=True)
        inpaint_path = os.path.join(cache_dir, "inpainted.png")
        removed_bg_path = os.path.join(cache_dir, "removed_bg.png")
        gs_path = os.path.join(cache_dir, "inpaint_gs.obj")

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
        # ui_utils.vis_depth(estimated_depth.cpu())
        object_center = (bbox[:2] + bbox[2:]) / 2

        fx = fov2focal(cam.FoVx, cam.image_width)
        fy = fov2focal(cam.FoVy, cam.image_height)

        object_center = (
            object_center
            - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
        ) / torch.tensor([fx, fy]).to("cuda")

        with torch.no_grad():
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        rendered_depth = render_pkg["depth_3dgs"][..., ~object_mask]

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

        if self.scale_depth:
            if new_object_gaussian is not None:
                self.gaussian.prune_with_mask()
            scaled_z_in_cam = z_in_cam * depth
            x_in_cam, y_in_cam = (object_center.cuda()) * scaled_z_in_cam
            T_in_cam = torch.stack([x_in_cam, y_in_cam, scaled_z_in_cam], dim=-1)

            bbox = bbox.cuda()
            real_scale = (
                (bbox[2:] - bbox[:2])
                / torch.tensor([fx, fy], device="cuda")
                * scaled_z_in_cam
            )

            new_object_gaussian = VanillaGaussianModel(self.gaussian.max_sh_degree)
            new_object_gaussian.load_ply(gs_path)
            new_object_gaussian._opacity.data = (
                torch.ones_like(new_object_gaussian._opacity.data) * 99.99
            )

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

            self.gaussian.concat_gaussians(new_object_gaussian)
            self.scale_depth = False
            self.gaussian.save_ply("tmp_add/merge.ply")
            network.render(self.pipe,self.gaussian,0, render,self.background_tensor,0,self.opt,show=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--cam_dir", type=str,required=True)
    parser.add_argument("--depth", type=float,required=False)
    # parser.add_argument("--mask_dir", type=str,required=True)
    parser.add_argument("--text_prompt", type=str ,default="turn him a clown", help="Text prompt.")
    parser.add_argument("--edit_train_steps", type=int, default=1500, help="Edit train steps.")
    parser.add_argument("--left_up", type=int, nargs=2)
    parser.add_argument("--right_down", type=int, nargs=2)
    parser.add_argument("--zoom", type=float)

    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = TrainCoarseAdd(args)
        # trainer.gaussian.update_anchor_term(
        #     anchor_weight_init_g0=0.05,
        #     anchor_weight_init=0.1,
        #     anchor_weight_multiplier=1.3,
        # )
        trainer.configure_optimizers()
        with open(args.cam_dir, 'rb') as f:
            cam = pickle.load(f)
        
        trainer.show(args.depth, cam)