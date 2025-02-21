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