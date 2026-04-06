from argparse import ArgumentParser
import pickle
from omegaconf import OmegaConf
import subprocess
from tqdm import tqdm
import sys
import os
import time
from editor.hundrededitor_gui.base import *
from third_party.gaussiansplatting.gaussian_renderer import render
from editor.hundrededitor_gui.utils import *
from editor.hundrededitor_gui.Network import EditorNetwork
from editor.hundrededitor_gui.guidance.add_guidance import AddGuidance
from utils import *
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
import rembg
from pathlib import Path
import datetime
from torchvision.utils import save_image
from threestudio.models.guidance.brushnet_guidance import (
            BrushNetGuidance,
        )
import cv2
from src.utils.path_utils import find_repo_root, resolve_runtime_subdir


def _supports_ansi() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _ansi(text: str, code: str) -> str:
    if not _supports_ansi():
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_kv(label: str, value: str, color: str = "36"):
    print(_ansi(f"  {label:<12}:", color), value)


def _run_external_step(
    step_name: str,
    cmd: list[str],
    *,
    cwd: str | None = None,
    log_path: Path | None = None,
    live_output: bool = False,
) -> None:
    print(_ansi(f"[Coarse Adding] {step_name} ...", "1;34"))
    step_start = time.perf_counter()
    tail_lines = []

    def _write_header(fp):
        fp.write(f"\n===== {step_name} =====\n")
        fp.write("Command:\n")
        fp.write(" ".join(cmd) + "\n")
        if cwd is not None:
            fp.write(f"CWD: {cwd}\n")

    if live_output:
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                _write_header(f)
                f.write("\n[live output]\n")
                f.write(
                    "Output is streamed directly to terminal to preserve tqdm progress bar.\n"
                )
        proc = subprocess.run(
            cmd,
            cwd=cwd,
        )
        return_code = proc.returncode
        elapsed = time.perf_counter() - step_start
        tail_lines = []
    else:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - step_start
        return_code = proc.returncode
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                _write_header(f)
                if proc.stdout:
                    f.write("\n[stdout]\n")
                    f.write(proc.stdout)
                if proc.stderr:
                    f.write("\n[stderr]\n")
                    f.write(proc.stderr)
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip().splitlines()
        tail_lines = combined[-8:] if combined else []

    if return_code != 0:
        print(
            _ansi(
                f"[Coarse Adding] {step_name} failed (exit={return_code}, {elapsed:.2f}s)",
                "1;31",
            )
        )
        if tail_lines:
            print(_ansi("  Last logs:", "31"))
            for line in tail_lines:
                print(f"    {line}")
        raise RuntimeError(f"{step_name} failed with exit code {return_code}")

    print(_ansi(f"[Coarse Adding] {step_name} done ({elapsed:.2f}s)", "1;32"))


class TrainFineeAdd(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cache_dir = Path(cfg.camera).resolve().parent
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.debug_render_dir = resolve_runtime_subdir(
            __file__, "renders", "add", create=True
        )
        self.mask_dir = cfg.mask_dir

        self.edit_cam_num = cfg.edit_cam_num
        self.guidance_type = cfg.guidance_type
        self.lambda_l1 = cfg.lambda_l1
        self.lambda_p = cfg.lambda_p
        self.lambda_anchor_color = cfg.lambda_anchor_color
        self.lambda_anchor_geo = cfg.lambda_anchor_geo
        self.lambda_anchor_scale = cfg.lambda_anchor_scale
        self.lambda_anchor_opacity = cfg.lambda_anchor_opacity
        self.per_editing_step = cfg.per_editing_step
        self.densify_until_step = cfg.densify_until_step
        self.edit_begin_step = cfg.edit_begin_step
        self.edit_until_step = cfg.edit_until_step
        self.cameara_update_step = cfg.cameara_update_step
        self.densification_interval = cfg.densification_interval
        self.seg_prompt = cfg.seg_prompt
        self.lang_sam = LangSAMTextSegmentor().to(get_device())
        self.output_dir = cfg.output_dir
        
        self.gaussian2 = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        self.gaussian2.load_ply((self.cache_dir / "merge.ply").as_posix())
        self.save_mask_tmp = (self.cache_dir / "mask").as_posix()

        self.mask_frames = {}

        self.use_masked_image = False
        self.t_max_step = [999, 300, 300, 21]

        with open(cfg.camera, 'rb') as f:
            self.cam  = pickle.load(f)

    def edit(self, video = True):
        now = datetime.datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.output_dir = os.path.join(self.output_dir, now)
        os.makedirs(self.output_dir, exist_ok=True)

        self.brushnet = BrushNetGuidance(
                    OmegaConf.create({"min_step_percent": 0.02,
                                      "max_step_percent": 0.98,
                                      "video": video})
                )
        cur_2D_guidance = self.brushnet
        print("using BrushNet!")

        # self.edit_cameras = sample_train_camera(self.colmap_cameras,
        #                                    self.edit_cam_num,
        #                                   )
        self.origin_frames = self.render_cameras_list(self.colmap_cameras, separate_sh=self.use_sparse_adam)

        random.seed(0)  # make sure same views
        self.n2n_view_index = random.sample(
            range(0, len(self.colmap_cameras)),
            min(len(self.colmap_cameras), self.edit_cam_num),
        )
        self.view_list = self.n2n_view_index

        # self.masks, _ = self.update_mask(self.colmap_cameras, text_prompt=self.seg_prompt)

        points3d = np.load((self.cache_dir / "center_3D.npy").as_posix())
        points2d = project_3d_to_2d(points3d, self.cam) if len(points3d) > 0 else np.empty((0,2))
        render_folder = os.path.join(os.path.dirname(self.save_mask_tmp), "render")
        init_render = render(self.cam, self.gaussian2, self.pipe ,self.background_tensor, separate_sh=self.use_sparse_adam)["render"]
        save_image(init_render[None], f"{render_folder}/{0:05d}" + ".jpg")
        print(_ansi("[Fine Add] SAM2(video) mask preparation ...", "1;34"))
        self.masks, _ = self.update_sam2_mask_with_point_prompt(self.colmap_cameras, points2d, np.empty((0,2)), type = "add")
        print(_ansi("[Fine Add] SAM2(video) mask preparation done", "1;32"))

        self.guidance = AddGuidance(
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
        
        for step in tqdm(
            range(self.edit_train_steps),
            desc="Fine Add Train",
            dynamic_ncols=True,
        ):
            network.render(self.pipe,self.gaussian,ema_loss_for_log,render,self.background_tensor,step,self.opt, self.use_sparse_adam)
            if step % self.cameara_update_step == 0 and video:
                self.edit_all_view(update_camera= step >= self.cameara_update_step, global_step=step)

            if not view_index_stack:
                view_index_stack = self.n2n_view_index.copy()
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(self.colmap_cameras[view_index], train=True, separate_sh=self.use_sparse_adam)["comp_rgb"]

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
            
            ema_loss_for_log = self.alpha * ema_loss_for_log + (1-self.alpha) * loss.item()

        self.gaussian.save_ply(f"{self.output_dir}/result.ply")

    def edit_all_view(self, update_camera=False, global_step=0):
        
        self.edited_cams = []
        if update_camera:
            self.update_cameras(random_seed = global_step + 1)
            self.view_list = self.n2n_view_index

        cameras = []
        images = []
        masked_frames = []
        self.guidance.guidance.max_step = self.t_max_step[min(len(self.t_max_step)-1, global_step// self.cameara_update_step)]
        with torch.no_grad():
            for id in self.view_list:
                cameras.append(self.colmap_cameras[id])
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]  
                   
            for id in view_sorted:
                cur_cam = self.colmap_cameras[id]
                out_pkg = self.render(cur_cam, separate_sh=self.use_sparse_adam)
                out = out_pkg["comp_rgb"]
                if self.use_masked_image:
                    out = out * out_pkg["masks"].unsqueeze(-1)
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
            )

            save_image(
                edited_images.permute(0, 3, 1, 2),
                (self.debug_render_dir / f"batch_image_{global_step}.png").as_posix(),
                nrow=4,
            )
            for view_index_tmp in range(len(self.view_list)):
                self.guidance.edit_frames[view_sorted[view_index_tmp]] = edited_images[view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C

def add_sketch(image_pil, text_prompt, cache_dir=None):
    total_start = time.perf_counter()
    from lang_sam import LangSAM
    print(_ansi("\n[Coarse Adding] Start", "1;36"))
    print(_ansi("[Coarse Adding] Step 1/4: Segment object with LangSAM", "1;34"))
    langsam = LangSAM()
    results = langsam.predict([image_pil], [text_prompt])
    mask = results[0]['masks'].astype(np.uint8) * 255
    mask = mask.squeeze()
    original_image = np.array(image_pil)
    bgra_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    bgra_image[:, :, 3] = mask
    out = Image.fromarray(bgra_image)

    removed_bg = rembg.remove(out)

    if cache_dir is None:
        cache_dir = resolve_runtime_subdir(__file__, "cache", "add", create=True).as_posix()
    cache_dir = Path(cache_dir).resolve()
    os.makedirs(cache_dir, exist_ok=True)
    log_name = f"coarse_add_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    log_path = cache_dir / log_name
    # mv_image_dir = os.path.join(cache_dir, "multiview_pred_images")
    # os.makedirs(mv_image_dir, exist_ok=True)
    inpaint_path = (cache_dir / "inpainted.png").as_posix()
    removed_bg_path = (cache_dir / "removed_bg.png").as_posix()
    mesh_path = (cache_dir / "inpaint_mesh.obj").as_posix()
    gs_path = (cache_dir / "inpaint_gs.ply").as_posix()
    _print_kv("Prompt", text_prompt)
    _print_kv("Cache dir", cache_dir.as_posix())
    _print_kv("Log file", log_path.as_posix())
    _print_kv("Output mesh", mesh_path)
    _print_kv("Output ply", gs_path)

    print(_ansi("[Coarse Adding] Step 2/4: Prepare RGBA inputs", "1;34"))
    image_pil.save(inpaint_path)
    removed_bg.save(removed_bg_path)

    repo_root = find_repo_root(__file__)
    dreamgaussian_dir = repo_root / "third_party" / "dreamgaussian"
    process_script = dreamgaussian_dir / "process.py"
    main_script = dreamgaussian_dir / "main.py"
    config_file = dreamgaussian_dir / "configs" / "image_sai.yaml"
    if not process_script.is_file() or not main_script.is_file() or not config_file.is_file():
        raise FileNotFoundError(
            "DreamGaussian scripts not found under third_party/dreamgaussian. "
            f"Checked: {process_script}, {main_script}, {config_file}"
        )

    _run_external_step(
        "Step 3/4: DreamGaussian preprocess",
        [
            f"{sys.prefix}/bin/python",
            process_script.as_posix(),
            f"{removed_bg_path}",
        ],
        cwd=dreamgaussian_dir.as_posix(),
        log_path=log_path,
    )

    _run_external_step(
        "Step 4/4: DreamGaussian reconstruction",
        [
            f"{sys.prefix}/bin/python",
            main_script.as_posix(),
            "--config",
            config_file.as_posix(),
            f"input={removed_bg_path.replace('.png', '_rgba.png')}",
            f"outdir={cache_dir.as_posix()}",
        ],
        cwd=dreamgaussian_dir.as_posix(),
        log_path=log_path,
        live_output=True,
    )

    total_elapsed = time.perf_counter() - total_start
    print(_ansi(f"[Coarse Adding] Done ({total_elapsed:.2f}s)\n", "1;32"))

if __name__ == "__main__":
    import time 
    start_time = time.time()
    default_add_cache_dir = resolve_runtime_subdir(__file__, "cache", "add", create=True)
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str,required=True)
    parser.add_argument("--seg_prompt", type=str ,default="hat", help="Seg Prompt.")
    parser.add_argument("--text_prompt", type=str ,default="turn him a clown", help="Text prompt.")
    parser.add_argument("--edit_train_steps", type=int, default=1500, help="Edit train steps.")
    parser.add_argument("--cameara_update_step", type=int, default=500, help="Cameara Update Step.")
    parser.add_argument("--optimizer_type", type=str, default="sparse_adam", help="default or sparse_adam")

    parser.add_argument("--guidance_type", type=str, default="InstructPix2Pix")
    parser.add_argument("--edit_cam_num", type=int, default=0, help="Camera number.")
    parser.add_argument("--per_train_step", type=int, default=1, help="Per train step.")
    parser.add_argument("--per_editing_step", type=int, default=1, help="Per editing step.")
    parser.add_argument("--edit_begin_step", type=int, default=0, help="Edit begin step.")
    parser.add_argument("--edit_until_step", type=int, default=100, help="Edit until step.")
    parser.add_argument("--densification_interval", type=float, default=50, help="Densification interval.")
    parser.add_argument("--densify_until_step", type=int, default=1300, help="Densify until step.")
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="Lambda L1.")
    parser.add_argument("--lambda_p", type=int, default=2, help="Lambda P.")
    parser.add_argument("--lambda_anchor_color", type=float, default=1.0, help="Lambda anchor color.")
    parser.add_argument("--lambda_anchor_geo", type=float, default=1.0, help="Lambda anchor geo.")
    parser.add_argument("--lambda_anchor_scale", type=float, default=1.0, help="Lambda anchor scale.")
    parser.add_argument("--lambda_anchor_opacity", type=float, default=1.0, help="Lambda anchor opacity.")
    parser.add_argument("--video", type=str, default=True, help="video editing pattern.")
    parser.add_argument("--gs_lr_scaler", type=float, default=1.0, help="Initial learning rate scaler for GS.")
    parser.add_argument("--gs_lr_end_scaler", type=float, default=1.0, help="Final learning rate scaler for GS.")
    parser.add_argument("--color_lr_scaler", type=float, default=3.0, help="Learning rate scaler for color.")
    parser.add_argument("--opacity_lr_scaler", type=float, default=2.0, help="Learning rate scaler for opacity.")
    parser.add_argument("--scaling_lr_scaler", type=float, default=2.0, help="Learning rate scaler for scaling.")
    parser.add_argument("--rotation_lr_scaler", type=float, default=2.0, help="Learning rate scaler for rotation.")
    parser.add_argument("--use_original_resolution", type=str, default="False", help="use original resolution.")
    parser.add_argument("--output_dir", type=str, default="save/", help="output dir.")
    parser.add_argument(
        "--camera",
        type=str,
        default=(default_add_cache_dir / "camera.pkl").as_posix(),
        help="camera file path.",
    )
    parser.add_argument("--mask_thres", type=float, default=0.5, help="mask threshold.")

    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = TrainFineeAdd(args)
        trainer.gaussian.update_anchor_term(
            anchor_weight_init_g0=0.05,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=1.3,
        )
        trainer.configure_optimizers()
        
        trainer.edit(video=eval(args.video))
    end_time = time.time()
    print("Time taken:", end_time - start_time)
    
    
    
