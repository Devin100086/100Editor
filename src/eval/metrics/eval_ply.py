import json
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm

# Ensure local package imports work regardless of current working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
for _path in (REPO_ROOT, SRC_DIR, THIRD_PARTY_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from editor.gaussiansplatting.arguments import PipelineParams
from editor.gaussiansplatting.gaussian_renderer import render
from editor.gaussiansplatting.scene import GaussianModel
from editor.gaussiansplatting.scene.camera_scene import CamScene
from threestudio.utils.clip_metrics import *


def _supports_color() -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    term = os.getenv("TERM", "")
    return term != "" and term.lower() != "dumb"


_COLOR_ENABLED = _supports_color()


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


def _style(text: str, *codes: str) -> str:
    if not _COLOR_ENABLED:
        return text
    return "".join(codes) + text + _Ansi.RESET


def get_runtime_eval_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    eval_dir = repo_root / "runtime" / "eval" / "clip"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def metric(
    origin_gaussian,
    edited_gaussian,
    colmap_dir,
    use_original_resolution,
    clip_prompt_origin,
    clip_prompt_target,
):
    parser = ArgumentParser(description="Training script parameters")
    pipe = PipelineParams(parser)
    background_tensor = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    if use_original_resolution != 0:
        scene = CamScene(colmap_dir, h=-1, w=-1)
    else:
        scene = CamScene(colmap_dir, h=512, w=512)
    colmap_cameras = scene.cameras

    clip_metrics = ClipSimilarity().to(origin_gaussian.get_xyz.device)
    total_sim_direction = 0.0
    total_sim = 0.0
    with torch.no_grad():
        for cam in tqdm(colmap_cameras, desc="Evaluating", unit="cam"):
            origin_render_pkg = render(cam, origin_gaussian, pipe, background_tensor, separate_sh=True)
            origin_out = origin_render_pkg["render"].unsqueeze(0)

            edited_render_pkg = render(cam, edited_gaussian, pipe, background_tensor, separate_sh=True)
            edited_out = edited_render_pkg["render"].unsqueeze(0)

            _, sim, cos_sim, _ = clip_metrics(
                origin_out, edited_out, clip_prompt_origin, clip_prompt_target
            )
            total_sim_direction += abs(cos_sim.item())
            total_sim += abs(sim.item())

    avg_sim = total_sim / len(colmap_cameras)
    avg_sim_direction = total_sim_direction / len(colmap_cameras)
    return {
        "camera_count": len(colmap_cameras),
        "clip_prompt_origin": clip_prompt_origin,
        "clip_prompt_target": clip_prompt_target,
        "use_original_resolution": use_original_resolution,
        "sim": avg_sim,
        "sim_direction": avg_sim_direction,
    }


def save_result_json(result: dict) -> str:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    eval_dir = get_runtime_eval_dir()
    json_path = eval_dir / f"{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **result,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return json_path.as_posix()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--clip_prompt_origin", type=str, required=True)
    parser.add_argument("--clip_prompt_target", type=str, required=True)
    parser.add_argument("--origin_gs_source", type=str, required=True)
    parser.add_argument("--edited_gs_source", type=str, required=True)
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--use_original_resolution", type=int, default=0)

    args = parser.parse_args()
    start_time = time.time()

    origin_gaussian = GaussianModel(
        sh_degree=0,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2,
    )
    origin_gaussian.load_ply(args.origin_gs_source)
    origin_gaussian.max_radii2D = torch.zeros(
        (origin_gaussian.get_xyz.shape[0]), device="cuda"
    )

    edited_gaussian = GaussianModel(
        sh_degree=0,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2,
    )
    edited_gaussian.load_ply(args.edited_gs_source)
    edited_gaussian.max_radii2D = torch.zeros(
        (edited_gaussian.get_xyz.shape[0]), device="cuda"
    )

    print(_style("=" * 72, _Ansi.BLUE))
    print(_style("PLY CLIP Evaluation", _Ansi.BOLD, _Ansi.CYAN))
    print(_style("-" * 72, _Ansi.BLUE))
    print(
        f"{_style('Origin Prompt', _Ansi.BOLD)} : "
        f"{_style(args.clip_prompt_origin, _Ansi.YELLOW)}"
    )
    print(
        f"{_style('Target Prompt', _Ansi.BOLD)} : "
        f"{_style(args.clip_prompt_target, _Ansi.YELLOW)}"
    )
    print(
        f"{_style('Origin PLY', _Ansi.BOLD)}    : "
        f"{_style(os.path.abspath(args.origin_gs_source), _Ansi.MAGENTA)}"
    )
    print(
        f"{_style('Edited PLY', _Ansi.BOLD)}    : "
        f"{_style(os.path.abspath(args.edited_gs_source), _Ansi.MAGENTA)}"
    )
    print(
        f"{_style('Colmap Dir', _Ansi.BOLD)}    : "
        f"{_style(os.path.abspath(args.colmap_dir), _Ansi.MAGENTA)}"
    )
    print(_style("=" * 72, _Ansi.BLUE))

    result = metric(
        origin_gaussian,
        edited_gaussian,
        args.colmap_dir,
        args.use_original_resolution,
        args.clip_prompt_origin,
        args.clip_prompt_target,
    )
    result["origin_gs_source"] = os.path.abspath(args.origin_gs_source)
    result["edited_gs_source"] = os.path.abspath(args.edited_gs_source)
    result["colmap_dir"] = os.path.abspath(args.colmap_dir)
    result["elapsed_seconds"] = round(time.time() - start_time, 3)
    json_path = save_result_json(result)
    sim_text = f"{result['sim']:.6f}"
    sim_direction_text = f"{result['sim_direction']:.6f}"
    elapsed_text = f"{result['elapsed_seconds']:.3f}"

    print("\n" + _style("Evaluation Summary", _Ansi.BOLD, _Ansi.CYAN))
    print(_style("-" * 72, _Ansi.BLUE))
    print(f"{_style('Cameras', _Ansi.BOLD)}        : {_style(str(result['camera_count']), _Ansi.GREEN)}")
    print(f"{_style('CLIP Sim', _Ansi.BOLD)}       : {_style(sim_text, _Ansi.GREEN)}")
    print(
        f"{_style('CLIP Direction', _Ansi.BOLD)} : "
        f"{_style(sim_direction_text, _Ansi.GREEN)}"
    )
    print(
        f"{_style('Elapsed (s)', _Ansi.BOLD)}    : "
        f"{_style(elapsed_text, _Ansi.GREEN)}"
    )
    print(f"{_style('Result JSON', _Ansi.BOLD)}    : {_style(json_path, _Ansi.MAGENTA)}")
    print(_style("=" * 72, _Ansi.BLUE))
