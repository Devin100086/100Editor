import glob
import json
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ensure local package imports work regardless of current working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
for _path in (REPO_ROOT, SRC_DIR, THIRD_PARTY_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

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


def metric(origin_image_dir, edited_image_dir, clip_prompt_origin, clip_prompt_target):
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    origin_img_paths = [
        p
        for p in glob.glob(os.path.join(origin_image_dir, "*"))
        if p.lower().endswith(img_extensions)
    ]
    origin_img_paths.sort()
    edited_img_paths = [
        p
        for p in glob.glob(os.path.join(edited_image_dir, "*"))
        if p.lower().endswith(img_extensions)
    ]
    edited_img_paths.sort()

    assert len(origin_img_paths) == len(
        edited_img_paths
    ), "The number of images in the two directories must be the same."
    assert len(origin_img_paths) > 0, "No images found to evaluate."

    preprocess = transforms.Compose([transforms.ToTensor()])

    origin = []
    edited = []
    for img_path in origin_img_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img)
        origin.append(img_tensor)
    for img_path in edited_img_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img)
        edited.append(img_tensor)

    clip_metrics = ClipSimilarity().to("cuda")
    total_sim_direction = 0.0
    total_sim = 0.0
    with torch.no_grad():
        for i in tqdm(range(len(origin)), desc="Evaluating", unit="img"):
            origin_out = origin[i].unsqueeze(0).to("cuda")
            edited_out = edited[i].unsqueeze(0).to("cuda")
            _, sim, cos_sim, _ = clip_metrics(
                origin_out, edited_out, clip_prompt_origin, clip_prompt_target
            )
            total_sim_direction += abs(cos_sim.item())
            total_sim += abs(sim.item())

    avg_sim = total_sim / len(origin)
    avg_sim_direction = total_sim_direction / len(origin)
    return {
        "image_count": len(origin),
        "clip_prompt_origin": clip_prompt_origin,
        "clip_prompt_target": clip_prompt_target,
        "origin_image_dir": os.path.abspath(origin_image_dir),
        "edited_image_dir": os.path.abspath(edited_image_dir),
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
    parser.add_argument("--origin_image_dir", type=str, required=True)
    parser.add_argument("--edited_image_dir", type=str, required=True)
    parser.add_argument("--clip_prompt_origin", type=str, required=True)
    parser.add_argument("--clip_prompt_target", type=str, required=True)

    args = parser.parse_args()
    start_time = time.time()

    print(_style("=" * 72, _Ansi.BLUE))
    print(_style("Image CLIP Evaluation", _Ansi.BOLD, _Ansi.CYAN))
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
        f"{_style('Origin Dir', _Ansi.BOLD)}    : "
        f"{_style(os.path.abspath(args.origin_image_dir), _Ansi.MAGENTA)}"
    )
    print(
        f"{_style('Edited Dir', _Ansi.BOLD)}    : "
        f"{_style(os.path.abspath(args.edited_image_dir), _Ansi.MAGENTA)}"
    )
    print(_style("=" * 72, _Ansi.BLUE))

    result = metric(
        args.origin_image_dir,
        args.edited_image_dir,
        args.clip_prompt_origin,
        args.clip_prompt_target,
    )
    result["elapsed_seconds"] = round(time.time() - start_time, 3)
    json_path = save_result_json(result)
    sim_text = f"{result['sim']:.6f}"
    sim_direction_text = f"{result['sim_direction']:.6f}"
    elapsed_text = f"{result['elapsed_seconds']:.3f}"

    print("\n" + _style("Evaluation Summary", _Ansi.BOLD, _Ansi.CYAN))
    print(_style("-" * 72, _Ansi.BLUE))
    print(f"{_style('Images', _Ansi.BOLD)}         : {_style(str(result['image_count']), _Ansi.GREEN)}")
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
