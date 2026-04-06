import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

# Ensure local package imports work regardless of current working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
for _path in (REPO_ROOT, SRC_DIR, THIRD_PARTY_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from met3r import MEt3R

DEFAULT_IMG_SIZE = 512
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


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
    eval_dir = repo_root / "runtime" / "eval" / "met3r"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


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


def load_image(path: str, img_size: Optional[int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if img_size is not None:
        image = image.resize((img_size, img_size), Image.BICUBIC)
    image = torch.from_numpy(np.array(image)).float() / 255.0
    image = image.permute(2, 0, 1)
    image = image * 2.0 - 1.0
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute MEt3R score for two images, or for every consecutive pair "
            "in a folder (sliding window)."
        )
    )
    parser.add_argument(
        "image_a",
        type=str,
        nargs="?",
        help="Path to the first image (two-image mode).",
    )
    parser.add_argument(
        "image_b",
        type=str,
        nargs="?",
        help="Path to the second image (two-image mode).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Path to a folder of images. Uses sliding-window pairs.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="Resize to a square size. Use 0 to keep original resolution.",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="cosine",
        choices=["cosine", "lpips", "rmse", "psnr", "mse", "ssim"],
        help="Distance metric for MEt3R.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--verbose-pairs",
        action="store_true",
        help="Print every pair score in folder mode.",
    )
    return parser.parse_args()


def _natural_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_image_paths(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")
    paths = [
        path
        for path in sorted(root.iterdir(), key=lambda p: _natural_key(p.name))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"No supported images found in: {image_dir}")
    return paths


def compute_score(metric: MEt3R, image_a: torch.Tensor, image_b: torch.Tensor) -> float:
    inputs = torch.stack([image_a, image_b], dim=0).unsqueeze(0)
    with torch.no_grad():
        score, *_ = metric(
            images=inputs,
            return_overlap_mask=False,
            return_score_map=False,
            return_projections=False,
        )
    return score.mean().item()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    img_size = None if args.img_size == 0 else args.img_size
    img_size_display = "original" if img_size is None else str(img_size)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(_style("=" * 72, _Ansi.BLUE))
    print(_style("MEt3R Evaluation", _Ansi.BOLD, _Ansi.CYAN))
    print(_style("-" * 72, _Ansi.BLUE))
    print(f"{_style('Mode', _Ansi.BOLD)}         : {_style('folder' if args.image_dir else 'pair', _Ansi.YELLOW)}")
    print(f"{_style('Distance', _Ansi.BOLD)}     : {_style(args.distance, _Ansi.YELLOW)}")
    print(f"{_style('Image Size', _Ansi.BOLD)}   : {_style(img_size_display, _Ansi.YELLOW)}")
    print(f"{_style('Device', _Ansi.BOLD)}       : {_style(str(device), _Ansi.YELLOW)}")
    if args.image_dir:
        print(
            f"{_style('Image Dir', _Ansi.BOLD)}    : "
            f"{_style(str(Path(args.image_dir).resolve()), _Ansi.MAGENTA)}"
        )
    else:
        print(
            f"{_style('Image A', _Ansi.BOLD)}      : "
            f"{_style(str(Path(args.image_a).resolve()), _Ansi.MAGENTA)}"
        )
        print(
            f"{_style('Image B', _Ansi.BOLD)}      : "
            f"{_style(str(Path(args.image_b).resolve()), _Ansi.MAGENTA)}"
        )
    print(_style("=" * 72, _Ansi.BLUE))

    metric = MEt3R(
        img_size=img_size,
        use_norm=True,
        backbone="mast3r",
        feature_backbone="dino16",
        feature_backbone_weights="mhamilton723/FeatUp",
        upsampler="featup",
        distance=args.distance,
        freeze=True,
    ).to(device)
    metric.eval()

    if args.image_dir:
        image_paths = list_image_paths(args.image_dir)
        if len(image_paths) < 2:
            raise ValueError("Image directory must contain at least 2 images.")

        scores: List[float] = []
        pair_scores = []
        prev_path = image_paths[0]
        prev_image = load_image(str(prev_path), img_size).to(device)

        for image_path in image_paths[1:]:
            curr_image = load_image(str(image_path), img_size).to(device)
            if img_size is None and prev_image.shape != curr_image.shape:
                raise ValueError(
                    "All images must have the same resolution when --img-size 0 is used."
                )
            score_value = compute_score(metric, prev_image, curr_image)
            scores.append(score_value)
            pair_item = {
                "from": prev_path.name,
                "to": image_path.name,
                "score": score_value,
            }
            pair_scores.append(pair_item)
            if args.verbose_pairs:
                print(
                    f"{_style(prev_path.name, _Ansi.MAGENTA)} -> "
                    f"{_style(image_path.name, _Ansi.MAGENTA)} : "
                    f"{_style(f'{score_value:.6f}', _Ansi.GREEN)}"
                )
            prev_path = image_path
            prev_image = curr_image

        elapsed = round(time.time() - start_time, 3)
        result = {
            "mode": "folder",
            "image_dir": str(Path(args.image_dir).resolve()),
            "distance": args.distance,
            "img_size": img_size,
            "device": str(device),
            "image_count": len(image_paths),
            "pair_count": len(scores),
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "std_score": float(np.std(scores)),
            "elapsed_seconds": elapsed,
            "pair_scores": pair_scores,
        }
    else:
        if not args.image_a or not args.image_b:
            raise ValueError(
                "Please provide two image paths or use --image-dir for folder mode."
            )

        image_a = load_image(args.image_a, img_size).to(device)
        image_b = load_image(args.image_b, img_size).to(device)
        if img_size is None and image_a.shape != image_b.shape:
            raise ValueError(
                "Input images must have the same resolution when --img-size 0 is used."
            )

        score_value = compute_score(metric, image_a, image_b)
        elapsed = round(time.time() - start_time, 3)
        result = {
            "mode": "pair",
            "image_a": str(Path(args.image_a).resolve()),
            "image_b": str(Path(args.image_b).resolve()),
            "distance": args.distance,
            "img_size": img_size,
            "device": str(device),
            "score": score_value,
            "elapsed_seconds": elapsed,
        }

    json_path = save_result_json(result)

    print("\n" + _style("Evaluation Summary", _Ansi.BOLD, _Ansi.CYAN))
    print(_style("-" * 72, _Ansi.BLUE))
    if result["mode"] == "folder":
        mean_text = f"{result['mean_score']:.6f}"
        min_text = f"{result['min_score']:.6f}"
        max_text = f"{result['max_score']:.6f}"
        std_text = f"{result['std_score']:.6f}"
        print(f"{_style('Images', _Ansi.BOLD)}       : {_style(str(result['image_count']), _Ansi.GREEN)}")
        print(f"{_style('Pairs', _Ansi.BOLD)}        : {_style(str(result['pair_count']), _Ansi.GREEN)}")
        print(
            f"{_style('Mean Score', _Ansi.BOLD)}   : "
            f"{_style(mean_text, _Ansi.GREEN)}"
        )
        print(
            f"{_style('Min / Max', _Ansi.BOLD)}    : "
            f"{_style(f'{min_text} / {max_text}', _Ansi.GREEN)}"
        )
        print(
            f"{_style('Std Score', _Ansi.BOLD)}    : "
            f"{_style(std_text, _Ansi.GREEN)}"
        )
    else:
        score_text = f"{result['score']:.6f}"
        print(
            f"{_style('Score', _Ansi.BOLD)}        : "
            f"{_style(score_text, _Ansi.GREEN)}"
        )
    elapsed_text = f"{result['elapsed_seconds']:.3f}"
    print(
        f"{_style('Elapsed (s)', _Ansi.BOLD)}   : "
        f"{_style(elapsed_text, _Ansi.GREEN)}"
    )
    print(f"{_style('Result JSON', _Ansi.BOLD)}  : {_style(json_path, _Ansi.MAGENTA)}")
    print(_style("=" * 72, _Ansi.BLUE))

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
