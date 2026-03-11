import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from met3r import MEt3R

DEFAULT_IMG_SIZE = 512
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


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
    img_size = None if args.img_size == 0 else args.img_size

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

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
            print(f"{prev_path.name} -> {image_path.name}: {score_value:.6f}")
            prev_path = image_path
            prev_image = curr_image

        avg_score = float(np.mean(scores))
        print(
            "Average MEt3R score "
            f"({args.distance}) over {len(scores)} pairs: {avg_score:.6f}"
        )
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
        print(f"MEt3R score ({args.distance}): {score_value:.6f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# python Met3r.py --image-dir /media/wucunqi/data/100Editor/exp/Comparison/DGE/Turn_his_face_into_vampire@20251109-163039/save/it1500-test --distance cosine --img-size 256
