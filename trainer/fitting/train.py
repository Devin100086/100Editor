import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs
from utils import image_path_to_tensor
from runner import SimpleTrainer

def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = False,
    img_path: Optional[Path] = None,
    iterations: int = 2000,
    lr: float = 0.01,
    alpha: float = 0.99,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        alpha=alpha,
        save_imgs=save_imgs,
        model_type=model_type,
    )


if __name__ == "__main__":
    tyro.cli(main,verbose=True)
