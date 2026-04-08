import contextlib
import io
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from einops import rearrange
from torchvision.transforms import ToPILImage, ToTensor

from lang_sam import LangSAM

# from threestudio.utils.typing import *


class LangSAMTextSegmentor(torch.nn.Module):
    def __init__(self, sam_type="vit_h", suppress_predict_logs: bool = True):
        super().__init__()
        # self.model = LangSAM(sam_type)
        self.model = LangSAM()
        self.suppress_predict_logs = suppress_predict_logs

        self.to_pil_image = ToPILImage(mode="RGB")
        self.to_tensor = ToTensor()

    def forward(self, images, prompt: str):
        images = rearrange(images, "b h w c -> b c h w")
        masks = []
        for image in images:
            # breakpoint()
            image = self.to_pil_image(image.clamp(0.0, 1.0))
            if self.suppress_predict_logs:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    predict_out = self.model.predict([image], [prompt])
            else:
                predict_out = self.model.predict([image], [prompt])
            mask = predict_out[0]['masks']
            if isinstance(mask, list):
                mask = torch.zeros_like(images[0, 0:1])
            else:
                mask = torch.from_numpy(mask)
            # breakpoint()
            if mask.ndim == 3:
                masks.append(mask[0:1].to(torch.float32))
            else:
                if not self.suppress_predict_logs:
                    print(f"None {prompt} Detected")
                masks.append(torch.zeros_like(images[0, 0:1]))

        return torch.stack(masks, dim=0)


if __name__ == "__main__":
    model = LangSAMTextSegmentor()

    image = Image.open("load/lego_bulldozer.jpg")
    prompt = "a lego bulldozer"

    image = ToTensor()(image)

    image = image.unsqueeze(0)

    mask = model(image, prompt)

    breakpoint()
