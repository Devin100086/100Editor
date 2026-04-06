import os
from glob import glob
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from einops import repeat
from omegaconf import OmegaConf
import sys

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

try:
    from .ldm.models.diffusion.ddim import DDIMSampler
    from .ldm.util import instantiate_from_config
    from .test_inpainting import load_state_dict, torch_init_model
except ImportError:
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.util import instantiate_from_config
    from test_inpainting import load_state_dict, torch_init_model

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# torch.set_grad_enabled(False)

target_image_size = 512
repeat_sp_token = 50
sp_token = "<special-token>"


def _resolve_model_root() -> Path:
    env_model_path = os.getenv("LEFTREFILL_MODEL_PATH")
    repo_root = PACKAGE_DIR.parent.parent
    candidates = []
    if env_model_path:
        candidates.append(Path(env_model_path).expanduser())
    candidates.extend(
        [
            repo_root / "runtime" / ".cache" / "LeftRefill" / "check_points" / "ref_guided_inpainting",
            Path.cwd() / "runtime" / ".cache" / "LeftRefill" / "check_points" / "ref_guided_inpainting",
            Path.home() / "runtime" / ".cache" / "LeftRefill" / "check_points" / "ref_guided_inpainting",
            PACKAGE_DIR / "check_points" / "ref_guided_inpainting",
        ]
    )

    for candidate in candidates:
        if (candidate / "model_config.yaml").exists() and (candidate / "ckpts").exists():
            return candidate

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Cannot find LeftRefill model directory. Set LEFTREFILL_MODEL_PATH or place files in one of:\n"
        f"{checked}"
    )


def _resolve_pretrained_model_path(model_root: Path) -> Path:
    env_pretrained_path = os.getenv("LEFTREFILL_PRETRAINED_PATH")
    candidates = []
    if env_pretrained_path:
        candidates.append(Path(env_pretrained_path).expanduser())
    candidates.extend(
        [
            model_root.parent.parent / "pretrained_models" / "512-inpainting-ema.ckpt",
            PACKAGE_DIR.parent.parent / ".cache" / "LeftRefill" / "pretrained_models" / "512-inpainting-ema.ckpt",
            Path.cwd() / ".cache" / "LeftRefill" / "pretrained_models" / "512-inpainting-ema.ckpt",
            Path.home() / ".cache" / "LeftRefill" / "pretrained_models" / "512-inpainting-ema.ckpt",
            PACKAGE_DIR / "pretrained_models" / "512-inpainting-ema.ckpt",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Cannot find LeftRefill pretrained model. Set LEFTREFILL_PRETRAINED_PATH or place file in one of:\n"
        f"{checked}"
    )


root_path = _resolve_model_root()
pretrained_model_path = _resolve_pretrained_model_path(root_path)


def initialize_model(path):
    path = Path(path)
    print(f"LeftRefill model root: {path}")
    print(f"LeftRefill pretrained model: {pretrained_model_path}")
    config = OmegaConf.load(path / "model_config.yaml")
    model = instantiate_from_config(config.model)
    # repeat_sp_token = config['model']['params']['data_config']['repeat_sp_token']
    # sp_token = config['model']['params']['data_config']['sp_token']

    ckpt_list = glob(os.path.join(str(path), 'ckpts/epoch=*.ckpt'))
    if len(ckpt_list) > 1:
        resume_path = sorted(ckpt_list, key=lambda x: int(x.split('/')[-1].split('.ckpt')[0].split('=')[-1]))[-1]
    else:
        resume_path = ckpt_list[0]
    print('Load ckpt', resume_path)

    reload_weights = load_state_dict(resume_path, location='cpu')
    torch_init_model(model, reload_weights, key='none')
    if getattr(model, 'save_prompt_only', False):
        pretrained_weights = load_state_dict(str(pretrained_model_path), location='cpu')
        torch_init_model(model, pretrained_weights, key='none')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)
        print(batch['image'].shape)
        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples)
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)
        pred = x_samples_ddim * batch['mask'] + batch['image'] * (1 - batch['mask'])

        result = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)

        result = (result.cpu().numpy().transpose(0, 2, 3, 1) * 255)
        result = result[:, :, 512:]

    return [Image.fromarray(img.astype(np.uint8)) for img in result]
    # return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

sampler = initialize_model(path=root_path)
def predict(source, reference, ddim_steps, num_samples, scale, seed):
    torch.set_grad_enabled(False)
    source_img = source["image"].convert("RGB")
    origin_w, origin_h = source_img.size
    ratio = origin_h / origin_w
    init_mask = source["mask"].convert("RGB")
    print('Source...', source_img.size)
    reference_img = reference.convert("RGB")
    print('Reference...', reference_img.size)
    # if min(width, height) > image_size_limit:
    #     if width > height:
    #         init_image = init_image.resize((int(width / (height / image_size_limit)), image_size_limit), resample=Image.BICUBIC)
    #         init_mask = init_mask.resize((int(width / (height / image_size_limit)), image_size_limit), resample=Image.LINEAR)
    #     else:
    #         init_image = init_image.resize((image_size_limit, int(height / (width / image_size_limit))), resample=Image.BICUBIC)
    #         init_mask = init_mask.resize((image_size_limit, int(height / (width / image_size_limit))), resample=Image.LINEAR)
    #     init_mask = np.array(init_mask)
    #     init_mask[init_mask > 0] = 255
    #     init_mask = Image.fromarray(init_mask)

    # directly resizing to 512x512
    source_img = source_img.resize((target_image_size, target_image_size), resample=Image.Resampling.BICUBIC)
    reference_img = reference_img.resize((target_image_size, target_image_size), resample=Image.Resampling.BICUBIC)
    init_mask = init_mask.resize((target_image_size, target_image_size), resample=Image.Resampling.BILINEAR)
    init_mask = np.array(init_mask)
    init_mask[init_mask > 0] = 255
    init_mask = Image.fromarray(init_mask)

    source_img = pad_image(source_img)  # resize to integer multiple of 32
    reference_img = pad_image(reference_img)
    mask = pad_image(init_mask)  # resize to integer multiple of 32
    width, height = source_img.size
    width *= 2
    print("Inpainting...", width, height)
    # print("Prompt:", prompt)

    # get inputs
    image = np.concatenate([np.asarray(reference_img), np.asarray(source_img)], axis=1)
    image = Image.fromarray(image)
    mask = np.asarray(mask)
    mask = np.concatenate([np.zeros_like(mask), mask], axis=1)
    mask = Image.fromarray(mask)

    prompt = ""
    for i in range(repeat_sp_token):
        prompt = prompt + sp_token.replace('>', f'{i}> ')
    prompt = prompt.strip()
    # print('Prompt:', prompt)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )
    result = [r.resize((int(512 / ratio), 512), resample=Image.Resampling.BICUBIC) for r in result]
    for r in result:
        print(r.size)

    return result


# source = {"image":Image.open("output_images2.png"), "mask":Image.open("mask2.png")}
# reference = Image.open("image2.png")

# A = predict(source, reference, 25, 1, 2.5, 124241)
# print(A)
