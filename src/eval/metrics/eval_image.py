import argparse
from tqdm import tqdm
import sys
import os
import glob
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser
from threestudio.utils.clip_metrics import *

def metric(origin_image_dir, edited_image_dir, clip_prompt_origin, clip_prompt_target):

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    origin_img_paths = [p for p in glob.glob(os.path.join(origin_image_dir, '*')) 
                 if p.lower().endswith(img_extensions)]
    origin_img_paths.sort()
    edited_img_paths = [p for p in glob.glob(os.path.join(edited_image_dir, '*')) 
                 if p.lower().endswith(img_extensions)]
    edited_img_paths.sort()

    assert len(origin_img_paths) == len(edited_img_paths), "The number of images in the two directories must be the same."

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    origin = []
    edited = []
    for img_path in origin_img_paths:
        img = Image.open(img_path).convert('RGB') 
        img_tensor = preprocess(img)
        origin.append(img_tensor)
    for img_path in edited_img_paths:
        img = Image.open(img_path).convert('RGB') 
        img_tensor = preprocess(img)
        edited.append(img_tensor)
    
    clip_metrics = ClipSimilarity().to("cuda")
    total_sim_direction = 0
    total_sim = 0
    with torch.no_grad():
        for i in tqdm(range(len(origin))):
            origin_out = origin[i].unsqueeze(0).to("cuda")
            edited_out = edited[i].unsqueeze(0).to("cuda")
            _, sim, cos_sim, _ = clip_metrics(origin_out, edited_out,
                                            clip_prompt_origin, clip_prompt_target)
            total_sim_direction += abs(cos_sim.item())
            total_sim += abs(sim.item())
    print(clip_prompt_origin, clip_prompt_target, "sim", total_sim / len(origin), "sim_direction", total_sim_direction / len(origin))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--origin_image_dir", type=str, required=True) 
    parser.add_argument("--edited_image_dir", type=str, required=True)
    parser.add_argument("--clip_prompt_origin", type=str, required=True)
    parser.add_argument("--clip_prompt_target", type=str, required=True)

    args = parser.parse_args()
    
    print("🚀Start Evaluating...")
    metric(args.origin_image_dir, args.edited_image_dir, args.clip_prompt_origin, args.clip_prompt_target)
    print("🌟Finish Evaluating!")