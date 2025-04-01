from argparse import ArgumentParser
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.GUIEditor.utils import *
from EditorGS.gaussiansplatting.scene.camera_scene import CamScene

from torchvision.utils import save_image

def render_cameras_list(gaussian, colmap_dir, save_dir):
    parser = ArgumentParser(description="Training script parameters")
    pipe = PipelineParams(parser)
    background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
    scene = CamScene(colmap_dir, h=512, w=512)
    colmap_cameras = scene.cameras

    os.makedirs(save_dir, exist_ok=True)

    for cam in tqdm(colmap_cameras):
        render_pkg = render(cam, gaussian, pipe, background_tensor)
        out = render_pkg["render"]
        image = out[None]
        save_image(image, f"{save_dir}/{cam.image_name}.png")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="save/")

    args = parser.parse_args()
    gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
    gaussian.load_ply(args.gs_source)
    gaussian.max_radii2D = torch.zeros(
        (gaussian.get_xyz.shape[0]), device="cuda"
    )
    print("🚀Start rendering...")
    render_cameras_list(gaussian, args.colmap_dir, args.save_dir)
    print("🌟Finish rendering!")
    
    