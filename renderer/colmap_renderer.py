import copy
import traceback
from typing import List
import numpy as np
import pycolmap
import torch
import torch.nn

from gaussiansplatting.gaussian_renderer import render_colmap
from gaussiansplatting.scene import GaussianModel
from renderer.base_renderer import Renderer
from lumina3D_utils.dict_utils import EasyDict


class ColmapRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0

    def _render_impl(
        self,
        res,
        fov,
        resolution,
        cam_params,
        data_source,
        img_normalize=False,
        use_splitscreen=False,
        highlight_border=False,
        slider={},
        **other_args,
    ):
        slider = EasyDict(slider)

        recon = pycolmap.Reconstruction(data_source)
        point_xyz = [point3D.xyz for point3D in recon.points3D.values()]
        point_color = [point3D.color/255 for point3D in recon.points3D.values()]
        pointxyz = np.array(point_xyz)
        pointcolor = np.array(point_color)

        # Render current view
        fov_rad = fov / 360 * 2 * np.pi

        render = render_colmap(pointxyz, pointcolor, resolution, fov_rad, cam_params.numpy())

        image = torch.from_numpy(render).permute(2, 0, 1)
        self._return_image(
            image,
            res,
            normalize=img_normalize,
            use_splitscreen=use_splitscreen,
            highlight_border=highlight_border,
        )
    
    @staticmethod
    def close():
        pass