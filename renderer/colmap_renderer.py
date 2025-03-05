import copy
import traceback
from typing import List
import numpy as np
import pycolmap
import torch
import torch.nn

from gaussiansplatting.gaussian_renderer import render_colmap
from renderer.base_renderer import Renderer
from lumina3D_utils.dict_utils import EasyDict


class ColmapRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.point_xyz: List = [None] * num_parallel_scenes
        self.point_color: List = [None] * num_parallel_scenes
        self._current_colmap_file_paths: List = [None] * num_parallel_scenes

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

        if len(data_source) == 0:
            res.error = "Select a colmap folder"
            return
        
        if self._current_colmap_file_paths[0] != data_source:
            recon = pycolmap.Reconstruction(data_source)
            self.point_xyz[0] = [point3D.xyz for point3D in recon.points3D.values()]
            self.point_color[0] = [point3D.color/255 for point3D in recon.points3D.values()]
            self._current_colmap_file_paths[0] = data_source
            
        pointxyz = np.array(self.point_xyz[0])
        pointcolor = np.array(self.point_color[0])

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