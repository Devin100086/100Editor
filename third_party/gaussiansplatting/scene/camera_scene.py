import os

from .dataset_readers import sceneLoadTypeCallbacks
from ..utils.camera_utils import cameraList_load


class CamScene:
    def __init__(self, source_path, h=512, w=512, aspect=-1):
        if aspect != -1:
            h = 512
            w = int(512 * aspect)

        if not os.path.exists(os.path.join(source_path, "sparse")):
            raise AssertionError("Could not recognize scene type!")

        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, None, False)
        if h == -1 or w == -1:
            h = scene_info.train_cameras[0].height
            w = scene_info.train_cameras[0].width

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras = cameraList_load(scene_info.train_cameras, h, w)

