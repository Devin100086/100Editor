import os
from imgui_bundle import imgui
import tkinter as tk
import torch
from tkinter import filedialog
from gaussiansplatting.scene.cameras import CustomCam
from lumina3D_utils.gui_utils import imgui_utils
from lumina3D_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
from scipy.spatial.transform import Rotation as R
from lumina3D_utils.gui_utils.constants import *
import numpy as np
import copy
from gaussiansplatting.scene.dataset_readers import sceneLoadTypeCallbacks
from scipy.spatial.transform import Rotation as R

class Pose:
    def __init__(self, yaw=0, pitch=0):
        self.yaw = yaw
        self.pitch = pitch

class LoadCameraWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "LoadCamera")
        self.data_source = ""
        self.cam = []

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            if imgui_utils.button(f"Browse data", width=viz.button_large_w):
                data_source = self._select_folder()
                self.data_source = self.data_source if isinstance(data_source, tuple) else data_source
                try:
                    scene_info = sceneLoadTypeCallbacks["Colmap"](self.data_source, None, False)
                    for i, cam in enumerate(scene_info.train_cameras):
                        pose = np.eye(4)
                        pose[:3, :3] = cam.R
                        pose[:3, 3] = cam.T
                        radius = np.linalg.norm(cam.T)
                        lookat_direction = pose[:3, 1] 
                        lookat_point = cam.T + lookat_direction
                        up_vector = -pose[:3, 1]
                        rotation = R.from_quat(cam.qvec)
                        yaw, pitch, roll = rotation.as_euler('xyz', degrees=False)
                        self.cam.append({
                            "camera": CustomCam(
                                width=viz.args.resolution,
                                height=viz.args.resolution,
                                fovy=viz.args.fov / 360 * 2 * np.pi,
                                fovx=viz.args.fov / 360 * 2 * np.pi,
                                znear=0.01,
                                zfar=100,
                                extr=viz.args.cam_params.to("cuda"),
                            ),
                            "pose": Pose(yaw=yaw, pitch=pitch),
                            "radius": radius,
                            "lookat_point": torch.from_numpy(lookat_point).to(torch.float32),
                            "up_vector": torch.from_numpy(up_vector).to(torch.float32),
                        })
                except:
                    print("Error loading cameras from folder.")
            print(viz.load_widgets[1].lookat_point)
            imgui.same_line()
            if imgui_utils.button("Clear Cameras", viz.button_large_w):
                self.cam.clear()

            size = imgui.ImVec2(0, 200)
            imgui.begin_child("Cameras", size=size, window_flags=WINDOW_HORIZONTAL_SCROLLING_BAR)
            for i, camera in enumerate(self.cam):
                label(f"Camera {i+1}", viz.label_w)
                imgui.same_line()
                if imgui_utils.button(f"Remove{i+1}", viz.button_large_w):
                    self.cam.pop(i)
                imgui.same_line()
                if imgui_utils.button(f"See{i+1}", viz.button_large_w):
                    viz.load_widgets[1].pose = self.cam[i]["pose"]
                    viz.load_widgets[1].radius = self.cam[i]["radius"]
                    viz.load_widgets[1].lookat_point = self.cam[i]["lookat_point"]
                    viz.load_widgets[1].up_vector = self.cam[i]["up_vector"]
            imgui.end_child()

    def _select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        return folder_path