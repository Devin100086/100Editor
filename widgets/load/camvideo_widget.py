import os
from imgui_bundle import imgui
import tkinter as tk
import torch
from tkinter import filedialog
from gaussiansplatting.scene.cameras import CustomCam
from HundredEditor_utils.gui_utils import imgui_utils
from HundredEditor_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
from scipy.spatial.transform import Rotation as R
from HundredEditor_utils.gui_utils.constants import *
import numpy as np
import copy

class CamvideoWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "CamVideo")
        self.num_frames = 100
        self.cam = []

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        viz.args.video_cams = []
        if show:
            label(f"Camearas:  {len(self.cam)}", viz.label_w)
            if imgui_utils.button("Add Cameras", viz.button_w*1.5):
                self.cam.append({
                    "camera":CustomCam(
                        width=viz.args.resolution,
                        height=viz.args.resolution,
                        fovy=viz.args.fov / 360 * 2 * np.pi,
                        fovx=viz.args.fov / 360 * 2 * np.pi,
                        znear=0.01,
                        zfar=100,
                        extr=viz.args.cam_params.to("cuda"),
                    ),
                    "pose":copy.deepcopy(viz.load_widgets[1].pose),
                    "radius":copy.deepcopy(viz.load_widgets[1].radius),
                    "lookat_point":copy.deepcopy(viz.load_widgets[1].lookat_point),
                    "up_vector":copy.deepcopy(viz.load_widgets[1].up_vector),
                }
                )
            imgui.same_line()
            if imgui_utils.button("Clean Cameras", viz.button_large_w):
                self.cam.clear()

            size = imgui.ImVec2(0, 80)
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

            label("Interpolations", viz.label_w)
            _changed, self.num_frames = imgui.input_int("##Interpolations", self.num_frames)
            if imgui_utils.button("Create Video", viz.button_w*1.5):
                for i in range(len(self.cam) - 1):
                    cam0 = self.cam[i]
                    cam1 = self.cam[i + 1]
                    R0 = cam0["camera"].extr[:3, :3]
                    R1 = cam1["camera"].extr[:3, :3]
                    t0 = cam0["camera"].extr[:3, 3]
                    t1 = cam1["camera"].extr[:3, 3]
                    q0 = R.from_matrix(R0).as_quat()
                    q1 = R.from_matrix(R1).as_quat()
                    for j in range(self.num_frames + 1):
                        t = j / (self.num_frames + 1)
                        q_interp = self.slerp(q0, q1, t)
                        t_interp = (1 - t) * t0 + t * t1
                        R_interp = R.from_quat(q_interp).as_matrix()
                        extr_interp = np.eye(4)
                        extr_interp[:3, :3] = R_interp
                        extr_interp[:3, 3] = t_interp
                        new_cam = CustomCam(
                            width=viz.args.resolution,
                            height=viz.args.resolution,
                            fovy=cam0["camera"].FoVy,
                            fovx=cam0["camera"].FoVx,
                            znear=cam0["camera"].znear,
                            zfar=cam0["camera"].zfar,
                            extr=torch.from_numpy(extr_interp).to(torch.float32).to("cuda")
                        )
                        viz.args.video_cams.append(new_cam)


    def slerp(self, q0, q1, t):
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.cos(theta_t) - dot * sin_theta_t / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        return s0 * q0 + s1 * q1
