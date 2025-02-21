from imgui_bundle import imgui
import torch
import numpy as np

from lumina3D_utils.gui_utils.easy_imgui import label, slider, checkbox
from lumina3D_utils.gui_utils import imgui_utils
from lumina3D_utils.dict_utils import EasyDict
from lumina3D_utils.cam_utils import (
    get_forward_vector,
    create_cam2world_matrix,
    get_origin,
    normalize_vecs,
)
from widgets.widget import Widget
from widgets.common import cam_widget

class EditcamWidget(cam_widget.CamWidget):
    def __init__(self, viz):
        super().__init__(viz)
    
    @imgui_utils.scoped_by_object_id
    def __call__(self, show: bool):
        viz = self.viz
        active_region = EasyDict(x=viz.pane_w, y=0, width=viz.content_width - viz.pane_w, height=viz.content_height)
        if not viz.args.painting and not viz.args.mask:
            self.handle_dragging_in_window(**active_region)
            self.handle_mouse_wheel()
            if not viz.args.text_change:
                self.handle_wasd()

        if show:
            label("Camera Mode", viz.label_w)
            _, self.current_control_mode = imgui.combo("##cam_modes", self.current_control_mode, self.control_modes)

            if self.control_modes[self.current_control_mode] == "WASD":
                label("Move Speed", viz.label_w)
                self.wasd_move_speed = slider(self.wasd_move_speed, "move_speed", 0.001, 1, log=True)

            label("Drag Speed", viz.label_w)
            self.drag_speed = slider(self.drag_speed, "drag_speed", 0.001, 0.1, log=True)

            label("Rotate Speed", viz.label_w)
            self.rotate_speed = slider(self.rotate_speed, "rot_speed", 0.001, 0.1, log=True)

            imgui.push_item_width(200)
            label("Up Vector", viz.label_w)
            _changed, up_vector_tuple = imgui.input_float3("##up_vector", v=self.up_vector.tolist(), format="%.1f")
            if _changed:
                self.up_vector = torch.tensor(up_vector_tuple)

            imgui.same_line()
            if imgui_utils.button("Set current direction", width=viz.button_large_w):
                self.up_vector = self.forward
                self.pose.yaw = 0
                self.pose.pitch = 0

            imgui.same_line()
            if imgui_utils.button("Flip", width=viz.button_w):
                self.up_vector = -self.up_vector

            label("FOV", viz.label_w)
            self.fov = slider(self.fov, "##fov", 1, 180, format="%.1f °")

            if self.control_modes[self.current_control_mode] == "Orbit":
                label("Radius", viz.label_w)
                self.radius = slider(self.radius, "##radius", 1, 20, format="%.1f °")

                imgui.same_line()
                if imgui_utils.button("Set to xyz stddev", width=viz.button_large_w) and "std_xyz" in viz.result.keys():
                    self.radius = viz.result.std_xyz.item()

                label("Look at Point", viz.label_w)
                _, look_at_point_tuple = imgui.input_float3("##lookat", self.lookat_point.tolist(), format="%.1f")
                self.lookat_point = torch.tensor(look_at_point_tuple)
                imgui.same_line()
                if imgui_utils.button("Set to xyz mean", width=viz.button_large_w) and "mean_xyz" in viz.result.keys():
                    self.lookat_point = viz.result.mean_xyz
            imgui.pop_item_width()

            label("Invert X", viz.label_w)
            self.invert_x = checkbox(self.invert_x, "invert_x")
            label("Invert Y", viz.label_w)
            self.invert_y = checkbox(self.invert_y, "invert_y")

        self.cam_params = create_cam2world_matrix(self.forward, self.cam_pos, self.up_vector)[0]
        viz.args.yaw = self.pose.yaw
        viz.args.pitch = self.pose.pitch
        viz.args.fov = self.fov
        viz.fov = self.fov
        viz.args.cam_params = self.cam_params
        viz.extr = self.cam_params
        
        # params for the video widget
        viz.args.lookat_point = self.lookat_point
        viz.args.up_vector = self.up_vector