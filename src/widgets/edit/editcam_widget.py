from imgui_bundle import imgui
import torch
import numpy as np

from utils.gui_utils.easy_imgui import label, slider, checkbox
from utils.gui_utils import imgui_utils
from utils.dict_utils import EasyDict
from utils.cam_utils import (
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
        dt = self._tick_delta_time()
        active_region = EasyDict(x=viz.pane_w, y=0, width=viz.content_width - viz.pane_w, height=viz.content_height)
        viz.args.show_image = True
        if not viz.args.painting:
            self.handle_dragging_in_window(**active_region)
            self.handle_mouse_wheel()
            if "mean_xyz" in viz.result.keys() and not torch.allclose(self.center, viz.result.mean_xyz.cpu()):
                target = viz.result.mean_xyz.cpu()
                self._set_lookat_target(target, immediate=True)
                self.center = target
            elif "center" in viz.result.keys():
                target = viz.result.center.cpu()
                center_changed = not torch.allclose(self.center_roate, target)
                if self._awaiting_pick_result or center_changed:
                    if self.control_modes[self.current_control_mode] == "FPS":
                        self._set_fps_turn_target_from_world(target)
                    else:
                        self._set_lookat_target(target, immediate=False)
                    self.center_roate = target
                    self._awaiting_pick_result = False

            self._animate_lookat(dt)
            self._animate_fps_turn(dt)
            io = imgui.get_io()
            editor_text_active = bool(getattr(viz, "_editor_text_active", False))
            keyboard_captured = bool(
                getattr(io, "want_text_input", False)
                or getattr(io, "want_capture_keyboard", False)
                or editor_text_active
                or getattr(viz.args, "text_change", False)
            )
            if not keyboard_captured:
                self.handle_wasd()
            elif self.control_modes[self.current_control_mode] == "Orbit":
                self.cam_pos = get_origin(
                    self.pose.yaw + np.pi / 2,
                    self.pose.pitch + np.pi / 2,
                    self.radius,
                    self.lookat_point,
                    up_vector=self.up_vector,
                )
                self.forward = normalize_vecs(self.lookat_point - self.cam_pos)
            elif self.control_modes[self.current_control_mode] == "FPS":
                self._update_fps_forward()
            
            if "mean_xyz" not in viz.result.keys() and "center" not in viz.result.keys():
                viz.args.show_image = False
            else:
                viz.args.show_image = True
        else:
            self._animate_lookat(dt)
            self._animate_fps_turn(dt)

        if show:
            label("Camera Mode", viz.label_w)
            _, self.current_control_mode = imgui.combo("##cam_modes", self.current_control_mode, self.control_modes)
            self._handle_control_mode_switch()

            if self.control_modes[self.current_control_mode] == "FPS":
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
                lookat_changed, look_at_point_tuple = imgui.input_float3(
                    "##lookat", self.lookat_point.tolist(), format="%.1f"
                )
                if lookat_changed:
                    self._set_lookat_target(torch.tensor(look_at_point_tuple), immediate=True)
                imgui.same_line()
                if imgui_utils.button("Set to xyz mean", width=viz.button_large_w) and "mean_xyz" in viz.result.keys():
                    self._set_lookat_target(viz.result.mean_xyz.cpu(), immediate=True)
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
        viz.args.click_ripples = self._build_click_ripples()
        viz.extr = self.cam_params
        
        # params for the video widget
        viz.args.lookat_point = self.lookat_point
        viz.args.up_vector = self.up_vector
    
    def handle_dragging_in_window(self, x, y, width, height):
        x_dir = -1 if self.invert_x else 1
        y_dir = -1 if self.invert_y else 1

        if 'z' not in self.viz.current_pressed_keys and 'x' not in self.viz.current_pressed_keys:
            if imgui.is_mouse_dragging(0):  # left mouse button
                new_delta = imgui.get_mouse_drag_delta(0)
                if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                    self._cancel_fps_turn()
                    delta = new_delta - self.last_drag_delta
                    self.last_drag_delta = new_delta
                    self.pose.yaw += x_dir * delta.x * self.rotate_speed * 0.1
                    self.pose.pitch += y_dir * delta.y * self.rotate_speed * 0.1
                    self.pose.pitch = np.clip(self.pose.pitch, -np.pi / 2, np.pi / 2)
            elif imgui.is_mouse_clicked(1):  # right mouse button
                if self._is_mouse_in_region(x, y, width, height):
                    mouse_pos = imgui.get_mouse_pos()
                    self.viz.args.roate_point = (mouse_pos.x - self.viz.pane_w, mouse_pos.y)
                    self._awaiting_pick_result = True
                    self._add_click_ripple(mouse_pos.x, mouse_pos.y)
            elif imgui.is_mouse_dragging(2):  # right mouse button
                new_delta = imgui.get_mouse_drag_delta(2)
                if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                    self._cancel_fps_turn()
                    delta = new_delta - self.last_drag_delta
                    self.last_drag_delta = new_delta

                    right = torch.linalg.cross(self.forward, self.up_vector)
                    right = right / torch.linalg.norm(right)
                    cam_up = torch.linalg.cross(right, self.forward)
                    cam_up = cam_up / torch.linalg.norm(cam_up)

                    x_change = x_dir * right * -delta.x * self.drag_speed
                    y_change = y_dir * cam_up * delta.y * self.drag_speed
                    self.cam_pos += x_change
                    self.cam_pos += y_change
                    if self.control_modes[self.current_control_mode] == "Orbit":
                        self.lookat_point += x_change
                        self.lookat_point += y_change
                        self.lookat_target = self.lookat_point.clone()
            else:
                self.last_drag_delta = imgui.ImVec2(0, 0)
        else:
            self.last_drag_delta = imgui.ImVec2(0, 0)
