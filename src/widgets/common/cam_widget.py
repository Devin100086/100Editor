import glfw
import time
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


class CamWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Camera")
        self.fov = 45
        self.radius = 16
        self.lookat_point = torch.tensor((0.0, 0.0, 0.0))
        self.lookat_target = self.lookat_point.clone()
        self.cam_pos = torch.tensor([0.0, 0.0, 1.0])
        self.up_vector = torch.tensor([0.0, -1.0, 0.0])
        self.forward = torch.tensor([0.0, 0.0, -1.0])

        # controls
        self.pose = EasyDict(yaw=3.2, pitch=0)
        self.invert_x = False
        self.invert_y = False
        self.move_speed = 0.02
        self.wasd_move_speed = 0.1
        self.drag_speed = 0.005
        self.rotate_speed = 0.02
        self.control_modes = ["Orbit", "FPS"]
        self.current_control_mode = 0
        self._last_control_mode = self.current_control_mode
        self.last_drag_delta = imgui.ImVec2(0, 0)
        self.lookat_transition_tau = 0.25
        self.lookat_snap_epsilon = 1e-3
        self.fps_turn_transition_tau = 0.22
        self.fps_turn_snap_epsilon = 1e-3
        self.fps_turn_target_yaw = None
        self.fps_turn_target_pitch = None
        self._last_tick_time = time.perf_counter()

        self.click_ripples = []
        self.ripple_duration = 0.8
        self.ripple_radius_start = 8.0
        self.ripple_radius_end = 42.0
        self.ripple_alpha = 0.62
        self.ripple_line_width = 3.2

        self.center = torch.tensor((0.0, 0.0, 0.0))
        self.center_roate = torch.tensor((0.0, 0.0, 0.0))
        self._awaiting_pick_result = False
        self.auto_center_initialized = False
        self.auto_center_threshold = 0.05

    @imgui_utils.scoped_by_object_id
    def __call__(self, show: bool):
        viz = self.viz
        dt = self._tick_delta_time()
        active_region = EasyDict(x=viz.pane_w, y=0, width=viz.content_width - viz.pane_w, height=viz.content_height)
        self.handle_dragging_in_window(**active_region)
        self.handle_mouse_wheel()
        if "mean_xyz" in viz.result.keys():
            target = viz.result.mean_xyz.cpu()
            should_recenter = (not self.auto_center_initialized) or (
                torch.linalg.norm(self.center - target) > self.auto_center_threshold
            )
            if should_recenter:
                self._set_lookat_target(target, immediate=True)
                self.center = target
                self.auto_center_initialized = True
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
        viz.args.cam_params = self.cam_params
        viz.args.click_ripples = self._build_click_ripples()

        # params for the video widget
        viz.args.lookat_point = self.lookat_point
        viz.args.up_vector = self.up_vector

    def handle_dragging_in_window(self, x, y, width, height):
        x_dir = -1 if self.invert_x else 1
        y_dir = -1 if self.invert_y else 1
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

    def handle_wasd(self):
        if self.control_modes[self.current_control_mode] == "FPS":
            self._update_fps_forward()
            self.sideways = torch.linalg.cross(self.forward, self.up_vector)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.cam_pos += self.forward * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.cam_pos -= self.sideways * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.cam_pos -= self.forward * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.cam_pos += self.sideways * self.wasd_move_speed
            if "q" in self.viz.current_pressed_keys:
                self.cam_pos += self.up_vector * self.wasd_move_speed
            if "e" in self.viz.current_pressed_keys:
                self.cam_pos -= self.up_vector * self.wasd_move_speed

        elif self.control_modes[self.current_control_mode] == "Orbit":
            self.cam_pos = get_origin(
                self.pose.yaw + np.pi / 2,
                self.pose.pitch + np.pi / 2,
                self.radius,
                self.lookat_point,
                up_vector=self.up_vector,
            )
            self.forward = normalize_vecs(self.lookat_point - self.cam_pos)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.pose.pitch += self.move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.pose.yaw += self.move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.pose.pitch -= self.move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.pose.yaw -= self.move_speed

    def handle_mouse_wheel(self):
        mouse_pos = imgui.get_io().mouse_pos
        if mouse_pos.x >= self.viz.pane_w:
            wheel = imgui.get_io().mouse_wheel
            if self.control_modes[self.current_control_mode] == "FPS":
                self.cam_pos += self.forward * self.move_speed * wheel
            elif self.control_modes[self.current_control_mode] == "Orbit":
                self.radius -= wheel / 10

    def _tick_delta_time(self):
        now = time.perf_counter()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        return float(np.clip(dt, 1.0 / 240.0, 0.2))

    def _set_lookat_target(self, point, immediate=False):
        if not torch.is_tensor(point):
            point = torch.tensor(point)
        point = point.detach().clone().to(torch.float32)
        self.lookat_target = point
        if immediate:
            self.lookat_point = point.clone()

    def _animate_lookat(self, dt):
        delta = self.lookat_target - self.lookat_point
        if torch.linalg.norm(delta).item() < self.lookat_snap_epsilon:
            self.lookat_point = self.lookat_target.clone()
            return
        alpha = 1.0 - np.exp(-dt / max(self.lookat_transition_tau, 1e-5))
        self.lookat_point = self.lookat_point + delta * alpha

    def _update_fps_forward(self):
        self.forward = get_forward_vector(
            lookat_position=self.cam_pos,
            horizontal_mean=self.pose.yaw + np.pi / 2,
            vertical_mean=self.pose.pitch + np.pi / 2,
            radius=0.01,
            up_vector=self.up_vector,
        )

    @staticmethod
    def _shortest_angle_delta(target, current):
        return (target - current + np.pi) % (2 * np.pi) - np.pi

    def _set_fps_turn_target_from_world(self, world_point):
        direction = world_point.to(torch.float32) - self.cam_pos.to(torch.float32)
        if torch.linalg.norm(direction).item() < 1e-8:
            return
        forward = normalize_vecs(direction)
        yaw, pitch = self._forward_to_yaw_pitch(forward)
        self.fps_turn_target_yaw = float(yaw)
        self.fps_turn_target_pitch = float(np.clip(pitch, -np.pi / 2, np.pi / 2))

    def _forward_to_yaw_pitch(self, forward):
        forward = normalize_vecs(forward.to(torch.float32))
        rot_base_to_up = self._rotation_base_to_up()
        local_forward = torch.matmul(rot_base_to_up.T, forward)
        local_forward = normalize_vecs(local_forward)

        pitch = float(np.arcsin(np.clip(float(local_forward[1]), -1.0, 1.0)))
        yaw = float(np.arctan2(-float(local_forward[0]), -float(local_forward[2])))
        return yaw, pitch

    def _rotation_base_to_up(self):
        base_vector = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
        up_vector = self.up_vector.detach().clone().to(torch.float32)
        if torch.linalg.norm(up_vector).item() < 1e-8:
            return torch.eye(3, dtype=torch.float32)

        up_vector = normalize_vecs(up_vector)
        dot_val = torch.dot(up_vector, base_vector)
        dot_val = torch.clamp(dot_val, -1.0, 1.0)
        theta = torch.arccos(dot_val)

        if theta.item() < 1e-6:
            return torch.eye(3, dtype=torch.float32)
        if abs(theta.item() - np.pi) < 1e-6:
            rot = -torch.eye(3, dtype=torch.float32)
            rot[0, 0] = 1.0
            return rot

        k = torch.cross(base_vector, up_vector, dim=0)
        k = normalize_vecs(k)
        K = torch.tensor(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
            dtype=torch.float32,
        )
        return torch.eye(3, dtype=torch.float32) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

    def _animate_fps_turn(self, dt):
        if self.control_modes[self.current_control_mode] != "FPS":
            return
        if self.fps_turn_target_yaw is None or self.fps_turn_target_pitch is None:
            return

        yaw_delta = self._shortest_angle_delta(self.fps_turn_target_yaw, self.pose.yaw)
        pitch_delta = self.fps_turn_target_pitch - self.pose.pitch
        if abs(yaw_delta) < self.fps_turn_snap_epsilon and abs(pitch_delta) < self.fps_turn_snap_epsilon:
            self.pose.yaw += yaw_delta
            self.pose.pitch = float(np.clip(self.fps_turn_target_pitch, -np.pi / 2, np.pi / 2))
            self._cancel_fps_turn()
            return

        alpha = 1.0 - np.exp(-dt / max(self.fps_turn_transition_tau, 1e-5))
        self.pose.yaw += yaw_delta * alpha
        self.pose.pitch += pitch_delta * alpha
        self.pose.pitch = float(np.clip(self.pose.pitch, -np.pi / 2, np.pi / 2))

    def _cancel_fps_turn(self):
        self.fps_turn_target_yaw = None
        self.fps_turn_target_pitch = None

    def _handle_control_mode_switch(self):
        if self.current_control_mode != self._last_control_mode:
            self._cancel_fps_turn()
            self._awaiting_pick_result = False
            self._last_control_mode = self.current_control_mode

    def _is_mouse_in_region(self, x, y, width, height):
        mouse_pos = imgui.get_mouse_pos()
        return x <= mouse_pos.x <= x + width and y <= mouse_pos.y <= y + height

    def _add_click_ripple(self, x, y):
        self.click_ripples.append(EasyDict(x=float(x), y=float(y), start=time.perf_counter()))

    def _build_click_ripples(self):
        now = time.perf_counter()
        active_ripples = []
        ripple_draw_data = []
        for ripple in self.click_ripples:
            age = now - ripple.start
            if age > self.ripple_duration:
                continue
            progress = age / self.ripple_duration
            radius = self.ripple_radius_start + (self.ripple_radius_end - self.ripple_radius_start) * progress
            alpha = (1.0 - progress) ** 1.5 * self.ripple_alpha
            ripple_draw_data.append(
                EasyDict(
                    x=ripple.x,
                    y=ripple.y,
                    radius=float(radius),
                    alpha=float(alpha),
                    line_width=self.ripple_line_width,
                )
            )
            active_ripples.append(ripple)
        self.click_ripples = active_ripples
        return ripple_draw_data
