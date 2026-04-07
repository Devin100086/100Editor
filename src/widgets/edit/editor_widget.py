import os
import subprocess
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import io
import base64
import json
from dashscope import MultiModalConversation
from imgui_bundle import imgui
from omegaconf import OmegaConf
import requests
from editor.gaussiansplatting.scene.cameras import CustomCam             
from editor.hundrededitor_gui.add import add_sketch                   
from utils.gui_utils import imgui_utils
from imgui_bundle import implot
from utils.command_utils import *
from utils.gui_utils.easy_imgui import label
from google import genai
from google.genai import types
from PIL import Image
from scipy.spatial.transform import Rotation as R
from io import BytesIO

from torchvision.transforms.functional import to_tensor
import torch
from utils.dict_utils import EasyDict
import sys

from widgets.widget import Widget
import glfw
from PIL import Image
import numpy as np
from OpenGL.GL import *
from editor.hundrededitor_gui.drag import animation_initialize, animation_reset
from utils.path_utils import find_repo_root, resolve_runtime_subdir

class Config:
    def __init__(self, ply_file_path, data_source, mask_prompt, edit_train_steps, left_up, right_down, zoom):
        self.gs_source = ply_file_path
        self.colmap_dir = data_source
        self.text_prompt = mask_prompt
        self.edit_train_steps = edit_train_steps
        self.left_up = [-1,-1] if left_up == None else (int(left_up[0]), int(left_up[1]))
        self.right_down = [-1,-1] if right_down == None else (int(right_down[0]), int(right_down[1]))
        self.zoom = -1 if zoom == None else zoom

 
class EditorWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Editor")
        # Option
        self.select_option = 0
        self.lambda_l1 = 10
        self.lambda_p = 10
        self.lambda_anchor_color = 0
        self.lambda_anchor_geo = 50
        self.lambda_anchor_scale = 50
        self.lambda_anchor_opacity = 50
        self.edit_until_step = 1000
        self.per_editing_step = 10
        self.edit_begin_step = 0
        self.edit_cam_num = 300
        self.edit_train_steps = 30000
        self.densify_until_step = 1000
        self.cameara_update_step = 500
        self.densification_interval = 100

        self.gs_lr_scaler = 1
        self.gs_lr_end_scaler = 1
        self.color_lr_scaler = 2
        self.opacity_lr_scaler = 1.0
        self.scaling_lr_scaler = 1.0
        self.rotation_lr_scaler = 1.0
        
        # text-edit
        self.guidance_type = ["InstructPix2Pix","ControlNet-Depth","BrushNet"]
        self.guidance_item = 0
        self.text_prompt = "Make it winter"
        self.origin_prompt = "a photo of a bear statue in the forest"
        self.text_sam_option = 0
        self.text_point_option = 0
        self.text_seg_prompt = "face"
        self.text_videoEditing = False
        self.edit_use_original_resolution = False
        self.text_sam_positive_points = []
        self.text_sam_negative_points = []
        self.edit_output_dir = ""
        self.hard_segmentation = True
        self.mask_thres = 0.5
        self.early_stopping = False
        self.cps_patience = 4
        self.cps_batch_count = 3
        self.clip_origin_prompt = "a photo of an outdoor garden"
        self.clip_target_prompt = "a photo of an outdoor garden in winter"

        # adding
        self.mask_prompt = "add a red hat"
        self.start_rec_pos = None
        self.end_rec_pos = None
        self.rec_start = None 
        self.rec_end = None
        self.drawing_rect = False
        self.traincoarseadd = None
        self.depth = 1
        self.editing_option = -1
        self.points = []
        self.current_color = [1.0, 1.0, 1.0, 1.0]
        self.line_width = 35.0
        self.sketch_prompt = "A man wears a red hat on head"
        self.generate_3D_prompt = "a red hat"
        self.is_drawing = False
        self.painting = False
        self.seed = 1
        self.single_image = None
        self.edit_single = False
        self.segmentation_prompt = "hat"
        self.concat = False

        # deleting
        self.delete_prompt = "remove the vase"
        self.seg_delete_prompt = "vase"
        self.inpaint_scale = 1.0
        self.mask_dilate = 15
        self.video_inpainting = False
        self.delete_sam_option = 0
        self.delete_point_option = 0
        self.delete_sam_positive_points = []
        self.delete_sam_negative_points = []
        self.delete_use_original_resolution = False
        self.delete_output_dir = ""

        # Drag
        self.animation = False
        self.last_drag_delta = imgui.ImVec2(0, 0)
        self.showing_overlay = True
        self.drag_sam_option = 0
        self.drag_point_option = 0
        self.drag_seg_prompt = "face"
        self.drag_sam_positive_points = []
        self.drag_sam_negative_points = []

        self.edit_trainer = None
        self.draw_image = False
        self.edit3D = False
        self.text_change = False
        self.plots = EasyDict(
            loss=dict(values=[], dtype=float),
            num_gaussians=dict(values=[], dtype=int),
        )
        self.iterations = []
        self.last_iteration = None
        self.stop_from_renderer = False

        self.runtime_edit_dir = resolve_runtime_subdir(__file__, "cache", "edit", create=True)
        self.runtime_add_dir = resolve_runtime_subdir(__file__, "cache", "add", create=True)
        self.runtime_delete_dir = resolve_runtime_subdir(__file__, "cache", "delete", create=True)
        self.runtime_drag_dir = resolve_runtime_subdir(__file__, "cache", "drag", create=True)
        self.runtime_edit_experiments_dir = resolve_runtime_subdir(
            __file__, "experiments", "edit", create=True
        )
        self.edit_output_dir = resolve_runtime_subdir(
            __file__, "experiments", "edit", "semantic", create=True
        ).as_posix()
        self.concat_save_path = (
            resolve_runtime_subdir(__file__, "experiments", "edit", "add", create=True)
            / "concat.ply"
        ).as_posix()
        self.show_stage2_add_tip = False
        self.delete_output_dir = resolve_runtime_subdir(
            __file__, "experiments", "edit", "delete", create=True
        ).as_posix()
        self._open_option_sections_once = True
        self._remove_executor = ThreadPoolExecutor(max_workers=1)
        self._remove_future = None
        self.remove_pending = False
        self._apply_option_preset(self.select_option)

    def _apply_option_preset(self, option: int):
        self.select_option = option
        if option == 0:
            self.edit_cam_num = 60
            self.edit_train_steps = 1500
            self.edit_until_step = 2000
            self.per_editing_step = 10
            self.densification_interval = 50
            self.densify_until_step = 1500
        elif option == 1:
            self.edit_cam_num = 20
            self.per_editing_step = 10000
            self.edit_train_steps = 1500
            self.edit_until_step = 4000
            self.densification_interval = 100
            self.densify_until_step = 4000
        elif option == 2:
            self.edit_cam_num = 48
            self.per_editing_step = 10
            self.densification_interval = 50
            self.edit_train_steps = 1500
            self.edit_until_step = 1000
            self.densify_until_step = 1300
        elif option == 3:
            self.edit_cam_num = 12
            self.per_editing_step = 10000
            self.edit_train_steps = 1500
            self.edit_until_step = 4000
            self.densification_interval = 100
            self.densify_until_step = 4000
        elif option == 4:
            self.edit_cam_num = 48
            self.per_editing_step = 10
            self.edit_train_steps = 2000
            self.edit_until_step = 1000
            self.densification_interval = 50
            self.densify_until_step = 4000

    @staticmethod
    def _open_tree_node_once_if_needed(flag: bool):
        if not flag:
            return
        if hasattr(imgui, "set_next_item_open"):
            cond = 0
            if hasattr(imgui, "Cond_") and hasattr(imgui.Cond_, "once"):
                cond = imgui.Cond_.once
            try:
                imgui.set_next_item_open(True, cond)
            except TypeError:
                imgui.set_next_item_open(True)

    @staticmethod
    def _clear_directory(path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)

    def _reset_training_history(self):
        self.iterations.clear()
        self.last_iteration = None
        self.stop_from_renderer = False
        for plot in self.plots.values():
            if "values" in plot:
                plot["values"].clear()

    def _clear_add_canvas_state(self):
        self.points = []
        self.is_drawing = False
        self.drawing_rect = False
        self.start_rec_pos = None
        self.end_rec_pos = None
        self.rec_start = None
        self.rec_end = None

    def _get_render_image_rect(self):
        pane_left = float(self.viz.pane_w)
        viewport_w = float(max(1, self.viz.content_width - self.viz.pane_w))
        viewport_h = float(max(1, self.viz.content_height))
        x0 = pane_left
        y0 = 0.0
        x1 = pane_left + viewport_w
        y1 = viewport_h

        if "image" not in self.viz.result:
            return x0, y0, x1, y1
        image = self.viz.result.image
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            return x0, y0, x1, y1

        img_h = float(max(1, int(image.shape[0])))
        img_w = float(max(1, int(image.shape[1])))
        zoom = min(viewport_w / img_w, viewport_h / img_h)
        draw_w = img_w * zoom
        draw_h = img_h * zoom
        draw_x0 = pane_left + (viewport_w - draw_w) * 0.5
        draw_y0 = (viewport_h - draw_h) * 0.5
        return draw_x0, draw_y0, draw_x0 + draw_w, draw_y0 + draw_h

    def _current_mouse_uv_in_render_image(self):
        if bool(getattr(self.viz, "_suppress_viewport_mouse", False)):
            return None
        mouse = imgui.get_mouse_pos()
        x0, y0, x1, y1 = self._get_render_image_rect()
        if mouse.x < x0 or mouse.x > x1 or mouse.y < y0 or mouse.y > y1:
            return None
        w = max(1e-6, x1 - x0)
        h = max(1e-6, y1 - y0)
        u = float(np.clip((mouse.x - x0) / w, 0.0, 1.0))
        v = float(np.clip((mouse.y - y0) / h, 0.0, 1.0))
        return [u, v]

    @staticmethod
    def _supports_ansi() -> bool:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @staticmethod
    def _ansi(text: str, code: str) -> str:
        if not EditorWidget._supports_ansi():
            return text
        return f"\033[{code}m{text}\033[0m"

    def _print_single_edit_start(self, prompt: str, seed: int, image_path: str, mask_path: str):
        title = self._ansi("[Single Image Edit] Start", "1;36")
        print(f"\n{title}")
        print(self._ansi("  Prompt    :", "36"), prompt)
        print(self._ansi("  Seed      :", "36"), seed)
        print(self._ansi("  Input     :", "36"), image_path)
        print(self._ansi("  Mask      :", "36"), mask_path)

    def _print_single_edit_done(self, elapsed_sec: float):
        done = self._ansi("[Single Image Edit] Done", "1;32")
        print(f"{done}  {self._ansi(f'Elapsed: {elapsed_sec:.2f}s', '32')}\n")

    def _print_single_edit_fail(self, elapsed_sec: float, error: Exception):
        failed = self._ansi("[Single Image Edit] Failed", "1;31")
        print(f"{failed}  {self._ansi(f'Elapsed: {elapsed_sec:.2f}s', '31')}")
        print(self._ansi("  Error     :", "31"), str(error))

    def _print_remove_start(self, prompt: str, image_size: tuple[int, int], model: str, size: str):
        title = self._ansi("[Single Image Remove] Start", "1;36")
        print(f"\n{title}")
        print(self._ansi("  Prompt    :", "36"), prompt)
        print(self._ansi("  InputSize :", "36"), f"{image_size[0]}x{image_size[1]}")
        print(self._ansi("  Model     :", "36"), model)
        print(self._ansi("  OutputSize:", "36"), size)

    def _print_remove_stage(self, message: str):
        print(self._ansi(f"[Single Image Remove] {message}", "36"))

    def _print_remove_done(self, elapsed_sec: float, save_path: str):
        done = self._ansi("[Single Image Remove] Done", "1;32")
        print(f"{done}  {self._ansi(f'Elapsed: {elapsed_sec:.2f}s', '32')}")
        print(self._ansi("  Saved To  :", "32"), save_path)
        print("")

    def _print_remove_fail(self, elapsed_sec: float, error_text: str):
        failed = self._ansi("[Single Image Remove] Failed", "1;31")
        print(f"{failed}  {self._ansi(f'Elapsed: {elapsed_sec:.2f}s', '31')}")
        print(self._ansi("  Reason    :", "31"), error_text)
        print("")

    def _consume_training_stats(self, viz):
        if "training_stats" not in viz.result.keys():
            return None
        stats = viz.result["training_stats"]
        current_iteration = int(stats["iteration"])
        if self.last_iteration is not None and current_iteration < self.last_iteration:
            self._reset_training_history()
        if self.last_iteration is None or current_iteration != self.last_iteration:
            self.iterations.append(current_iteration)
            self.plots.loss["values"].append(stats["loss"])
            self.plots.num_gaussians["values"].append(stats["num_gaussians"])
        self.last_iteration = current_iteration
        self.stop_from_renderer = bool(stats.get("paused", False))
        return stats

    def _draw_training_plots(self, viz):
        for plot_name, plot_values in self.plots.items():
            plot_size = imgui.ImVec2(viz.pane_w - 150, 200)
            implot.set_next_axes_to_fit()
            if implot.begin_plot(plot_name, plot_size):
                implot.plot_line(
                    plot_name,
                    ys=np.array(plot_values["values"], dtype=plot_values["dtype"]),
                    xs=np.array(self.iterations, dtype=plot_values["dtype"]),
                )
                implot.end_plot()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        edit_image = False
        single_training_step = False
        save_concat_ply_path = None
        if show:
            if imgui.begin_tab_bar("EditBar"):
                if imgui.begin_tab_item("Option")[0]:

                    if imgui.radio_button("Edit", self.select_option == 0):
                        self._apply_option_preset(0)
                    imgui.same_line()
                    if imgui.radio_button("Batch Edit", self.select_option == 1):
                        self._apply_option_preset(1)
                    imgui.same_line()
                    if imgui.radio_button("Add", self.select_option == 2):
                        self._apply_option_preset(2)
                    imgui.same_line()
                    if imgui.radio_button("Batch Add", self.select_option == 3):
                        self._apply_option_preset(3)
                    imgui.same_line()
                    if imgui.radio_button("Delete", self.select_option == 4):
                        self._apply_option_preset(4)

                    self._open_tree_node_once_if_needed(self._open_option_sections_once)
                    if imgui.tree_node("Editing Training"):
                        label("Camera Num", viz.label_w_large)
                        _, self.edit_cam_num = imgui.slider_int("##Camera Num", self.edit_cam_num, 12, 200, format="%d")
                        label("Total Step", viz.label_w_large)
                        _, self.edit_train_steps = imgui.slider_int("##Total Step", self.edit_train_steps, 0, 5000, format="%d")
                        label("Camera Update Step", viz.label_w_large)
                        _, self.cameara_update_step = imgui.slider_int("##Camera Update Step", self.cameara_update_step, 0, 5000, format="%d")
                        label("Edit Until Step", viz.label_w_large)
                        _, self.edit_until_step = imgui.slider_int("##Edit Until Step", self.edit_until_step, 0, 5000, format="%d")
                        label("Edit Begining", viz.label_w_large)
                        _, self.edit_begin_step = imgui.slider_int("##Edit Begining", self.edit_begin_step, 0, 5000, format="%d")
                        label("Edit Interval", viz.label_w_large)
                        _, self.per_editing_step = imgui.slider_int("##Edit Interval", self.per_editing_step, 4, 12000, format="%d")
                        label("Densification Interval", viz.label_w_large)
                        _, self.densification_interval = imgui.slider_int("##Densification Interval", self.densification_interval, 1, 200, format="%d")
                        label("Densification Until Step", viz.label_w_large)
                        _, self.densify_until_step = imgui.slider_int("##Densification Until Step", self.densify_until_step, 0, 5000, format="%d")
                        imgui.tree_pop()
                    
                    self._open_tree_node_once_if_needed(self._open_option_sections_once)
                    if imgui.tree_node("Loss Weight"):
                        label("Lambda L1", viz.label_w_large)
                        _, self.lambda_l1 = imgui.slider_int("##Lambda L1", self.lambda_l1, 0, 100, format="%d")
                        label("Lambda Perceptual", viz.label_w_large)
                        _, self.lambda_p = imgui.slider_int("##Lambda Perceptual", self.lambda_p, 0, 100, format="%d")
                        label("Lambda Anchor Color", viz.label_w_large)
                        _, self.lambda_anchor_color = imgui.slider_int("##Lambda Anchor Color", self.lambda_anchor_color, 0, 500, format="%d")
                        label("Lambda Anchor Geo", viz.label_w_large)
                        _, self.lambda_anchor_geo = imgui.slider_int("##Lambda Anchor Geo", self.lambda_anchor_geo, 0, 500, format="%d")
                        label("Lambda Anchor Scale", viz.label_w_large)
                        _, self.lambda_anchor_scale = imgui.slider_int("##Lambda Anchor Scale", self.lambda_anchor_scale, 0, 500, format="%d")
                        label("Lambda Anchor Opacity", viz.label_w_large)
                        _, self.lambda_anchor_opacity = imgui.slider_int("##Lambda Anchor Opacity", self.lambda_anchor_opacity, 0, 500, format="%d")
                        imgui.tree_pop()
                    
                    self._open_tree_node_once_if_needed(self._open_option_sections_once)
                    if imgui.tree_node("LR weight"):
                        label("GS LR Scaler", viz.label_w_large)
                        _, self.gs_lr_scaler = imgui.slider_float("##GS LR Scaler", self.gs_lr_scaler, 0.0, 5.0, format="%.1f")
                        label("GS LR End Scaler", viz.label_w_large)
                        _, self.gs_lr_end_scaler = imgui.slider_float("##GS LR End Scaler", self.gs_lr_end_scaler, 0.0, 5.0, format="%.1f")
                        label("Color LR Scaler", viz.label_w_large)
                        _, self.color_lr_scaler = imgui.slider_float("##Color LR Scaler", self.color_lr_scaler, 0.0, 5.0, format="%.1f")
                        label("Opacity LR Scaler", viz.label_w_large)
                        _, self.opacity_lr_scaler = imgui.slider_float("##Opacity LR Scaler", self.opacity_lr_scaler, 0.0, 5.0, format="%.1f")
                        label("Scaling LR Scaler", viz.label_w_large)
                        _, self.scaling_lr_scaler = imgui.slider_float("##Scaling LR Scaler", self.scaling_lr_scaler, 0.0, 5.0, format="%.1f")
                        label("Rotation LR Scaler", viz.label_w_large)
                        _, self.rotation_lr_scaler = imgui.slider_float("##Rotation LR Scaler", self.rotation_lr_scaler, 0.0, 5.0, format="%.1f")
                        imgui.tree_pop()
                    self._open_option_sections_once = False
                   
                    imgui.end_tab_item()

                if imgui.begin_tab_item("semantic")[0]:
                    if self.delete_sam_positive_points != [] or self.delete_sam_negative_points != []:
                        self.delete_sam_positive_points = []
                        self.delete_sam_negative_points = []
                        
                    imgui.separator_text("parameters")
                    label("guidance type", viz.label_w)
                    _, self.guidance_item = imgui.combo(
                        "##guidance type",                
                        self.guidance_item,           
                        self.guidance_type                   
                    )
                    if self.guidance_item == 1:
                        label("origin prompt", viz.label_w)
                        _, self.origin_prompt = imgui.input_text("##Origin Prompt", self.origin_prompt, 256)
                    label("prompt", viz.label_w)
                    _, self.text_prompt = imgui.input_text("##Prompt", self.text_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    label("mask threshold", viz.label_w)
                    _, self.mask_thres = imgui.slider_float("##Mask Threshold", self.mask_thres, 0.0, 1.0, format="%.2f")
                    label("Batch", viz.label_w)
                    _, self.text_videoEditing = imgui.checkbox("##Batch", self.text_videoEditing)
                    label("Original Resolution", viz.label_w)
                    _, self.edit_use_original_resolution = imgui.checkbox("##Use Original Resolution", self.edit_use_original_resolution)
                    label("hard segmentation", viz.label_w)
                    _, self.hard_segmentation = imgui.checkbox("##Hard Segmentation", self.hard_segmentation)

                    imgui.separator_text("CPS Option")
                    label("CPS", viz.label_w)
                    _, self.early_stopping = imgui.checkbox("##CPS", self.early_stopping)
                    if self.early_stopping:
                        label("patience", viz.label_w)
                        _, self.cps_patience = imgui.slider_int(
                            "##CPS Patience", self.cps_patience, 1, 20, format="%d"
                        )
                        label("batch count", viz.label_w)
                        _, self.cps_batch_count = imgui.slider_int(
                            "##CPS Batch Count", self.cps_batch_count, 1, 20, format="%d"
                        )
                        label("clip origin prompt", viz.label_w)
                        _, self.clip_origin_prompt = imgui.input_text("##CLIP Origin Prompt", self.clip_origin_prompt, 256)
                        label("clip target prompt", viz.label_w)
                        _, self.clip_target_prompt = imgui.input_text("##CLIP Target Prompt", self.clip_target_prompt, 256)
                    imgui.separator_text("SAM Option")
                    label("Sam Type", viz.label_w)
                    _, self.text_sam_option = imgui.combo(
                        "##SAM Type", 
                        self.text_sam_option, 
                        ["No Sam", "Lang-sam", "SAM2(image)","SAM2(video)"]  
                    )

                    if self.text_sam_option == 1:
                        label("seg prompt", viz.label_w)
                        _, self.text_seg_prompt = imgui.input_text("##seg prompt", self.text_seg_prompt, 256)
                        self.text_change = True if imgui.is_item_active() else False

                    if self.text_sam_option == 2 or self.text_sam_option == 3:
                        if imgui.radio_button("No Points", self.text_point_option == 0):
                            self.text_point_option = 0
                        imgui.same_line()
                        if imgui.radio_button("Positive Point", self.text_point_option == 1):
                            self.text_point_option = 1
                        imgui.same_line()
                        if imgui.radio_button("Negative Point", self.text_point_option == 2):
                            self.text_point_option = 2
                        imgui.same_line()
                        if self.text_point_option == 1 or self.text_point_option == 2:
                            if imgui.is_mouse_clicked(0):
                                uv = self._current_mouse_uv_in_render_image()
                                if uv is None:
                                    pass
                                elif self.text_point_option == 1:
                                    self.text_sam_positive_points.append(uv)
                                elif self.text_point_option == 2:
                                    self.text_sam_negative_points.append(uv)
                        if imgui_utils.button("clean SAM", width=viz.button_w):
                            self.text_sam_positive_points = []
                            self.text_sam_negative_points = []

                    imgui.text(f"Save Path: {self.edit_output_dir}")

                    if not self.edit3D:
                        if imgui_utils.button("Edit", width=viz.button_w):
                            self._reset_training_history()
                            self.edit3D = True
                            edit_cache_dir = self.runtime_edit_dir
                            edit_render_dir = edit_cache_dir / "render"
                            os.makedirs(edit_cache_dir, exist_ok=True)
                            np.save(
                                (edit_cache_dir / "sam2_positive_points.npy").as_posix(),
                                np.array(self.text_sam_positive_points),
                            )
                            np.save(
                                (edit_cache_dir / "sam2_negative_points.npy").as_posix(),
                                np.array(self.text_sam_negative_points),
                            )
                            origin = Image.fromarray(viz.result.image).convert("RGB")
                            self._clear_directory(edit_render_dir)
                            R = viz.extr.inverse()[:3, :3].T.numpy()
                            T = viz.extr.inverse()[:3, 3].numpy()
                            fov_rad = viz.fov / 360 * 2 * np.pi
                            cam = CustomCam(origin.size[0]//2, origin.size[1]//2, fov_rad, fov_rad, R, T, viz.extr.cuda())
                            with open((edit_cache_dir / "camera.pkl").as_posix(), 'wb') as f:
                                pickle.dump(cam, f)  
                            self.edit_trainer = training_text_editing_command(gs_source=viz.args.ply_file_paths[0],colmap_dir=viz.args.data_source,
                                                                             edit_cam_num=self.edit_cam_num,guidance_type=self.guidance_type[self.guidance_item],
                                                                             text_prompt=self.text_prompt, origin_prompt = self.origin_prompt, edit_train_steps=self.edit_train_steps,
                                                                             per_editing_step=self.per_editing_step,edit_begin_step=self.edit_begin_step,
                                                                             edit_until_step=self.edit_until_step,lambda_l1=self.lambda_l1,
                                                                             lambda_p=self.lambda_p,lambda_anchor_color=self.lambda_anchor_color,
                                                                             lambda_anchor_geo=self.lambda_anchor_geo,lambda_anchor_scale=self.lambda_anchor_scale,
                                                                             lambda_anchor_opacity=self.lambda_anchor_opacity,sam_option=self.text_sam_option,
                                                                             seg_prompt=self.text_seg_prompt,text_videoEditing=self.text_videoEditing,
                                                                             gs_lr_scaler = self.gs_lr_scaler, gs_lr_end_scaler = self.gs_lr_end_scaler, 
                                                                             color_lr_scaler = self.color_lr_scaler,  opacity_lr_scaler = self.opacity_lr_scaler, 
                                                                             scaling_lr_scaler = self.scaling_lr_scaler,  rotation_lr_scaler = self.rotation_lr_scaler, 
                                                                             positive_sam_points = (edit_cache_dir / "sam2_positive_points.npy").as_posix(), negative_sam_points = (edit_cache_dir / "sam2_negative_points.npy").as_posix(),
                                                                            camera = (edit_cache_dir / "camera.pkl").as_posix(), use_original_resolution = self.edit_use_original_resolution, output_dir = self.edit_output_dir,
                                                                            hard_segmentation = self.hard_segmentation, mask_thres=self.mask_thres, earlystop=self.early_stopping,
                                                                             cps_patience_counter=0, cps_patience=self.cps_patience, cps_batch_count=self.cps_batch_count,
                                                                             clip_origin_prompt=self.clip_origin_prompt, clip_target_prompt=self.clip_target_prompt
                                                                            )
                    else:
                        stats = self._consume_training_stats(viz)
                        if stats is not None and bool(stats.get("paused", False)):
                            imgui.text("Reached final iteration. Click Pause Training to save.")
                            if imgui_utils.button("Pause Training", width=viz.button_w * 1.8):
                                single_training_step = True
                        elif imgui_utils.button("Stop", width=viz.button_w):
                            self.edit3D = False
                            self.edit_trainer.terminate()
                            self.edit_trainer.wait()

                    if self._consume_training_stats(viz) is not None:
                        self._draw_training_plots(viz)

                    imgui.end_tab_item()

                if imgui.begin_tab_item("add")[0]:
                    self.sam_points = []
                    label("option", viz.label_w)
                    previous_editing_option = self.editing_option
                    if imgui.radio_button("No", self.editing_option == -1):
                        self.editing_option = -1
                    imgui.same_line()
                    if imgui.radio_button("Sketch", self.editing_option == 0):
                        self.editing_option = 0
                    imgui.same_line()
                    if imgui.radio_button("Mask", self.editing_option == 1):
                        self.editing_option = 1
                    if self.editing_option != previous_editing_option:
                        self._clear_add_canvas_state()

                    if self.editing_option == 0:
                        _, self.line_width = imgui.slider_float(
                            "width", self.line_width, 1.0, 40.0, format="%.1f"
                        )
                        _, self.current_color = imgui.color_edit4("color choice", self.current_color)
                        if imgui.button("clear"):
                            self.points = []
                        self.handle_mouse_input()
                    elif self.editing_option == 1:
                        self.draw_mask()
                        if imgui.button("clear"):
                            self._clear_add_canvas_state()
                
                    imgui.separator_text("edit")

                    self.draw_image = True if self.editing_option >= 0 else False
                    if self.editing_option < 0 and self.judge_move():  
                        self._clear_add_canvas_state()
                    edit_image = True
                    label("prompt", viz.label_w)
                    _, self.sketch_prompt = imgui.input_text("##Prompt", self.sketch_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False

                    label("Seed", viz.label_w)
                    _, self.seed = imgui.slider_int("##Seed", self.seed, 0, 10, format="%d")
                    if imgui_utils.button("Edit", width=viz.button_w):
                        self.edit3D = False
                        self.edit_single = True
                        mask = self.get_mask(viz.origin_image, viz.edit_image)
                        cache_dir = self.runtime_add_dir
                        os.makedirs(cache_dir, exist_ok=True)
                        mask = self.get_mask(viz.origin_image, viz.edit_image)
                        mask.save((cache_dir / "mask.png").as_posix())
                        viz.result.image = (viz.result.image * 255).astype(np.uint8) if np.all(viz.result.image <= 1.0) else viz.result.image
                        origin = Image.fromarray(viz.result.image).convert("RGB")
                        origin.save((cache_dir / "origin.png").as_posix())

                        self.single_image = self.edit_single_image(
                                                  prompts = self.sketch_prompt,
                                                  seed = self.seed,
                                                  image_path = (cache_dir / "origin.png").as_posix(),
                                                  mask_path = (cache_dir / "mask.png").as_posix())
                        self.points = []
                        self.rec_start, self.rec_end = None, None

                    imgui.separator_text("add")

                    label("Segmentation", viz.label_w)
                    changed, self.segmentation_prompt = imgui.input_text("##Segmentation", self.segmentation_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    if imgui_utils.button("Mesh", width=viz.button_w):
                        self.show_stage2_add_tip = False
                        cache_dir = self.runtime_add_dir
                        os.makedirs(cache_dir, exist_ok=True)
                        viz.result.image = (viz.result.image * 255).astype(np.uint8) if np.all(viz.result.image <= 1.0) else viz.result.image
                        origin = Image.fromarray(viz.result.image).convert("RGB")
                        origin.save((cache_dir / "origin.png").as_posix())
                        cfg = Config(
                            ply_file_path=viz.args.ply_file_paths[0],
                            data_source=viz.args.data_source,
                            mask_prompt=self.mask_prompt,
                            edit_train_steps=self.edit_train_steps,
                            left_up=None,
                            right_down=None,
                            zoom=None,
                        )
                        add_sketch(origin, self.segmentation_prompt, cache_dir=cache_dir.as_posix())
                    
                    imgui.separator_text("concat")
                    label("Depth", viz.label_w)
                    _, self.depth = imgui.slider_float("##Depth", self.depth, 0, 10, format="%.2f")
                    concat_source_ply = self.runtime_add_dir / "inpaint_gs.ply"
                    concat_source_ply_path = (
                        concat_source_ply.as_posix() if concat_source_ply.is_file() else ""
                    )
                    imgui.text(f"Path: {concat_source_ply_path}")

                    if concat_source_ply.is_file():
                        if imgui_utils.button("Show", width=viz.button_w): 
                            self.edit_single = False
                            self.edit3D = False 
                            self.concat = True
                            viz.result.image = (viz.result.image * 255).astype(np.uint8) if np.all(viz.result.image <= 1.0) else viz.result.image
                            origin = Image.fromarray(viz.result.image).convert("RGB")
                            R = viz.extr.inverse()[:3, :3].T.numpy()
                            T = viz.extr.inverse()[:3, 3].numpy()
                            fov_rad = viz.fov / 360 * 2 * np.pi
                            cam = CustomCam(origin.size[0], origin.size[1], fov_rad, fov_rad, R, T, viz.extr.cuda())
                            with open((self.runtime_add_dir / "camera.pkl").as_posix(), 'wb') as f:
                                pickle.dump(cam, f)    

                        else:
                            self.concat = False    
                        imgui.same_line()
                        if imgui_utils.button("Stop", width=viz.button_w):
                            viz.args.stop_concat = True
                        else:
                            viz.args.stop_concat = False
                        imgui.same_line()
                        if imgui_utils.button("Save", width=viz.button_w):
                            save_concat_ply_path = self.concat_save_path
                            self.show_stage2_add_tip = True
                        if self.show_stage2_add_tip:
                            imgui.text(f"Save Path: {self.concat_save_path}")
                            imgui.text(
                                'You can now perform fine-grained additive edits in Stage 2 within the "semantic" module.'
                            )
                    else:
                        self.concat = False
                        viz.args.stop_concat = False
                        imgui.begin_disabled()
                        if imgui_utils.button("Show", width=viz.button_w):
                            pass
                        imgui.same_line()
                        if imgui_utils.button("Stop", width=viz.button_w):
                            pass
                        imgui.same_line()
                        if imgui_utils.button("Save", width=viz.button_w):
                            pass
                        imgui.end_disabled()

                    imgui.end_tab_item()

                if imgui.begin_tab_item("delete")[0]:
                    if self.remove_pending and self._remove_future is not None and self._remove_future.done():
                        try:
                            remove_result = self._remove_future.result()
                            if remove_result is not None:
                                self.single_image = remove_result
                                self.edit_single = True
                        except Exception as exc:
                            print(f"[Remove] Failed: {exc}")
                        finally:
                            self.remove_pending = False
                            self._remove_future = None

                    if self.text_sam_positive_points != [] or self.text_sam_negative_points != []:
                        self.text_sam_positive_points = []
                        self.text_sam_negative_points = []

                    imgui.separator_text("remove")
                    label("prompt", viz.label_w)
                    _, self.delete_prompt = imgui.input_text("##Remove Prompt", self.delete_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    if self.remove_pending:
                        imgui.begin_disabled()
                        if imgui_utils.button("remove", width=viz.button_w):
                            pass
                        imgui.end_disabled()
                    elif imgui_utils.button("remove", width=viz.button_w):
                        self.edit_single = True
                        image_for_remove = np.array(viz.result.image, copy=True)
                        self.remove_pending = True
                        self._remove_future = self._remove_executor.submit(
                            self.remove_single_image,
                            prompts=self.delete_prompt,
                            image=image_for_remove,
                        )
                    imgui.same_line()
                    if imgui_utils.button("stop", width=viz.button_w):
                        self.edit_single = False
                    if self.remove_pending:
                        imgui.same_line()
                        imgui.text("Please wait approximately 40-50 seconds...")

                    imgui.separator_text("parameters")
                    label("Original Resolution", viz.label_w)
                    _, self.delete_use_original_resolution = imgui.checkbox("##Use Original Resolution", self.delete_use_original_resolution)
                    label("mask dilate", viz.label_w)
                    _, self.mask_dilate = imgui.slider_int("##Mask Dilate", self.mask_dilate, 0, 30, format="%d")
                    
                    imgui.separator_text("SAM option")
                    label("Sam Type", viz.label_w)
                    _, self.delete_sam_option = imgui.combo(
                        "##SAM Type", 
                        self.delete_sam_option, 
                        ["Lang-sam", "SAM2(image)","SAM2(video)"] 
                    )

                    if self.delete_sam_option == 0:
                        label("seg prompt", viz.label_w)
                        _, self.seg_delete_prompt = imgui.input_text("##seg prompt", self.seg_delete_prompt, 256)
                    elif self.delete_sam_option == 1 or self.delete_sam_option == 2:
                        if imgui.radio_button("No Points", self.delete_point_option == 0):
                            self.delete_point_option = 0
                        imgui.same_line()
                        if imgui.radio_button("Positive Point", self.delete_point_option == 1):
                            self.delete_point_option = 1
                        imgui.same_line()
                        if imgui.radio_button("Negative Point", self.delete_point_option == 2):
                            self.delete_point_option = 2
                        imgui.same_line()

                        if self.delete_point_option == 1 or self.delete_point_option == 2:
                            if imgui.is_mouse_clicked(0):
                                uv = self._current_mouse_uv_in_render_image()
                                if uv is None:
                                    pass
                                elif self.delete_point_option == 1:
                                    self.delete_sam_positive_points.append(uv)
                                elif self.delete_point_option == 2:
                                    self.delete_sam_negative_points.append(uv)

                        if imgui_utils.button("clean SAM", width=viz.button_w):
                            self.delete_sam_positive_points = []
                            self.delete_sam_negative_points = []

                    imgui.text(f"Save Path: {self.delete_output_dir}")

                    if not self.edit3D:
                        if imgui_utils.button("Delete", width=viz.button_w):
                            self._reset_training_history()
                            self.edit3D = True
                            delete_cache_dir = self.runtime_delete_dir
                            np.save(
                                (delete_cache_dir / "sam2_positive_points.npy").as_posix(),
                                np.array(self.delete_sam_positive_points),
                            )
                            np.save(
                                (delete_cache_dir / "sam2_negative_points.npy").as_posix(),
                                np.array(self.delete_sam_negative_points),
                            )
                            origin = Image.fromarray(viz.result.image).convert("RGB")
                            R = viz.extr.inverse()[:3, :3].T.numpy()
                            T = viz.extr.inverse()[:3, 3].numpy()
                            fov_rad = viz.fov / 360 * 2 * np.pi
                            cam = CustomCam(origin.size[0]//2, origin.size[1]//2, fov_rad, fov_rad, R, T, viz.extr.cuda())
                            with open((delete_cache_dir / "camera.pkl").as_posix(), 'wb') as f:
                                pickle.dump(cam, f)  
                            self._clear_directory(delete_cache_dir / "render")
                            self.edit_trainer = training_delete_command(
                                gs_source=viz.args.ply_file_paths[0],colmap_dir=viz.args.data_source,
                                inpaint_scale=self.inpaint_scale,mask_dilate=self.mask_dilate,
                                edit_cam_num=self.edit_cam_num,delete_prompt=self.delete_prompt,
                                edit_train_steps=self.edit_train_steps,
                                per_editing_step=self.per_editing_step,edit_begin_step=self.edit_begin_step,
                                edit_until_step=self.edit_until_step,lambda_l1=self.lambda_l1,
                                lambda_p=self.lambda_p,lambda_anchor_color=self.lambda_anchor_color,
                                lambda_anchor_geo=self.lambda_anchor_geo,lambda_anchor_scale=self.lambda_anchor_scale,
                                lambda_anchor_opacity=self.lambda_anchor_opacity, video = self.video_inpainting,
                                gs_lr_scaler = self.gs_lr_scaler, gs_lr_end_scaler = self.gs_lr_end_scaler, 
                                color_lr_scaler = self.color_lr_scaler,  opacity_lr_scaler = self.opacity_lr_scaler, 
                                scaling_lr_scaler = self.scaling_lr_scaler,  rotation_lr_scaler = self.rotation_lr_scaler,
                                sam_type = self.delete_sam_option, positive_sam_points = (delete_cache_dir / "sam2_positive_points.npy").as_posix(),
                                negative_sam_points = (delete_cache_dir / "sam2_negative_points.npy").as_posix(), camera = (delete_cache_dir / "camera.pkl").as_posix(),
                                use_original_resolution = self.delete_use_original_resolution, output_dir = self.delete_output_dir
                            )
                    else:
                        stats = self._consume_training_stats(viz)
                        if stats is not None and bool(stats.get("paused", False)):
                            imgui.text("Reached final iteration. Click Pause Training to save.")
                            if imgui_utils.button("Pause Training", width=viz.button_w * 1.8):
                                single_training_step = True
                        elif imgui_utils.button("Stop", width=viz.button_w):
                            self.edit3D = False
                            self.edit_trainer.terminate()
                            self.edit_trainer.wait()

                    if self._consume_training_stats(viz) is not None:
                        self._draw_training_plots(viz)

                    imgui.end_tab_item()

                if imgui.begin_tab_item("drag")[0]:
                    imgui.text("Instruction:")
                    imgui.text("1.You should first select the area to be dragged. After that, you can turn off the mask display.")
                    imgui.text("2.Pressing 'q' together with the left mouse button allows you to select and drag the corresponding point.")
                    imgui.text("3.Pressing 'e' together with the left mouse button enables you to extend the corresponding point.")
                    imgui.text("4.Pressing 'z' together with the left mouse button allows you to rotate the corresponding point.")
                    imgui.text("5.Pressing 'x' together with the left mouse button allows you to translation (moving) of the point.")
                    imgui.separator_text("SAM Option")
                    label("Sam Type", viz.label_w)
                    _, self.drag_sam_option = imgui.combo(
                        "##SAM Type", 
                        self.drag_sam_option, 
                        ["No Sam", "Lang-sam", "SAM2(image)","SAM2(video)"]  
                    )

                    if self.drag_sam_option == 1:
                        label("Seg prompt", viz.label_w)
                        _, self.drag_seg_prompt = imgui.input_text("##seg prompt", self.drag_seg_prompt, 256)
                        self.drag_change = True if imgui.is_item_active() else False

                    if self.drag_sam_option == 2 or self.drag_sam_option == 3:
                        if imgui.radio_button("No Points", self.drag_point_option == 0):
                            self.drag_point_option = 0
                        imgui.same_line()
                        if imgui.radio_button("Positive Point", self.drag_point_option == 1):
                            self.drag_point_option = 1
                        imgui.same_line()
                        if imgui.radio_button("Negative Point", self.drag_point_option == 2):
                            self.drag_point_option = 2
                        imgui.same_line()
                        if self.drag_point_option == 1 or self.drag_point_option == 2:
                            if imgui.is_mouse_clicked(0):
                                uv = self._current_mouse_uv_in_render_image()
                                if uv is None:
                                    pass
                                elif self.drag_point_option == 1:
                                    self.drag_sam_positive_points.append(uv)
                                elif self.drag_point_option == 2:
                                    self.drag_sam_negative_points.append(uv)
                        if imgui_utils.button("clean SAM", width=viz.button_w):
                            self.drag_sam_positive_points = []
                            self.drag_sam_negative_points = []

                    drag_mask_path = self.runtime_drag_dir / "mask.pt"
                    if imgui_utils.button("seg", width=viz.button_w * 1.2):
                        drag_cache_dir = self.runtime_drag_dir
                        os.makedirs(drag_cache_dir, exist_ok=True)
                        np.save(
                            (drag_cache_dir / "sam2_positive_points.npy").as_posix(),
                            np.array(self.drag_sam_positive_points),
                        )
                        np.save(
                            (drag_cache_dir / "sam2_negative_points.npy").as_posix(),
                            np.array(self.drag_sam_negative_points),
                        )
                        origin = Image.fromarray(viz.result.image).convert("RGB")
                        self._clear_directory(drag_cache_dir / "render")
                        R = viz.extr.inverse()[:3, :3].T.numpy()
                        T = viz.extr.inverse()[:3, 3].numpy()
                        fov_rad = viz.fov / 360 * 2 * np.pi
                        cam = CustomCam(origin.size[0]//2, origin.size[1]//2, fov_rad, fov_rad, R, T, viz.extr.cuda())
                        with open((drag_cache_dir / "camera.pkl").as_posix(), 'wb') as f:
                            pickle.dump(cam, f)  
                        mask_trainer = get_3DGS_mask_command(
                            gs_source=viz.args.ply_file_paths[0],colmap_dir=viz.args.data_source,
                            sam_option=self.drag_sam_option,
                            seg_prompt=self.drag_seg_prompt,
                            camera=(drag_cache_dir / "camera.pkl").as_posix(),
                            positive_sam_points=(drag_cache_dir / "sam2_positive_points.npy").as_posix(),
                            negative_sam_points=(drag_cache_dir / "sam2_negative_points.npy").as_posix(),
                        )
                        mask_trainer.wait()
                    imgui.same_line()
                    if imgui_utils.button("cancel", width=viz.button_w * 1.2):
                        if drag_mask_path.is_file():
                            try:
                                drag_mask_path.unlink()
                            except OSError as exc:
                                print(f"[EditorWidget] Failed to remove drag mask: {exc}")
                    current_mask_path = (
                        drag_mask_path.as_posix()
                        if drag_mask_path.is_file() and os.access(drag_mask_path.as_posix(), os.R_OK)
                        else ""
                    )
                    imgui.same_line()
                    imgui.text(f"Path: {current_mask_path}")

                    imgui.separator_text("Drag Option")
                    self.keypoint_add()
                    self.keypoint_drag()
                    if imgui_utils.button("Init Drag", width=viz.button_w*1.2):
                        self.animation = True
                        control = animation_initialize(self.viz.args.ply_file_paths[0])
                        for key, value in control.items():
                            setattr(self, key, value)
                    imgui.same_line()
                    if imgui_utils.button("Clear Graph", width=viz.button_w*1.2):
                        with torch.no_grad():
                            self.is_animation = True
                            if hasattr(self, 'animate_tool'):
                                self.animate_tool.reset()
                            control = animation_reset()
                            for key, value in control.items():
                                setattr(self, key, value)
                    imgui.same_line()
                    if imgui_utils.button("Close Overlay", width=viz.button_w*1.2):
                        self.showing_overlay = False
                    imgui.same_line()
                    if imgui_utils.button("Open Overlay", width=viz.button_w*1.2):
                        self.showing_overlay = True

                    imgui.end_tab_item()

            imgui.end_tab_bar() 

        if self.edit_trainer!= None and self.edit_trainer.poll() is not None:
            self.edit3D = False

        viz.args.text_change = self.text_change
        viz.args.draw_image = self.draw_image
        viz.args.edit3D = self.edit3D
        viz.args.edit_single = self.edit_single
        viz.args.single_image = self.single_image

        viz.args.rec_start = self.rec_start
        viz.args.depth = self.depth
        viz.args.concat = self.concat
        viz.args.rec_end = self.rec_end
        viz.args.edit_image = edit_image
        viz.args.painting = self.editing_option >= 0  
        viz.args.points = self.points
        viz.args.current_color = self.current_color
        viz.args.line_width = self.line_width
        viz.args.showing_overlay = self.showing_overlay
        viz.args.stop_at_value = self.edit_train_steps - 1 if self.edit3D else -1
        viz.args.single_training_step = single_training_step
        viz.args.save_concat_ply_path = save_concat_ply_path
        if self.text_sam_positive_points != [] or self.text_sam_negative_points != []:
            viz.args.sam_positive_points = self.text_sam_positive_points
            viz.args.sam_negative_points = self.text_sam_negative_points
        elif self.delete_sam_positive_points != [] or self.delete_sam_negative_points != []:
            viz.args.sam_positive_points = self.delete_sam_positive_points
            viz.args.sam_negative_points = self.delete_sam_negative_points
        elif self.drag_sam_positive_points != [] or self.drag_sam_negative_points != []:
            viz.args.sam_positive_points = self.drag_sam_positive_points
            viz.args.sam_negative_points = self.drag_sam_negative_points
        else:
            viz.args.sam_positive_points = []
            viz.args.sam_negative_points = []

    def handle_mouse_input(self):
        if glfw.get_mouse_button(self.viz._glfw_window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.viz._glfw_window)

            region_left = self.viz.pane_w
            region_right = self.viz.content_width
            region_top = 0
            region_bottom = self.viz.content_height
            if region_left <= x <= region_right and region_top <= y <= region_bottom:
                if not self.is_drawing:
                    self.is_drawing = True
                self.points.append((x, y))
        else:
            self.is_drawing = False
            if self.points:
                self.points.append(None)
    
    def judge_move(self):
        if imgui.is_mouse_dragging(0) or imgui.is_mouse_dragging(1) or imgui.is_mouse_dragging(2):
            if imgui.get_mouse_drag_delta(0).x != 0 or imgui.get_mouse_drag_delta(1).x != 0 or imgui.get_mouse_drag_delta(2).x != 0:
                if imgui.get_mouse_drag_delta(0).y != 0 or imgui.get_mouse_drag_delta(1).y != 0 or imgui.get_mouse_drag_delta(2).y != 0:
                    return True
        if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
            return True
        if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
            return True
        if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
            return True
        if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
            return True
        if imgui.get_io().mouse_wheel != 0:
            return True
        return False
    
    def get_mask(self, ori, edit):
        ori = np.array(ori)
        edit = np.array(edit)
        diff = np.abs(ori - edit) 
        mask = np.any(diff > 0, axis=-1)
        mask_image = np.zeros_like(ori[:, :, 0])
        mask_image[mask] = 255
        mask_image_pil = Image.fromarray(np.stack([mask_image] * 3, axis=-1).astype(np.uint8))
        return mask_image_pil

    def draw_mask(self):
        if imgui.is_mouse_down(0):
            if not self.drawing_rect:
                self.start_rec_pos = imgui.get_mouse_pos()
                self.drawing_rect = True
            self.end_rec_pos = imgui.get_mouse_pos()

        if not imgui.is_mouse_down(0) and self.drawing_rect:
            self.drawing_rect = False
        if self.start_rec_pos and self.end_rec_pos:
            if self.start_rec_pos[0] >= self.viz.pane_w and self.end_rec_pos[0] >= self.viz.pane_w \
                   and self.start_rec_pos[1] >= 0 and self.end_rec_pos[1] >= 0: 
                self.rec_start, self.rec_end =  [self.start_rec_pos[0], self.start_rec_pos[1]], [self.end_rec_pos[0], self.end_rec_pos[1]]
            else:
                return
        else:
            self.rec_start, self.rec_end = None, None
    
    def edit_single_image(self, prompts, seed, image_path, mask_path):
        from threestudio.models.guidance.brushnet_guidance import (
                    BrushNetGuidance,
                )
        from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
        start_t = time.perf_counter()
        self._print_single_edit_start(prompts, seed, image_path, mask_path)

        try:
            brushnet = BrushNetGuidance(
                        OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98, "generator_seed": seed})
                    )

            image = Image.open(f"{image_path}")
            image = to_tensor(image).unsqueeze(0).permute(0,2,3,1).to("cuda")
            mask = Image.open(f"{mask_path}").convert('L') 
            mask = to_tensor(mask).unsqueeze(0).permute(0,2,3,1).repeat(1,1,1,3).float().to("cuda")
            masked_image =  image*(1-mask)
            prompt_utils = StableDiffusionPromptProcessor(
                {
                    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                    "prompt": prompts,
                }
            )()
            result = brushnet(
                    masked_image,
                    mask,
                    prompt_utils,
                )
            elapsed = time.perf_counter() - start_t
            self._print_single_edit_done(elapsed)
            return np.array(result['edit_images'].squeeze(0).cpu())
        except Exception as exc:
            elapsed = time.perf_counter() - start_t
            self._print_single_edit_fail(elapsed, exc)
            raise

    def remove_single_image(self, prompts, image):
        start_t = time.perf_counter()
        model_name = "qwen-image-edit-max"
        output_size = "1024*1024"

        try:
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = np.clip(image, 0.0, 1.0)
                    image = (image * 255.0).astype(np.uint8)
                pil_image = Image.fromarray(image).convert("RGB")
            else:
                pil_image = image.convert("RGB")

            self._print_remove_start(
                prompt=prompts,
                image_size=pil_image.size,
                model=model_name,
                size=output_size,
            )

            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

            api_cfg = find_repo_root(__file__) / "configs" / "api.json"
            with open(api_cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            api_key = data.get("api_key")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_base64},
                        {"text": f'{prompts}, After removal, make sure the background of the area is filled consistently with the surrounding area so that the modified image looks authentic and without any abrupt traces.'},
                    ],
                }
            ]

            self._print_remove_stage("Submitting request to image-edit model ...")
            response = MultiModalConversation.call(
                api_key=api_key,
                model=model_name,
                messages=messages,
                stream=False,
                n=1,
                watermark=False,
                negative_prompt=" ",
                prompt_extend=False,
                size=output_size,
            )

            if response.status_code != 200:
                error_text = f"{response.code}: {response.message}"
                self._print_remove_fail(time.perf_counter() - start_t, error_text)
                return None

            content_list = response.output.choices[0].message.content or []
            image_urls = [item.get("image") for item in content_list if isinstance(item, dict) and item.get("image")]
            if not image_urls:
                self._print_remove_fail(time.perf_counter() - start_t, "No image URL returned by server.")
                return None

            save_path = (self.runtime_delete_dir / "Reference.png").as_posix()
            self._print_remove_stage(f"Downloading {len(image_urls)} generated image(s) ...")
            for idx, image_url in enumerate(image_urls, start=1):
                self._print_remove_stage(f"  Download {idx}/{len(image_urls)}")
                r = requests.get(image_url, stream=True, timeout=300)
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)

            self._print_remove_done(time.perf_counter() - start_t, save_path)
            return np.array(Image.open(save_path).convert("RGB"))
        except Exception as exc:
            self._print_remove_fail(time.perf_counter() - start_t, str(exc))
            return None

    def keypoint_add(self):
        if imgui.is_mouse_clicked(0) and ('q' in self.viz.current_pressed_keys or 'e' in self.viz.current_pressed_keys):
            if not self.animation:
                control = animation_initialize(self.viz.args.ply_file_paths[0])
                for key, value in control.items():
                    setattr(self, key, value)
                self.animation = True
                return 
            else:
                # fid = torch.tensor(self.animation_time).cuda().float()
                with torch.no_grad():
                    self.viz.args.drag_point = (imgui.get_mouse_pos().x-self.viz.pane_w, imgui.get_mouse_pos().y)

        if hasattr(self.viz.result,"p3d") and self.viz.result.p3d is not None:
            if not hasattr(self, 'p3d') or not torch.equal(self.p3d, self.viz.result.p3d):
                p3d = self.viz.result.p3d.cuda()
                nodes = self.control_nodes + self.animation_trans_bias if self.animation_trans_bias is not None else self.control_nodes
                keypoint_idxs = torch.tensor([(p3d - nodes).norm(dim=-1).argmin()]).cuda()
                keypoint_idxs = self.animate_tool.add_n_ring_nbs(keypoint_idxs, n=2)
                keypoint_3ds = nodes[keypoint_idxs]
                if 'e' in self.viz.current_pressed_keys:
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, expand=True)
                elif 'q' in self.viz.current_pressed_keys:
                    self.deform_keypoints.add_kpts(keypoint_3ds, keypoint_idxs, expand=False)
                print(f'Add kpt: {self.deform_keypoints.selective_keypoints_idx_list}')
                self.p3d = self.viz.result.p3d
        

        if hasattr(self, 'deform_keypoints'):
            self.viz.args.selective_keypoints_idx_list = self.deform_keypoints.selective_keypoints_idx_list
            self.viz.args.source_drag_points = self.deform_keypoints.get_kpt()
            self.viz.args.target_drag_points = self.deform_keypoints.get_deformed_kpt_np()

    def keypoint_drag(self):
        if not self.animation:
            # print("Please switch to animation mode!")
            return
        if len(self.deform_keypoints.get_kpt()) == 0:
            return
        if self.animate_tool is None:
            animation_initialize()
        if imgui.is_mouse_dragging(0):
            new_delta = imgui.get_mouse_drag_delta(0)
            delta = new_delta - self.last_drag_delta
            dx = delta.x
            dy = delta.y
            if "z" in self.viz.current_pressed_keys:
                rot = self.viz.result.cam_params.cpu().numpy()[:3, :3]
                up = rot[:3, 1]
                forward = rot[:3, 2]
                rotvec_z = forward * np.radians(-0.05 * dx)
                rotvec_y = up * np.radians(-0.05 * dy)
                rot_mat = (R.from_rotvec(rotvec_z)).as_matrix() @ (R.from_rotvec(rotvec_y)).as_matrix()
                self.deform_keypoints.set_rotation_delta(rot_mat)
                animated_pcl, quat, ani_d_scaling = self.animate_tool.deform_arap(handle_idx=self.deform_keypoints.get_kpt_idx(), handle_pos=self.deform_keypoints.get_deformed_kpt_np(), init_verts=None, return_R=True)
                self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                self.animation_rot_bias = quat
                self.animation_scaling_bias = ani_d_scaling
            elif "x" in self.viz.current_pressed_keys:
                delta = 0.0010 * self.viz.result.cam_params.cpu().numpy()[:3, :3] @ np.array([dx, -dy, 0])
                self.deform_keypoints.update_delta(delta)
                animated_pcl, quat, ani_d_scaling = self.animate_tool.deform_arap(handle_idx=self.deform_keypoints.get_kpt_idx(), handle_pos=self.deform_keypoints.get_deformed_kpt_np(), init_verts=None, return_R=True)
                self.animation_trans_bias = animated_pcl - self.animate_tool.init_pcl
                self.animation_rot_bias = quat
                self.animation_scaling_bias = ani_d_scaling

            self.last_drag_delta = new_delta
        else:
            self.last_drag_delta = imgui.ImVec2(0, 0)

        if self.animation_trans_bias is not None :
            d_values = self.animator(self.gaussians_xyz, self.control_nodes, self.animation_trans_bias)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
            d_rotation_bias = d_values['d_rotation_bias']

            self.viz.args.drag = {
                'xyz': d_xyz,
                'rotation': d_rotation,
                'scaling': d_scaling,
                'opacity': d_opacity,
                'color': d_color,
                'rotation_bias': d_rotation_bias
            }

    def close(self):
        if hasattr(self, "_remove_executor") and self._remove_executor is not None:
            self._remove_executor.shutdown(wait=False, cancel_futures=True)
        if self.edit_trainer != None:
            self.edit_trainer.terminate()
            self.edit_trainer.wait()
        super().close()
