import subprocess
from imgui_bundle import imgui
import tkinter as tk
from tkinter import filedialog
from lumina3D_utils.gui_utils import imgui_utils
from lumina3D_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
import glfw
from PIL import Image
import numpy as np
from OpenGL.GL import *

 
class EditorWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Editor")
        # Option
        self.lambda_l1 = 10
        self.lambda_p = 10
        self.lambda_anchor_color = 0
        self.lambda_anchor_geo = 50
        self.lambda_anchor_scale = 50
        self.lambda_anchor_opacity = 50
        self.edit_until_step = 1000
        self.per_editing_step = 10
        self.edit_begin_step = 0
        self.edit_cam_num = 48
        self.edit_train_steps = 1500
        
        # text-edit
        self.guidance_type = ["InstructPix2Pix","ControlNet-Pix2Pix"]
        self.guidance_item = 0
        self.text_prompt = "turn him a clown"
        self.text_edit_trainer = None

        # mask-edit
        self.mask_prompt = "turn him a clown"
        self.mask = False
        self.start_rec_pos = None
        self.end_rec_pos = None
        self.rec_start = None 
        self.rec_end = None
        self.drawing_rect = False

        # sketch-edit
        self.points = []
        self.current_color = [1.0, 1.0, 1.0, 1.0]
        self.line_width = 2.0
        self.sketch_prompt = "turn him a clown"
        self.is_drawing = False
        self.turn_camera = False
        

        self.draw_image = False
        self.edit3D = False
        self.text_change = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        edit_image = False
        if show:
            if imgui.begin_tab_bar("EditBar"):
                if imgui.begin_tab_item("Option")[0]:
                    label("Camera Num", viz.label_w)
                    _, self.edit_cam_num = imgui.slider_int("##Camera Num", self.edit_cam_num, 12, 200, format="%d")
                    label("Total Step", viz.label_w)
                    _, self.edit_train_steps = imgui.slider_int("##Total Step", self.edit_train_steps, 0, 5000, format="%d")
                    label("Lambda L1", viz.label_w)
                    _, self.lambda_l1 = imgui.slider_int("##Lambda L1", self.lambda_l1, 0, 100, format="%d")
                    label("Lambda Perceptual", viz.label_w)
                    _, self.lambda_p = imgui.slider_int("##Lambda Perceptual", self.lambda_p, 0, 100, format="%d")
                    label("Lambda Anchor Color", viz.label_w)
                    _, self.lambda_anchor_color = imgui.slider_int("##Lambda Anchor Color", self.lambda_anchor_color, 0, 500, format="%d")
                    label("Lambda Anchor Geo", viz.label_w)
                    _, self.lambda_anchor_geo = imgui.slider_int("##Lambda Anchor Geo", self.lambda_anchor_geo, 0, 500, format="%d")
                    label("Lambda Anchor Scale", viz.label_w)
                    _, self.lambda_anchor_scale = imgui.slider_int("##Lambda Anchor Scale", self.lambda_anchor_scale, 0, 500, format="%d")
                    label("Lambda Anchor Opacity", viz.label_w)
                    _, self.lambda_anchor_opacity = imgui.slider_int("##Lambda Anchor Opacity", self.lambda_anchor_opacity, 0, 500, format="%d")
                    label("Edit Until Step", viz.label_w)
                    _, self.edit_until_step = imgui.slider_int("##Edit Until Step", self.edit_until_step, 0, 5000, format="%d")
                    label("Edit Begining", viz.label_w)
                    _, self.edit_begin_step = imgui.slider_int("##Edit Begining", self.edit_begin_step, 0, 5000, format="%d")
                    label("Edit Interval", viz.label_w)
                    _, self.per_editing_step = imgui.slider_int("##Edit Interval", self.per_editing_step, 4, 48, format="%d")
                    imgui.end_tab_item()

                if imgui.begin_tab_item("text")[0]:
                    label("guidance_type", viz.label_w)
                    clicked, self.guidance_item = imgui.combo(
                        "##guidance_type",                
                        self.guidance_item,           
                        self.guidance_type                   
                    )
                    label("prompt", viz.label_w)
                    changed, self.text_prompt = imgui.input_text("##Prompt", self.text_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    if not self.edit3D:
                        if imgui_utils.button("Edit", width=viz.button_w):
                            self.edit3D = True
                            self.text_edit_trainer = subprocess.Popen([
                                "python", 
                                "EditorGS/GUIEditor/train_edit.py", 
                                "--gs_source",str(viz.args.ply_file_paths[0]),
                                "--colmap_dir",str(viz.args.data_source),
                                "--edit_cam_num", str(self.edit_cam_num),
                                "--guidance_type", str(self.guidance_type[self.guidance_item]),
                                "--text_prompt", str(self.text_prompt),
                                "--edit_train_steps", str(self.edit_train_steps),
                                "--per_editing_step", str(self.per_editing_step),
                                "--edit_begin_step", str(self.edit_begin_step),
                                "--edit_until_step", str(self.edit_until_step),
                                "--lambda_l1", str(self.lambda_l1),
                                "--lambda_p", str(self.lambda_p),
                                "--lambda_anchor_color", str(self.lambda_anchor_color),
                                "--lambda_anchor_geo", str(self.lambda_anchor_geo),
                                "--lambda_anchor_scale", str(self.lambda_anchor_scale),
                                "--lambda_anchor_opacity", str(self.lambda_anchor_opacity)
                            ])
                    else:
                        if imgui_utils.button("Stop", width=viz.button_w):
                            self.edit3D = False
                            self.text_edit_trainer.terminate()
                            self.text_edit_trainer.wait()
                    imgui.end_tab_item()
                
                if imgui.begin_tab_item("mask")[0]:
                    self.points = []
                    if imgui.button("clear"):
                        self.rec_start = None
                        self.rec_end = None
                    label("Paint Mask", viz.label_w)
                    changed, self.mask = imgui.checkbox("##Mask", self.mask)
                    label("prompt", viz.label_w)
                    changed, self.sketch_prompt = imgui.input_text("##Prompt", self.sketch_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    if not self.mask and not self.text_change and self.judge_move():  
                        self.draw_image = False
                        self.rec_start, self.rec_end = None, None
                    if self.mask:
                        self.draw_mask()
                        self.draw_image = True
                        imgui.begin_disabled()
                        if imgui_utils.button("Edit", width=viz.button_w):
                            pass
                        imgui.end_disabled()    
                    else:
                        edit_image = True
                        if imgui_utils.button("Edit", width=viz.button_w):
                            self.edit3D = False
                            mask = self.get_mask(viz.origin_image, viz.edit_image)
                            mask.save("mask.png")
                            viz.origin_image.save("origin.png")
                            
                    imgui.end_tab_item()
                
                if imgui.begin_tab_item("sketch")[0]:
                    self.rec_start = None
                    self.rec_end = None
                    changed, self.line_width = imgui.slider_float("width", self.line_width, 1.0, 10.0)
                    _, self.current_color = imgui.color_edit4("color choice", self.current_color)
                    if imgui.button("clear"):
                        self.points = []
                    label("Painting", viz.label_w)
                    changed, self.turn_camera = imgui.checkbox("##painting", self.turn_camera)

                    label("prompt", viz.label_w)
                    changed, self.sketch_prompt = imgui.input_text("##Prompt", self.sketch_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    if not self.turn_camera and not self.text_change and self.judge_move():  
                        self.draw_image = False
                        self.points = []
                    if self.turn_camera:
                        self.handle_mouse_input()
                        self.draw_image = True
                        imgui.begin_disabled()
                        if imgui_utils.button("Edit", width=viz.button_w):
                            pass
                        imgui.end_disabled()    
                    else:
                        edit_image = True
                        if imgui_utils.button("Edit", width=viz.button_w):
                            self.edit3D = False
                            mask = self.get_mask(viz.origin_image, viz.edit_image)
                            mask.save("mask.png")
                            viz.edit_image.save("edit.png")
                            viz.origin_image.save("origin.png")
                    imgui.end_tab_item()

            imgui.end_tab_bar()   

        viz.args.text_change = self.text_change
        viz.args.draw_image = self.draw_image
        viz.args.edit3D = self.edit3D

        viz.args.rec_start = self.rec_start
        viz.args.rec_end = self.rec_end
        viz.args.edit_image = edit_image
        viz.args.turn_camera = self.turn_camera  
        viz.args.mask = self.mask     
        viz.args.points = self.points
        viz.args.current_color = self.current_color
        viz.args.line_width = self.line_width
        
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
    
    def close(self):
        if self.text_edit_trainer != None:
            self.text_edit_trainer.terminate()
            self.text_edit_trainer.wait()
        super().close()
























