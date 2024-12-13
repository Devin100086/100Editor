import os
from imgui_bundle import imgui
import tkinter as tk
from tkinter import filedialog
from lumina3D_utils.gui_utils import imgui_utils
from lumina3D_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
import glfw
from PIL import Image
from OpenGL.GL import *

 
class EditorWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Editor")

        self.points = []
        self.current_color = [1.0, 1.0, 1.0, 1.0]
        self.line_width = 2.0
        self.text_prompt = "turn him a clown"
        self.mask_prompt = "turn him a clown"
        self.sketch_prompt = "turn him a clown"
        self.is_drawing = False
        self.turn_camera = False

        self.text_change = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        edit_image = False
        if show:
            if imgui.begin_tab_bar("MyTabBar"):

                if imgui.begin_tab_item("text")[0]:
                    label("prompt", viz.label_w)
                    changed, self.text_prompt = imgui.input_text("##Prompt", self.text_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    if imgui_utils.button("Edit", width=viz.button_w):
                        pass
                    imgui.end_tab_item()
                
                if imgui.begin_tab_item("mask")[0]:
                    label("prompt", viz.label_w)
                    changed, self.sketch_prompt = imgui.input_text("##Prompt", self.sketch_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    imgui.end_tab_item()
                
                if imgui.begin_tab_item("sketch")[0]:
                    changed, self.line_width = imgui.slider_float("width", self.line_width, 1.0, 10.0)
                    _, self.current_color = imgui.color_edit4("color choice", self.current_color)
                    if imgui.button("clear"):
                        self.points = []
                    label("Painting", viz.label_w)
                    changed, self.turn_camera = imgui.checkbox("##painting", self.turn_camera)

                    label("prompt", viz.label_w)
                    changed, self.sketch_prompt = imgui.input_text("##Prompt", self.sketch_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False

                    if imgui_utils.button("Edit", width=viz.button_w):
                        edit_image = True

                    if self.turn_camera:
                        self.handle_mouse_input()
                    imgui.end_tab_item()

            imgui.end_tab_bar()    
        print(viz.edit_image)
        viz.args.text_change = self.text_change

        viz.args.edit_image = edit_image
        viz.args.turn_camera = self.turn_camera       
        viz.args.points = self.points
        viz.args.current_color = self.current_color
        viz.args.line_width = self.line_width
        
    def handle_mouse_input(self):
        if glfw.get_mouse_button(self.viz._glfw_window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.viz._glfw_window)

            region_left = self.viz.content_width - self.viz.pane_w
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