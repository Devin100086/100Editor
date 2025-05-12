from widgets.widget import Widget
from imgui_bundle import imgui,ImVec2
from lumina3D_utils.gui_utils import imgui_utils
import tkinter as tk
from lumina3D_utils.gui_utils.easy_imgui import label
import os
from lumina3D_utils.command_utils import *
from tkinter import filedialog

class ShowColmapWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "showing colmap")
        self.data_source = "/home/wucunqi/Desktop/results/face/sparse/0"
        self.showing_colmap = False
        self.colmap_process = None
        # self.resolution = 1024
    
    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            if imgui_utils.button(f"Browse sparse", width=viz.button_large_w):
                self.data_source = self._select_folder()
            imgui.same_line()
            imgui.text(f"Selected Sparse: {self.data_source}")

            # label("Resolution", viz.label_w)
            # _changed, self.resolution = imgui.input_int("##Resolution", self.resolution, 64)

            if self.showing_colmap== False or self.colmap_process.poll() != None:
                if imgui_utils.button(f"Show colmap", width=viz.button_large_w):
                    self.colmap_process = showing_colmap_command(self.data_source)
                    self.showing_colmap = True
            else:
                if imgui_utils.button(f"Stop", width=viz.button_large_w):
                    self.colmap_process.terminate()
                    self.colmap_process.wait()
                    self.showing_colmap = False

        viz.args.data_source = self.data_source
        # viz.args.resolution = self.resolution

    def _select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        return folder_path