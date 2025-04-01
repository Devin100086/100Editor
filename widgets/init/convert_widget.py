import io
import logging
from threading import Thread
import shutil
import time
from widgets.widget import Widget
from imgui_bundle import imgui,ImVec2
from lumina3D_utils.gui_utils import imgui_utils
import tkinter as tk
from lumina3D_utils.gui_utils.easy_imgui import label
import os
import GPUtil
import torch
import cv2
from tkinter import filedialog
from lumina3D_utils.command_utils import colmap_reconstruction

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.start()
        self.gpu = GPUtil.getGPUs()

    def run(self):
        while not self.stopped:
            self.gpu = GPUtil.getGPUs()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

class ConverWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "convert")
        self.colmap_executable = "Default"
        self.root= os.getcwd()
        self.source_path = "choose your data path..."
        self.colmap_progress = 0.0
        self.use_gpu = True
        self.items = self.list_runs_and_colmap()
        self.gpu_monitor = Monitor(0.5)
        self.cuda_version = torch.version.cuda
        self.colmap_status = "waiting..."
        self.colmap_rec = None
        self.colmap = True
    
    def close(self):
        self.gpu_monitor.stop()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show = True):
        viz = self.viz
        if show:
            if imgui_utils.button(f"Source", width=viz.button_w):
                self.colmap_status = "waiting..."
                self.progress = 0.0
                source_path = self._select_folder()
                self.source_path = self.source_path if isinstance(source_path, tuple) else source_path
                self.frame_number = 0
            imgui.same_line()
            imgui.text(f"Source Path: {self.source_path}")

            if imgui.begin_popup(f"browse_colmap_popup"):
                for item in self.items:
                    clicked = imgui.menu_item_simple(os.path.relpath(item, self.root))
                    if clicked:
                        self.colmap_executable = item
                imgui.end_popup()

            if imgui_utils.button(f"Browse ", width=viz.button_w):
                imgui.open_popup(f"browse_colmap_popup")
                self.items = self.list_runs_and_colmap()
            imgui.same_line() 
            imgui.text(f"colmap path: {self.colmap_executable}")
            imgui.set_next_item_width(viz.button_w)

            changed, self.use_gpu = imgui.checkbox("Use GPU", self.use_gpu)

            if imgui_utils.button("colmap", width=viz.button_w):
                self.colmap = False
                self.colmap_progress = 0.0
                self.colmap_rec = self.colmap_process()
            
            if self.colmap_rec!= None and self.colmap_rec.poll() is not None:
                self.colmap_status = "finish!"

            imgui.same_line()
            imgui.text(f"{self.colmap_status}")

            imgui.new_line()
            label("Device:", viz.label_w)
            imgui.text(f"{self.gpu_monitor.gpu[0].name}")
            for i, gpu in enumerate(self.gpu_monitor.gpu):
                label(f"gpu{i}:")
                label(f"{self.gpu_monitor.gpu[0].temperature}° C" ,viz.label_w)
                imgui.progress_bar(gpu.memoryUsed / gpu.memoryTotal, imgui.ImVec2(300, 30), f"{gpu.memoryUsed / 1024:.2f}GB / {gpu.memoryTotal / 1024:.2f}GB")
                
    def _select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        return folder_path

    def list_runs_and_colmap(self):
        self.items = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith("colmap"):
                    current_path = os.path.join(root, file)
                    self.items.append(str(current_path))
        return sorted(self.items)
    
    def colmap_process(self):
        if self.colmap_executable == "Default":
            return colmap_reconstruction(self.source_path, "", self.use_gpu)
        else:
            return colmap_reconstruction(self.source_path, self.colmap_executable, self.use_gpu)