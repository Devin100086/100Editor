import io
import logging
from threading import Thread
import shutil
import time
from widgets.widget import Widget
from imgui_bundle import imgui,ImVec2, portable_file_dialogs as pfd
from utils.gui_utils import imgui_utils
from utils.gui_utils.easy_imgui import label
import os
import GPUtil
import torch
import cv2
from utils.command_utils import sfm_reconstruction,vggt_reconstruction

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
        self.selected_colmap = 0  # 0 for SfM, 1 for vggt
        self.camera_models = ["OPENCV", "OPENCV_FISHEYE"]
        self.camera_model_index = 0
    
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
                if source_path:
                    self.source_path = source_path
                self.frame_number = 0
            imgui.same_line()
            imgui.text(f"Source Path: {self.source_path}")

            if imgui.radio_button("SfM", self.selected_colmap == 0):
                self.selected_colmap = 0 
            imgui.same_line(viz.label_w)
            if imgui.radio_button("VGGT", self.selected_colmap == 1):
                self.selected_colmap = 1 
            if self.selected_colmap == 0: 
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
                camera_model_width = max(viz.button_w * 1.8, 140)
                imgui.set_next_item_width(camera_model_width)
                _, self.camera_model_index = imgui.combo(
                    "camera model",
                    self.camera_model_index,
                    self.camera_models
                )

            if imgui_utils.button("colmap", width=viz.button_w):
                self.colmap = False
                self.colmap_progress = 0.0
                if self.selected_colmap == 0:
                    self.colmap_rec = self.sfm_process()
                elif self.selected_colmap == 1:
                    self.colmap_rec = self.vggt_process()
            
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
        dialog = pfd.select_folder("Select Source Folder")
        folder_path = dialog.result() if hasattr(dialog, "result") else dialog
        if isinstance(folder_path, (list, tuple)):
            folder_path = folder_path[0] if folder_path else ""
        return folder_path if folder_path else None

    def list_runs_and_colmap(self):
        self.items = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith("colmap"):
                    current_path = os.path.join(root, file)
                    self.items.append(str(current_path))
        return sorted(self.items)
    
    def sfm_process(self):
        camera_model = self.camera_models[self.camera_model_index]
        if self.colmap_executable == "Default":
            return sfm_reconstruction(self.source_path, "", self.use_gpu, camera_model)
        else:
            return sfm_reconstruction(self.source_path, self.colmap_executable, self.use_gpu, camera_model)
    
    def vggt_process(self):
        return vggt_reconstruction(self.source_path)
    
