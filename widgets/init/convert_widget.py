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
        self.use_gpu = 0
        self.gpu_items = [str(i) for i in range(torch.cuda.device_count())]
        self.items = self.list_runs_and_colmap()
        self.gpu_monitor = Monitor(0.5)
        self.cuda_version = torch.version.cuda
        self.colmap_status = "waiting..."
    
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
            changed, self.use_gpu = imgui.combo(
                "choose gpu",
                self.use_gpu,
                self.gpu_items
            )
            if imgui_utils.button("colmap", width=viz.button_w):
                self.colmap_progress = 0.0
                self.colmap_process()
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
        camera = "OPENCV"
        colmap_command = '"{}"'.format(self.colmap_executable) if self.colmap_executable != "Default" else "colmap"

        os.makedirs(self.source_path + "/distorted/sparse", exist_ok=True)
        ## Feature extraction
        feat_extracton_cmd = colmap_command + " feature_extractor "\
            "--database_path " + self.source_path + "/distorted/database.db \
            --image_path " + self.source_path + "/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model " + camera + " \
            --SiftExtraction.use_gpu " + str(self.gpu_items[self.use_gpu])
        exit_code = os.system(feat_extracton_cmd)
        
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ## Feature matching
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + self.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(self.gpu_items[self.use_gpu])
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ### Bundle adjustment
        # The default Mapper tolerance is unnecessarily large,
        # decreasing it speeds up bundle adjustment steps.
        mapper_cmd = (colmap_command + " mapper \
            --database_path " + self.source_path + "/distorted/database.db \
            --image_path "  + self.source_path + "/input \
            --output_path "  + self.source_path + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001")
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ### Image undistortion
        ## We need to undistort our images into ideal pinhole intrinsics.
        img_undist_cmd = (colmap_command + " image_undistorter \
            --image_path " + self.source_path + "/input \
            --input_path " + self.source_path + "/distorted/sparse/0 \
            --output_path " + self.source_path + "\
            --output_type COLMAP")
        exit_code = os.system(img_undist_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

        files = os.listdir(self.source_path + "/sparse")
        os.makedirs(self.source_path + "/sparse/0", exist_ok=True)
        # Copy each file from the source directory to the destination directory
        for file in files:
            if file == '0':
                continue
            source_file = os.path.join(self.source_path, "sparse", file)
            destination_file = os.path.join(self.source_path, "sparse", "0", file)
            shutil.move(source_file, destination_file)

        self.colmap_status = "finish!"