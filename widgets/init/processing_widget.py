from widgets.widget import Widget
from imgui_bundle import imgui,ImVec2
from lumina3D_utils.gui_utils import imgui_utils
import tkinter as tk
from lumina3D_utils.gui_utils.easy_imgui import label
import os
import cv2
from tkinter import filedialog
 
class ProcessingWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "processing")
        self.video_path = os.getcwd()
        self.output_path = os.getcwd()
        self.progress = 0.0
        self.frame_number = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self, show = True):
        viz = self.viz
        if show:
            if imgui_utils.button(f"Video Path", width=viz.button_w):
                self.progress = 0.0
                video_path = self._select_video()
                self.video_path = self.video_path if isinstance(video_path, tuple) else video_path
                self.frame_number = 0
            imgui.same_line()
            imgui.text(f"Selected Video: {self.video_path}")
            if imgui_utils.button(f"output Path", width=viz.button_w):
                output_path = self._select_folder()
                self.output_path = self.output_path if isinstance(output_path, tuple) else output_path
            imgui.same_line()
            imgui.text(f"Output Path: {self.output_path}")
            if imgui_utils.button(f"Processing", width=viz.button_w):
                self._process_video(self.video_path, self.output_path)
            imgui.same_line()
            label(f"frame: {self.frame_number}",viz.label_w_large)
            label("progress: ")
            imgui.progress_bar(fraction=self.progress)
            
    def _select_video(self):
        root = tk.Tk()
        root.withdraw()
        file_types = [
            ("Video Files", "*.mp4"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        return file_path
    
    def _select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        return folder_path

    def _process_video(self,video_path, output_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_path, exist_ok=True)
        self._extract_frames(video_path, output_path, video_name)
    
    def _extract_frames(self, video_path, output_path, video_name):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        fps = fps * 5
        frame_interval = round(fps / 30)
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_filename = f"{output_path}/{video_name}_{frame_count:05d}.jpg"
                cv2.imwrite(frame_filename, frame)
                self.frame_number += 1
            frame_count += 1
            self.progress += 0.01
            imgui.progress_bar(fraction=self.progress)
        video.release()