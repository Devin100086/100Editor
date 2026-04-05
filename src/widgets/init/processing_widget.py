from widgets.widget import Widget
from imgui_bundle import imgui,ImVec2, portable_file_dialogs as pfd
from utils.gui_utils import imgui_utils
from utils.gui_utils.easy_imgui import label
import os
import cv2
 
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
                if video_path:
                    self.video_path = video_path
                self.frame_number = 0
            imgui.same_line()
            imgui.text(f"Selected Video: {self.video_path}")
            if imgui_utils.button(f"output Path", width=viz.button_w):
                output_path = self._select_folder()
                if output_path:
                    self.output_path = output_path
            imgui.same_line()
            imgui.text(f"Output Path: {self.output_path}")
            if imgui_utils.button(f"Processing", width=viz.button_w):
                self._process_video(self.video_path, self.output_path)
            imgui.same_line()
            label(f"frame: {self.frame_number}",viz.label_w_large)
            label("progress: ")
            imgui.progress_bar(fraction=self.progress)
            
    def _select_video(self):
        dialog = pfd.open_file("Select Video File")
        file_path = dialog.result() if hasattr(dialog, "result") else dialog
        if isinstance(file_path, (list, tuple)):
            file_path = file_path[0] if file_path else ""
        return file_path if file_path else None
    
    def _select_folder(self):
        dialog = pfd.select_folder("Select Output Folder")
        folder_path = dialog.result() if hasattr(dialog, "result") else dialog
        if isinstance(folder_path, (list, tuple)):
            folder_path = folder_path[0] if folder_path else ""
        return folder_path if folder_path else None

    def _process_video(self,video_path, output_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_path, exist_ok=True)
        self._extract_frames(video_path, output_path, video_name)
    
    def _extract_frames(self, video_path, output_path, video_name):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        fps = fps * 5
        frame_interval = round(fps / 50)
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
