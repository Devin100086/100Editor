from widgets.widget import Widget
from imgui_bundle import imgui,ImVec2, portable_file_dialogs as pfd
from utils.gui_utils import imgui_utils
from utils.gui_utils.easy_imgui import label
import os
from utils.command_utils import *

class ShowColmapWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "showing colmap")
        self.data_source = ""
        self.showing_colmap = False
        self.colmap_process = None
        # self.resolution = 1024
    
    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            if imgui_utils.button(f"Browse sparse", width=viz.button_large_w):
                data_source = self._select_folder()
                if data_source:
                    self.data_source = data_source
            imgui.same_line()
            imgui.text(f"Selected Sparse: {self.data_source}")

            # label("Resolution", viz.label_w)
            # _changed, self.resolution = imgui.input_int("##Resolution", self.resolution, 64)

            if self.showing_colmap== False or self.colmap_process.poll() != None:
                can_show = bool(self.data_source)
                if not can_show:
                    imgui.begin_disabled()
                clicked_show = imgui_utils.button(f"Show", width=viz.button_large_w)
                if not can_show:
                    imgui.end_disabled()
                if clicked_show and can_show:
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
        dialog = pfd.select_folder("Select Sparse Folder")
        folder_path = dialog.result() if hasattr(dialog, "result") else dialog
        if isinstance(folder_path, (list, tuple)):
            folder_path = folder_path[0] if folder_path else ""
        return folder_path if folder_path else None
