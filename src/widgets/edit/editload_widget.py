import os
from imgui_bundle import imgui, portable_file_dialogs as pfd
from utils.gui_utils import imgui_utils
from utils.gui_utils.easy_imgui import label
from widgets.common.load_widget_ply import LoadWidget

class EditLoadWidget(LoadWidget):
    def __init__(self, viz, root):
        super().__init__(viz, root)
        self.plys = [root]
        self.data_source = "/media/wucunqi/data/results/face"

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            if imgui_utils.button(f"Browse ply", width=viz.button_large_w):
                ply_file = self._select_ply()
                if ply_file:
                    self.plys = [ply_file]
            imgui.same_line()
            imgui.text(f"Scene : " + os.path.basename(self.plys[0]))
            if imgui_utils.button(f"Browse data", width=viz.button_large_w):
                data_source = self._select_folder()
                if data_source:
                    self.data_source = data_source
            imgui.same_line()
            imgui.text(f"Data source : " + os.path.basename(self.data_source))
        
        viz.args.ply_file_paths = self.plys
        viz.args.edit_text = ""
        viz.args.data_source = self.data_source
        viz.args.current_ply_names = [self.plys[0].replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")]
    
    def _select_folder(self):
        dialog = pfd.select_folder("Select Folder")
        folder_path = dialog.result() if hasattr(dialog, "result") else dialog
        if isinstance(folder_path, (list, tuple)):
            folder_path = folder_path[0] if folder_path else ""
        return folder_path if folder_path else None
