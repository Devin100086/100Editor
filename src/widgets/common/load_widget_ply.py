import os
from imgui_bundle import imgui, portable_file_dialogs as pfd
from utils.gui_utils import imgui_utils
from utils.gui_utils.easy_imgui import label
from widgets.widget import Widget

 
class LoadWidget(Widget):
    def __init__(self, viz, root):
        super().__init__(viz, "Load")
        self.root = root
        self.plys: list[str] = [root]
        self.use_splitscreen = False
        self.highlight_border = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            plys_to_remove = []

            for i, ply in enumerate(self.plys):
                if imgui_utils.button(f"Browse {i + 1}", width=viz.button_large_w):
                    ply_file = self._select_ply()
                    if ply_file:
                        self.plys[i] = ply_file
                imgui.same_line()
                if i > 0:
                    if imgui_utils.button(f"Remove {i + 1}", width=viz.button_large_w):
                        plys_to_remove.append(i)
                    imgui.same_line()
                imgui.text(f"Scene {i + 1}: " + os.path.basename(ply))

            for i in plys_to_remove[::-1]:
                self.plys.pop(i)
            if imgui_utils.button("Add Scene", width=viz.button_large_w):
                self.plys.append(self.plys[-1])

            use_splitscreen, self.use_splitscreen = imgui.checkbox("Splitscreen", self.use_splitscreen)
            highlight_border, self.highlight_border = imgui.checkbox("Highlight Border", self.highlight_border)

        viz.args.highlight_border = self.highlight_border
        viz.args.use_splitscreen = self.use_splitscreen
        viz.args.ply_file_paths = self.plys
        viz.args.current_ply_names = [
            ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_") for ply in self.plys
        ]
    
    def _select_ply(self):
        dialog = pfd.open_file("Select PLY File")
        file_path = dialog.result() if hasattr(dialog, "result") else dialog
        if isinstance(file_path, (list, tuple)):
            file_path = file_path[0] if file_path else ""
        return file_path if file_path else None
