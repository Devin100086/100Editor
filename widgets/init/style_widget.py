from widgets.widget import Widget
from imgui_bundle import imgui
from lumina3D_utils.gui_utils import imgui_utils


class StyleWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "style")
    @imgui_utils.scoped_by_object_id
    def __call__(self, show = True):
        if show:
            imgui.show_style_editor()