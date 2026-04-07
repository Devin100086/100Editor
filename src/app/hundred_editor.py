from imgui_bundle import imgui
from OpenGL.GL import *
import numpy as np
import torch
import sys

torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)

from renderer.renderer_wrapper import RendererWrapper
from renderer.gaussian_renderer import GaussianRenderer
from renderer.editing_renderer import EditingRenderer
from renderer.fitting_renderer import FittingRenderer
from renderer.attach_renderer import AttachRenderer
from utils.gui_utils import imgui_window
from utils.gui_utils import imgui_utils
from utils.gui_utils import gl_utils
from utils.gui_utils import text_utils
from utils.gui_utils.constants import *
from utils.dict_utils import EasyDict
from widgets.common import (
    cam_widget,
    edit_widget,
    eval_widget,
    load_widget_pkl,
    load_widget_ply,
    performance_widget,
    render_widget,
    video_widget
)
from widgets.init import (
    processing_widget,
    style_widget,
    convert_widget,
    showcolmap_widget,
)
from widgets.load import (
    capture_widget,
    camvideo_widget,
)
from widgets.train import (
    latent_widget,
    training_widget,
)
from widgets.edit import (
    editcam_widget,
    editload_widget,
    editor_widget
)
from widgets.other import (
    fitting_widget
)

class HundredEditor(imgui_window.ImguiWindow):
    def __init__(self, args):
        data_path, mode, host, port = args.data_path, args.mode, args.host, args.port
        self.code_font_path = "resources/fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "resources/fonts/source_sans_pro/SourceSansPro-Regular.otf"

        super().__init__(
            title="100Editor",
            window_width=1920,
            window_height=1080,
            font=self.regular_font_path,
            code_font=self.code_font_path,
        )

        self.code_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self.regular_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self._imgui_renderer.refresh_font_texture()

        # Internals.
        self._last_error_print = None

        self.process_widgets = []
        self.load_widgets = []
        self.train_widgets = []
        self.edit_widgets = []
        
        self.init_widgets = [
            style_widget.StyleWidget(self),
            processing_widget.ProcessingWidget(self),
            convert_widget.ConverWidget(self),
            showcolmap_widget.ShowColmapWidget(self),
        ]

        self.load_widgets = [
                load_widget_ply.LoadWidget(self, data_path),
                cam_widget.CamWidget(self),
                performance_widget.PerformanceWidget(self),  
                capture_widget.CaptureWidget(self),
                render_widget.RenderWidget(self),
                edit_widget.EditWidget(self),
                eval_widget.EvalWidget(self),
                camvideo_widget.CamvideoWidget(self),
                video_widget.VideoWidget(self),
        ]
        self.train_widgets = [
                training_widget.TrainingWidget(self),
                cam_widget.CamWidget(self),
                performance_widget.PerformanceWidget(self),
                render_widget.RenderWidget(self),
        ]
        self.edit_widgets = [
                editload_widget.EditLoadWidget(self, data_path),
                editor_widget.EditorWidget(self),
                editcam_widget.EditcamWidget(self),
                performance_widget.PerformanceWidget(self), 
                render_widget.RenderWidget(self),
                eval_widget.EvalWidget(self),
        ]
        self.other_widgets = [
            fitting_widget.FittingWidget(self)
        ]
        # renderer = GaussianRenderer()
        # update_all_the_time = True

        renderer = {
                    "load":GaussianRenderer(),
                    "train":AttachRenderer(host=host, port=port),
                    "fitting":FittingRenderer(host="127.0.0.1", port=7090),
                    "editing":EditingRenderer(host="127.0.0.1", port=8084)
                   }
        
        update_all_the_time = {    
                               "load":False,
                               "train":True,
                               "fitting":True,
                               "editing":True
                              }
        
        self.renderer = RendererWrapper(renderer, update_all_the_time)
        self._tex_img = None
        self._tex_obj = None
        self.eval_result = ""

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()

        self.edit_image = None
        self.origin_image = None

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()

        # Splitter / pane layout state.
        self._pane_width = None
        self._pane_restore_width = None
        self._pane_collapsed = False
        self._pane_dragging = False
        self._pane_min_open_w = 360
        self._pane_min_render_w = 320
        self._pane_splitter_hit_w = 10
        self._pane_toggle_btn_w = 18
        self._pane_toggle_btn_h = 36
        self._last_renderer_type = None
        self._last_renderer_args = None
        self._suppress_viewport_mouse = False
        self._active_tab = "load"
        # Dynamic resolution scaling with viewport to avoid blur when resizing panes.
        self._auto_resolution_with_viewport = True
        self._max_dynamic_resolution = 2048

        self.skip_frame()

    def close(self):
        for widget in self.init_widgets:
            widget.close()
        for widget in self.load_widgets:
            widget.close()
        for widget in self.train_widgets:
            widget.close()
        for widget in self.edit_widgets:
            widget.close()
        self.renderer.close()
        super().close()

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print(f"\n{error}\n")
            self._last_error_print = error

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()

    def _default_pane_width(self):
        return max(self.content_width - self.content_height, 500)

    def _pane_constraints(self):
        max_open_w = max(0, self.content_width - self._pane_min_render_w)
        min_open_w = min(self._pane_min_open_w, max_open_w)
        return min_open_w, max_open_w

    def _set_sizes(self):
        min_open_w, max_open_w = self._pane_constraints()
        if self._pane_width is None:
            self._pane_width = self._default_pane_width()

        if not self._pane_collapsed:
            if max_open_w <= 0:
                self._pane_width = 0
            else:
                self._pane_width = float(np.clip(self._pane_width, min_open_w, max_open_w))

        self.pane_w = 0 if self._pane_collapsed else int(round(self._pane_width))
        self.button_w = self.font_size * 5
        self.button_large_w = self.font_size * 10
        self.label_w = round(self.font_size * 5.5) + 100
        self.label_w_large = round(self.font_size * 5.5) + 150

    def _toggle_pane_collapsed(self):
        if self._pane_collapsed:
            self._pane_collapsed = False
            if self._pane_width is None or self._pane_width <= 0:
                self._pane_width = (
                    self._pane_restore_width if self._pane_restore_width is not None else self._default_pane_width()
                )
        else:
            if self.pane_w > 0:
                self._pane_restore_width = float(self.pane_w)
            self._pane_collapsed = True
            self._pane_dragging = False

    def _pane_toggle_button_rect(self):
        center_x = float(self.pane_w + 8)
        center_y = float(self.content_height) * 0.5
        half_w = float(self._pane_toggle_btn_w) * 0.5
        half_h = float(self._pane_toggle_btn_h) * 0.5
        x0 = max(0.0, center_x - half_w)
        x1 = min(float(self.content_width), center_x + half_w)
        y0 = max(0.0, center_y - half_h)
        y1 = min(float(self.content_height), center_y + half_h)
        return x0, y0, x1, y1

    @staticmethod
    def _point_in_rect(x, y, rect):
        x0, y0, x1, y1 = rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def _handle_pane_splitter(self):
        splitter_half = self._pane_splitter_hit_w * 0.5
        splitter_x = float(self.pane_w)
        x0 = max(0.0, splitter_x - splitter_half)
        x1 = min(float(self.content_width), splitter_x + splitter_half)
        mouse = imgui.get_mouse_pos()
        hovered = (x0 <= mouse.x <= x1) and (0.0 <= mouse.y <= float(self.content_height))
        button_rect = self._pane_toggle_button_rect()
        button_hovered = self._point_in_rect(mouse.x, mouse.y, button_rect)

        # Prevent viewport mouse interactions (e.g., camera drag) while interacting with splitter/button.
        if self._pane_dragging:
            self._suppress_viewport_mouse = True
        if button_hovered and (imgui.is_mouse_clicked(0) or imgui.is_mouse_down(0)):
            self._suppress_viewport_mouse = True
        if hovered and (imgui.is_mouse_clicked(0) or imgui.is_mouse_down(0)):
            self._suppress_viewport_mouse = True

        if button_hovered:
            imgui.set_mouse_cursor(MOUSE_CURSOR_HAND)
        elif hovered or self._pane_dragging:
            imgui.set_mouse_cursor(MOUSE_CURSOR_RESIZE_EW)

        if button_hovered and imgui.is_mouse_clicked(0):
            self._toggle_pane_collapsed()
            return

        if hovered and imgui.is_mouse_clicked(0):
            self._pane_dragging = True
            if self._pane_collapsed:
                self._pane_collapsed = False
                if self._pane_restore_width is not None:
                    self._pane_width = self._pane_restore_width
                elif self._pane_width is None:
                    self._pane_width = self._default_pane_width()

        if not self._pane_dragging:
            return

        if not imgui.is_mouse_down(0):
            self._pane_dragging = False
            return

        min_open_w, max_open_w = self._pane_constraints()
        target_w = float(mouse.x)
        collapse_snap_w = max(4.0, self._pane_splitter_hit_w)
        if target_w <= collapse_snap_w:
            if self._pane_width and self._pane_width > 0:
                self._pane_restore_width = self._pane_width
            self._pane_collapsed = True
            return

        self._pane_collapsed = False
        if max_open_w <= 0:
            self._pane_width = 0
            return

        clamped_w = float(np.clip(target_w, min_open_w, max_open_w))
        self._pane_width = clamped_w
        self._pane_restore_width = clamped_w

    def _draw_pane_splitter(self):
        splitter_x = float(self.pane_w)
        line_half_w = 1.0
        x0 = max(0.0, splitter_x - line_half_w)
        x1 = min(float(self.content_width), splitter_x + line_half_w)
        if x1 <= x0:
            return

        splitter_half = self._pane_splitter_hit_w * 0.5
        mouse = imgui.get_mouse_pos()
        hovered = (
            max(0.0, splitter_x - splitter_half) <= mouse.x <= min(float(self.content_width), splitter_x + splitter_half)
            and 0.0 <= mouse.y <= float(self.content_height)
        )
        if self._pane_dragging:
            color, alpha = [0.40, 0.44, 0.47], 1.0
        elif hovered:
            color, alpha = [0.44, 0.44, 0.44], 0.85
        else:
            color, alpha = [0.28, 0.28, 0.28], 0.55
        gl_utils.draw_rect(pos=(x0, 0), pos2=(x1, self.content_height), color=color, alpha=alpha, rounding=0)

        btn_x0, btn_y0, btn_x1, btn_y1 = self._pane_toggle_button_rect()
        if btn_x1 <= btn_x0 or btn_y1 <= btn_y0:
            return
        mouse = imgui.get_mouse_pos()
        btn_hovered = self._point_in_rect(mouse.x, mouse.y, (btn_x0, btn_y0, btn_x1, btn_y1))
        if btn_hovered:
            btn_color, btn_alpha = [0.25, 0.28, 0.32], 0.95
        else:
            btn_color, btn_alpha = [0.20, 0.22, 0.25], 0.88
        gl_utils.draw_rect(
            pos=(btn_x0, btn_y0),
            pos2=(btn_x1, btn_y1),
            color=btn_color,
            alpha=btn_alpha,
            rounding=(6, 6),
        )

        # Left arrow when open (collapse), right arrow when collapsed (expand).
        if self._pane_collapsed:
            arrow = np.array([[0.36, 0.26], [0.36, 0.74], [0.72, 0.50]], dtype="float32")
        else:
            arrow = np.array([[0.64, 0.26], [0.64, 0.74], [0.28, 0.50]], dtype="float32")
        arrow_margin_x = 4.0
        arrow_margin_y = 8.0
        arrow_w = max(1.0, (btn_x1 - btn_x0) - 2 * arrow_margin_x)
        arrow_h = max(1.0, (btn_y1 - btn_y0) - 2 * arrow_margin_y)
        gl_utils.draw_shape(
            arrow,
            mode=GL_TRIANGLE_FAN,
            pos=(btn_x0 + arrow_margin_x, btn_y0 + arrow_margin_y),
            size=(arrow_w, arrow_h),
            color=[0.88, 0.90, 0.94],
            alpha=0.98,
        )

    def _apply_viewport_resolution(self, render_args):
        base_resolution = render_args.get("resolution", None)
        if base_resolution is None:
            return render_args

        viewport_w = max(1, int(self.content_width - self.pane_w))
        viewport_h = max(1, int(self.content_height))
        try:
            base_resolution = int(base_resolution)
        except Exception:
            base_resolution = max(viewport_w, viewport_h)
        if base_resolution <= 0:
            base_resolution = max(viewport_w, viewport_h)

        if self._auto_resolution_with_viewport:
            target_long = max(base_resolution, max(viewport_w, viewport_h))
            target_long = min(target_long, int(self._max_dynamic_resolution))
        else:
            target_long = base_resolution

        if viewport_w >= viewport_h:
            resolution_x = target_long
            resolution_y = max(1, int(round(target_long * viewport_h / max(viewport_w, 1))))
        else:
            resolution_y = target_long
            resolution_x = max(1, int(round(target_long * viewport_w / max(viewport_h, 1))))

        render_args["resolution_x"] = int(resolution_x)
        render_args["resolution_y"] = int(resolution_y)
        return render_args

    def _update_renderer(self, render_type):
        render_args = dict(self.args)
        render_args = self._apply_viewport_resolution(render_args)

        self.renderer.set_args(type=render_type, **render_args)
        self._last_renderer_type = render_type
        self._last_renderer_args = dict(render_args)
        result = self.renderer.result
        if result is not None:
            self.result = result

    def _run_collapsed_camera_input(self):
        # Keep camera interaction alive when the control pane is collapsed.
        active = getattr(self, "_active_tab", "load")
        if active == "edit":
            self.args.painting = bool(getattr(self.args, "painting", False))
            for widget in self.edit_widgets:
                if isinstance(widget, editcam_widget.EditcamWidget):
                    widget(False)
                    return
        elif active == "train":
            for widget in self.train_widgets:
                if isinstance(widget, cam_widget.CamWidget):
                    widget(False)
                    return
        elif active == "load":
            for widget in self.load_widgets:
                if isinstance(widget, cam_widget.CamWidget):
                    widget(False)
                    return

    def _update_renderer_with_cached_args(self):
        if self._last_renderer_type is None or self._last_renderer_args is None:
            return
        render_args = dict(self._last_renderer_args)
        if len(self.args) > 0:
            render_args.update(dict(self.args))
        render_args = self._apply_viewport_resolution(render_args)

        self.renderer.set_args(type=self._last_renderer_type, **render_args)
        result = self.renderer.result
        if result is not None:
            self.result = result

    def draw_frame(self):
        self.begin_frame()
        self.args = EasyDict()
        self._suppress_viewport_mouse = False
        self._set_sizes()
        self._handle_pane_splitter()
        self._set_sizes()

        # Control pane
        pane_window_w = self.pane_w
        pane_window_x = 0
        if self._pane_collapsed:
            hidden_w = (
                self._pane_restore_width
                if self._pane_restore_width is not None
                else (self._pane_width if self._pane_width is not None else self._default_pane_width())
            )
            pane_window_w = max(1, int(round(hidden_w)))
            pane_window_x = -pane_window_w

        imgui.set_next_window_pos(imgui.ImVec2(pane_window_x, 0))
        imgui.set_next_window_size(imgui.ImVec2(max(1, pane_window_w), self.content_height))
        control_pane_flags = WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE
        imgui.begin("##control_pane", p_open=True, flags=control_pane_flags)
        
        if self._pane_collapsed:
            self._run_collapsed_camera_input()
            self._update_renderer_with_cached_args()
        elif imgui.begin_tab_bar("MyTabBar"):
            if imgui.begin_tab_item("init")[0]:
                self._active_tab = "init"
                for widget in self.init_widgets:
                    expanded, _visible = imgui_utils.collapsing_header(widget.name, default=False)
                    imgui.indent()
                    widget(expanded)
                    imgui.unindent()
                imgui.end_tab_item()
            
            if imgui.begin_tab_item("load")[0]:
                self._active_tab = "load"
                for widget in self.load_widgets:
                    expanded, _visible = imgui_utils.collapsing_header(widget.name, default=widget.name == "Load")
                    imgui.indent()
                    widget(expanded)
                    imgui.unindent()
                imgui.end_tab_item()
                
                # Render
                if self.is_skipping_frames():
                    pass
                else:
                    self._update_renderer("load")
                
            if imgui.begin_tab_item("train")[0]:
                self._active_tab = "train"
                # Widgets
                for widget in self.train_widgets:
                    expanded, _visible = imgui_utils.collapsing_header(
                        widget.name,
                        default=widget.name == "Training",
                    )
                    imgui.indent()
                    widget(expanded)
                    imgui.unindent()
                imgui.end_tab_item()

                # Render
                if self.is_skipping_frames():
                    pass
                else:
                    self._update_renderer("train")

            if imgui.begin_tab_item("edit")[0]:
                self._active_tab = "edit"
                for widget in self.edit_widgets:
                    expanded, _visible = imgui_utils.collapsing_header(widget.name, default=(widget.name == "Load" or widget.name == "Editor"))
                    imgui.indent()
                    widget(expanded)
                    imgui.unindent()
                imgui.end_tab_item()

                # Render
                if self.is_skipping_frames():
                    pass
                else:
                    if not self.args.edit3D:
                        self._update_renderer("load")
                    else:
                        self._update_renderer("editing")

                    if self.args.edit_single:
                        self.result.image = self.args.single_image
            
            if imgui.begin_tab_item("other")[0]:
                self._active_tab = "other"
                for widget in self.other_widgets:
                    expanded, _visible = imgui_utils.collapsing_header(widget.name, default=widget.name == "Fitting")
                    imgui.indent()
                    widget(expanded)
                    imgui.unindent()
                imgui.end_tab_item()

                # Render
                if self.is_skipping_frames():
                    pass
                else:
                    self._update_renderer("fitting")

            imgui.end_tab_bar()

        # Display
        max_w = self.content_width - self.pane_w
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        # Fill render area with a neutral background so letterboxing is not pure black.
        gl_utils.draw_rect(
            pos=(self.pane_w, 0),
            pos2=(self.content_width, self.content_height),
            color=[0.14, 0.15, 0.17],
            alpha=1.0,
            rounding=0,
        )
        if "image" in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else: 
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
            if hasattr(self.args, 'current_color'):
                gl_utils.sketch(self.content_width, self.content_height, self.args.points, self.args.current_color, self.args.line_width)
            if hasattr(self.args, 'rec_start') and hasattr(self.args, 'rec_end') and self.args.rec_start and self.args.rec_end:
               if self.args.rec_start[0] >= self.pane_w and self.args.rec_end[0] >= self.pane_w \
                   and self.args.rec_start[1] >= 0 and self.args.rec_end[1] >= 0:
                   gl_utils.draw_rect(pos=self.args.rec_start, pos2=self.args.rec_end, color=[0,0,0], alpha=1, rounding=0)
            if hasattr(self.args, "click_ripples"):
                gl_utils.draw_click_ripples(self.content_width, self.content_height, self.args.click_ripples)
            # save the image
            if hasattr(self.args, 'edit_image') and self.args.edit_image:
                self.edit_image = gl_utils.get_image(self.pane_w, 0, int(max_w/zoom), int(max_h/zoom))
            if hasattr(self.args, 'draw_image') and not self.args.draw_image:
                self.origin_image = gl_utils.get_image(self.pane_w, 0, int(max_w/zoom), int(max_h/zoom))
        if "error" in self.result:
            self.print_error(self.result.error)
            if "message" not in self.result:
                self.result.message = str(self.result.error)
        if "message" in self.result:
            tex = text_utils.get_texture(
                self.result.message,
                size=self.font_size,
                max_width=max_w,
                max_height=max_h,
                outline=2,
            )
            tex.draw(pos=pos, align=0.5, rint=True, color=1)
        if "eval" in self.result:
            self.eval_result = self.result.eval
        else:
            self.eval_result = None
        self._draw_pane_splitter()

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()
