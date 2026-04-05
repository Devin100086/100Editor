import re
import traceback
import PIL
from imgui_bundle import imgui
import numpy as np

from utils.gui_utils.easy_imgui import label
from utils.gui_utils import imgui_utils
from utils.path_utils import resolve_runtime_subdir
from widgets.widget import Widget


class CaptureWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Save")
        self.path = resolve_runtime_subdir(__file__, "renders", create=True)
        self.path_ply = resolve_runtime_subdir(__file__, "exports", create=True)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            label("Save Screenshot", viz.label_w)
            if imgui_utils.button("Save img", width=viz.button_w):
                if "image" in viz.result:
                    self.save_png(viz.result.image)
                else:
                    viz.result.message = "No image to save in current view."

            label("Save PLY", viz.label_w)
            if imgui_utils.button("Save ply", width=viz.button_w):
                viz.args.save_ply_path = self.path_ply.as_posix()
            else:
                viz.args.save_ply_path = None

    def save_png(self, image):
        viz = self.viz
        try:
            image = np.asarray(image)
            if image.ndim == 2:
                image = image[:, :, None]
            if image.ndim != 3:
                raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}")

            _height, _width, channels = image.shape
            if channels not in [1, 3, 4]:
                raise ValueError(f"Unsupported channel count: {channels}")

            if image.dtype != np.uint8:
                image = image.astype(np.float32)
                if image.max() <= 1.0:
                    image = image * 255.0
                image = np.clip(image, 0, 255).astype(np.uint8)

            self.path.mkdir(parents=True, exist_ok=True)
            file_id = 0
            for entry in self.path.iterdir():
                if entry.is_file():
                    match = re.fullmatch(r"(\d+).*", entry.name)
                    if match:
                        file_id = max(file_id, int(match.group(1)) + 1)

            save_path = self.path / f"{file_id:05d}.png"
            if channels == 1:
                pil_image = PIL.Image.fromarray(image[:, :, 0], "L")
            elif channels == 4:
                pil_image = PIL.Image.fromarray(image, "RGBA")
            else:
                pil_image = PIL.Image.fromarray(image, "RGB")
            pil_image.save(save_path.as_posix())
            print(f"Image saved in {save_path.as_posix()}.")
            return save_path.as_posix()
        except Exception as e:
            viz.result.error = "".join(traceback.format_exception(e)) + str(e)
            return None
