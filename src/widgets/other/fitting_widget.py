from pathlib import Path
import os
import subprocess
import sys
from imgui_bundle import imgui, ImVec2, portable_file_dialogs as pfd
from utils.gui_utils import imgui_utils
from imgui_bundle import implot
from utils.gui_utils.easy_imgui import label
from utils.path_utils import find_repo_root
from widgets.widget import Widget
from PIL import Image
from utils.dict_utils import EasyDict
import numpy as np



class FittingWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Fitting")
        self.repo_root = find_repo_root(__file__)
        self.fitting_train_script = self.repo_root / "src" / "trainer" / "fitting" / "train.py"
        self.image_path = "."
        self.num_points = 2000
        # self.save_image = True
        self.max_step = 1000
        self.alpha = 0.99
        self.iterations = []
        self.last_iteration = None
        self.stop_at_value = -1
        self.model_types = ["3dgs","2dgs"]
        self.model_item = 0
        self.stop_training = True
        self.stop_from_renderer = False
        self.fitting_trainer = None
        self.img_h = 512
        self.img_w = 512
        self.plots = EasyDict(
            loss=dict(values=[], dtype=float),
        )

    def _reset_training_history(self):
        self.iterations.clear()
        self.last_iteration = None
        for plot in self.plots.values():
            if "values" in plot:
                plot["values"].clear()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        fitting_path = []

        # Keep UI state in sync with the subprocess lifecycle.
        if self.fitting_trainer is not None and self.fitting_trainer.poll() is not None:
            self.stop_training = True
            self.stop_from_renderer = False
            self.fitting_trainer = None

        if show:
            if imgui_utils.button(f"Image Path", width=viz.button_w):
                image_path = self._select_image()
                if image_path:
                    self.image_path = image_path
                    
            imgui.same_line()
            imgui.text(f"Selected Image Path: {self.image_path}")
            if self.is_image_file(self.image_path):
                fitting_path.append(self.image_path)
                image = Image.open(self.image_path)
                self.img_w, self.img_h = image.size

            imgui.push_item_width(200)
            label("Number of Points", viz.label_w)
            _, self.num_points = imgui.input_int("##number of points", v=self.num_points)
            label("Iterations", viz.label_w)
            _, self.max_step = imgui.input_int("##iterations", v=self.max_step)
            # label("Save", viz.label_w)
            # _, self.save_image = imgui.checkbox("##save", self.save_image)
            label("Model Type", viz.label_w)
            if imgui.radio_button("3DGS", self.model_item == 0):
                self.model_item = 0
            imgui.same_line()
            if imgui.radio_button("2DGS", self.model_item == 1):
                self.model_item = 1
            label("Ema alpha", viz.label_w)
            _change, self.alpha = imgui.slider_float(label="##Ema alpha", v=self.alpha,v_min=0.00,v_max=1.00,format="%.2f")
            if self.stop_training or self.stop_from_renderer:
                if imgui.button("Start Training", ImVec2(viz.label_w_large, 0)):
                    self.stop_training = False
                    self._reset_training_history()
                    env = os.environ.copy()
                    src_path = str(self.repo_root / "src")
                    if env.get("PYTHONPATH"):
                        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
                    else:
                        env["PYTHONPATH"] = src_path
                    # if self.save_image:
                    #     self.fitting_trainer =subprocess.Popen([
                    #         "python",
                    #         "trainer/fitting/train.py",
                    #         "--num_points", str(self.num_points),
                    #         "--save_imgs",
                    #         "--img_path",self.image_path,
                    #         "--model-type", self.model_types[self.model_item]
                    #     ])
                    # else:
                    self.fitting_trainer =subprocess.Popen([
                        sys.executable,
                        str(self.fitting_train_script),
                        "--num_points", str(self.num_points),
                        "--img_path",self.image_path,
                        "--model-type", self.model_types[self.model_item],
                        "--iterations", str(self.max_step),
                        "--alpha", str(self.alpha)
                    ], cwd=str(self.repo_root), env=env)
                    if self.stop_from_renderer:
                        self.stop_at_value = -1
            else:
                if imgui.button("Pause Training", ImVec2(viz.label_w_large, 0)):
                    if self.fitting_trainer is not None:
                        self.fitting_trainer.terminate()
                        self.fitting_trainer.wait()
                        self.fitting_trainer = None
                    self.stop_training = True
        
        viz.args.do_training = not self.stop_training
        viz.args.img_size = (self.img_h, self.img_w)
                

        viz.args.fitting_path = fitting_path

        if "training_stats" in viz.result.keys():
            stats = viz.result["training_stats"]
        else:
            if show:
                imgui.text("No training stats send by the renderer.")
            return

        current_iteration = int(stats["iteration"])
        if self.last_iteration is not None and current_iteration < self.last_iteration:
            self._reset_training_history()
        self.last_iteration = current_iteration

        self.iterations.append(current_iteration)
        self.plots.loss["values"].append(float(stats["loss"]))
        self.stop_from_renderer = stats["paused"]
        if show:
            if self.stop_training or self.stop_from_renderer:
                if imgui.button("Single Training Step", ImVec2(viz.label_w_large, 0)):
                    viz.args.single_training_step = True
                    self.stop_training = True

            label(f"Current Iteration", viz.label_w_large)
            imgui.text(str(stats['iteration']))
            label("Pause Training at", viz.label_w_large)
            _, self.stop_at_value = imgui.input_int("##stop_at", self.stop_at_value)

            for plot_name, plot_values in self.plots.items():
                plot_size = imgui.ImVec2(viz.pane_w - 150, 200)
                implot.set_next_axes_to_fit()
                if implot.begin_plot(plot_name, plot_size):
                    implot.plot_line( 
                        plot_name,
                        ys=np.array(plot_values["values"], dtype=plot_values["dtype"]),
                        xs=np.array(self.iterations, dtype=plot_values["dtype"]),
                    )
                    implot.end_plot()

        viz.args.stop_at_value = self.stop_at_value   
    
    def _select_image(self):
        dialog = pfd.open_file("Select Image File")
        file_path = dialog.result() if hasattr(dialog, "result") else dialog
        if isinstance(file_path, (list, tuple)):
            file_path = file_path[0] if file_path else ""
        return file_path if file_path else None
    
    def is_image_file(self,image_path):
        valid_extensions = ['.jpg', '.jpeg', '.png']
        ext = Path(image_path).suffix
        return ext.lower() in valid_extensions
