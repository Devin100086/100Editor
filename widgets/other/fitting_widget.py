from pathlib import Path
import subprocess
from imgui_bundle import imgui, ImVec2
from lumina3D_utils.gui_utils import imgui_utils
from imgui_bundle import implot
from lumina3D_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from lumina3D_utils.dict_utils import EasyDict
import numpy as np



class FittingWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Fitting")
        self.image_path = "."
        self.num_points = 2000
        # self.save_image = True
        self.max_step = 1000
        self.alpha = 0.99
        self.iterations = []
        self.stop_at_value = -1
        self.model_types = ["3dgs","2dgs"]
        self.model_item = 0
        self.stop_training = True
        self.stop_from_renderer = False
        self.wh = 512
        self.plots = EasyDict(
            loss=dict(values=[], dtype=float),
        )

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        fitting_path = []
        if show:
            imgui.text("Fitting a single image")
            if imgui_utils.button(f"Image Path", width=viz.button_w):
                image_path = self._select_image()
                self.image_path = self.image_path if isinstance(image_path, tuple) else image_path
                    
            imgui.same_line()
            imgui.text(f"Selected Image Path: {self.image_path}")
            if self.is_image_file(self.image_path):
                fitting_path.append(self.image_path)
                image = Image.open(self.image_path)
                self.wh = image.size[0]

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
                        "python",
                        "trainer/fitting/train.py",
                        "--num_points", str(self.num_points),
                        "--img_path",self.image_path,
                        "--model-type", self.model_types[self.model_item],
                        "--iterations", str(self.max_step),
                        "--alpha", str(self.alpha)
                    ])
                    if self.stop_from_renderer:
                        self.stop_at_value = -1
            else:
                if imgui.button("Pause Training", ImVec2(viz.label_w_large, 0)):
                    self.fitting_trainer.terminate()
                    self.fitting_trainer.wait()
                    self.stop_training = True
        
        viz.args.do_training = not self.stop_training
        viz.args.img_size = self.wh
                

        viz.args.fitting_path = fitting_path

        if "training_stats" in viz.result.keys():
            stats = viz.result["training_stats"]
        else:
            if show:
                imgui.text("No training stats send by the renderer.")
            return

        self.iterations.append(stats["iteration"])
        self.plots.loss["values"].append(stats["loss"])
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
        root = tk.Tk()
        root.withdraw()
        file_types = [
            ("png FIles", "*.png"),
            ("jpg Files", "*.jpg"),
            ("jpg Files", "*.jpeg"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        return file_path
    
    def is_image_file(self,image_path):
        valid_extensions = ['.jpg', '.jpeg', '.png']
        ext = Path(image_path).suffix
        return ext.lower() in valid_extensions
