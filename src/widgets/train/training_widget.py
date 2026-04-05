from argparse import ArgumentParser
import threading
from imgui_bundle import imgui, ImVec2, portable_file_dialogs as pfd
from imgui_bundle import implot
import numpy as np
import os
import time
import multiprocessing

# from arguments import ModelParams, OptimizationParams, PipelineParams
# from gaussian_renderer import network_gui
from utils.gui_utils import imgui_utils
import torch
from third_party.gaussiansplatting.utils.general_utils import safe_state
from utils.gui_utils.easy_imgui import label
from utils.command_utils import *
import subprocess
from utils.dict_utils import EasyDict
from utils.path_utils import resolve_runtime_subdir
from widgets.widget import Widget
import datetime

class TrainingWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Training")
        self.text = "gaussian"
        self.hist_cache = dict()
        self.use_cache_dict = dict()
        self.iterations = []
        self.plots = EasyDict(
            num_gaussians=dict(values=[], dtype=int),
            loss=dict(values=[], dtype=float),
            sh_degree=dict(values=[], dtype=int),
        )
        self.last_iteration = None
        self.stop_at_value = -1
        self.stop_training = True
        self.stop_from_renderer = False
        self.training_path = os.getcwd()
        self.origin_output_root = resolve_runtime_subdir(
            __file__, "experiments", "train", "origin", create=True
        )
        self.gsplat_output_root = resolve_runtime_subdir(
            __file__, "experiments", "train", "gsplat", create=True
        )
        self.output_dir = str(self.origin_output_root)
        self.use_gpu = 0
        self.gpu_items = [str(i) for i in range(torch.cuda.device_count())]
        self.selected_option = 0
        self.gsplat_training_dir = os.getcwd()
        self.gsplat_output_dir = str(self.gsplat_output_root)
        self.gsplat_mode = 0
        self.MODE = ["default","mcmc"]
        self.gsplat_use_bilateral_grid = False
        self.gsplat_use_taming_3dgs = False
        self.gsplat_use_depth_loss = False
        self.origin_use_depth_loss = False
        self.origin_use_appearance_embedding = False
        self.origin_use_random_background = False
        self.origin_trainer = None
        self.gsplat_trainer = None
        self.alpha = 0.99

    def _reset_training_history(self):
        self.iterations.clear()
        self.last_iteration = None
        for plot in self.plots.values():
            if "values" in plot:
                plot["values"].clear()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text("Choose your model")
            if imgui.radio_button("Origin", self.selected_option == 0):
                self.selected_option = 0 
            imgui.same_line(viz.label_w)
            if imgui.radio_button("gsplat", self.selected_option == 1):
                self.selected_option = 1 
            # imgui.text("Training Parameter")
            if self.selected_option == 0:
                if imgui_utils.button("Choose", width=viz.button_w):
                    trainning_folder = self._select_folder()
                    if trainning_folder:
                        self.training_path = trainning_folder

                imgui.same_line()
                imgui.text(f"Training Path: {self.training_path}")

                if imgui_utils.button("Output", width=viz.button_w):
                    output_dir = self._select_folder()
                    target_name = os.path.basename(self.training_path)
                    if output_dir:
                        self.output_dir = output_dir
                imgui.same_line()
                imgui.text(f"Output Path: {self.output_dir}")
                
                imgui.set_next_item_width(viz.button_w)
                changed, self.use_gpu = imgui.combo(
                    "choose gpu",
                    self.use_gpu,
                    self.gpu_items
                )
                label("Ema alpha", viz.label_w)
                _change, self.alpha = imgui.slider_float(label="##Ema alpha", v=self.alpha,v_min=0.00,v_max=1.00,format="%.2f")
                label("Depth Loss", viz.label_w)
                _changed, self.origin_use_depth_loss = imgui.checkbox(
                    "##origin_use_depth_loss", self.origin_use_depth_loss
                )
                label("Appearance Embedding", viz.label_w)
                _changed, self.origin_use_appearance_embedding = imgui.checkbox(
                    "##origin_use_appearance_embedding", self.origin_use_appearance_embedding
                )
                label("Random Background", viz.label_w)
                _changed, self.origin_use_random_background = imgui.checkbox(
                    "##origin_use_random_background", self.origin_use_random_background
                )
                if self.stop_training or self.stop_from_renderer:
                    if imgui.button("Start Training", ImVec2(viz.label_w_large, 0)):
                        self.stop_training = False
                        self._reset_training_history()
                        # self.origin_trainer =subprocess.Popen([
                        #     "python",
                        #     "trainer/origin/train.py",
                        #     "-s", self.training_path,
                        #     "-m", self.output_dir,
                        #     "--gpu", self.gpu_items[self.use_gpu],
                        #     "--alpha", str(self.alpha)
                        # ])
                        now = datetime.datetime.now()
                        now = now.strftime("%Y_%m_%d_%H_%M_%S")
                        self.origin_trainer = training_3DGS(
                            self.training_path,
                            os.path.join(self.output_dir, now),
                            self.gpu_items[self.use_gpu],
                            self.alpha,
                            quiet=False,
                            detect_anomaly=False,
                            use_depth_loss=self.origin_use_depth_loss,
                            use_appearance_embedding=self.origin_use_appearance_embedding,
                            use_random_background=self.origin_use_random_background,
                        )
                        if self.stop_from_renderer:
                            self.stop_at_value = -1
                else:
                    if imgui.button("Pause Training", ImVec2(viz.label_w_large, 0)):
                        self.origin_trainer.terminate()
                        self.origin_trainer.wait()
                        self.stop_training = True
            elif self.selected_option == 1:
                imgui.set_next_item_width(viz.button_w)
                change, self.gsplat_mode = imgui.combo(
                                            "Mode",
                                            self.gsplat_mode,
                                            self.MODE
                                            )
                label("Bilateral Grid", viz.label_w)
                _changed, self.gsplat_use_bilateral_grid = imgui.checkbox(
                    "##gsplat_use_bilateral_grid", self.gsplat_use_bilateral_grid
                )
                label("Taming-3DGS", viz.label_w)
                _changed, self.gsplat_use_taming_3dgs = imgui.checkbox(
                    "##gsplat_use_taming_3dgs", self.gsplat_use_taming_3dgs
                )
                label("Depth Loss", viz.label_w)
                _changed, self.gsplat_use_depth_loss = imgui.checkbox(
                    "##gsplat_use_depth_loss", self.gsplat_use_depth_loss
                )
                if imgui_utils.button("Choose", width=viz.button_w):
                    gsplat_trainning_folder = self._select_folder()
                    if gsplat_trainning_folder:
                        self.gsplat_training_dir = gsplat_trainning_folder
                imgui.same_line()
                imgui.text(f"Training Path: {self.gsplat_training_dir}")
                
                if imgui_utils.button("Output", width=viz.button_w):
                    gsplat_output_dir = self._select_folder()
                    target_name = os.path.basename(self.gsplat_training_dir)
                    if gsplat_output_dir:
                        self.gsplat_output_dir = gsplat_output_dir
                imgui.same_line()
                imgui.text(f"Output Path: {self.gsplat_output_dir}")
                label("Ema alpha", viz.label_w)
                _change, self.alpha = imgui.slider_float(label="##Ema alpha", v=self.alpha,v_min=0.00,v_max=1.00,format="%.2f")
                if self.stop_training or self.stop_from_renderer:
                    if imgui.button("Start Training", ImVec2(viz.label_w_large, 0)):
                        self.stop_training = False
                        self._reset_training_history()
                        # self.gsplat_trainer = subprocess.Popen([
                        #     "python", 
                        #     "trainer/gsplat/train.py", 
                        #     self.MODE[self.gsplat_mode],
                        #     "--data_dir", self.gsplat_training_dir, 
                        #     "--data_factor", "1",
                        #     "--result_dir",self.gsplat_output_dir,
                        #     "--alpha", str(self.alpha)
                        # ])
                        now = datetime.datetime.now()
                        now = now.strftime("%Y_%m_%d_%H_%M_%S")
                        self.gsplat_trainer = training_gsplat_3DGS(
                            self.MODE[self.gsplat_mode],
                            self.gsplat_training_dir,
                            os.path.join(self.gsplat_output_dir, now),
                            self.alpha,
                            use_bilateral_grid=self.gsplat_use_bilateral_grid,
                            use_taming_3dgs=self.gsplat_use_taming_3dgs,
                            use_depth_loss=self.gsplat_use_depth_loss,
                        )
                        if self.stop_from_renderer:
                            self.stop_at_value = -1
                else:
                    if imgui.button("Pause Training", ImVec2(viz.label_w_large, 0)):
                        self.gsplat_trainer.terminate()
                        self.gsplat_trainer.wait()
                        self.stop_training = True

        viz.args.do_training = not self.stop_training

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
        self.plots.num_gaussians["values"].append(stats["num_gaussians"])
        self.plots.loss["values"].append(stats["loss"])
        self.plots.sh_degree["values"].append(stats["sh_degree"])
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

            imgui.text("Training Params:")
            with imgui_utils.indent():
                for param, value in stats["train_params"].items():
                    label(str(param), viz.label_w_large)
                    label(str(value), viz.label_w_large)
                    imgui.new_line()

        viz.args.stop_at_value = self.stop_at_value
    
    def _select_folder(self):
        dialog = pfd.select_folder("Select Folder")
        folder_path = dialog.result() if hasattr(dialog, "result") else dialog
        if isinstance(folder_path, (list, tuple)):
            folder_path = folder_path[0] if folder_path else ""
        return folder_path if folder_path else None
    
    def close(self):
        if self.gsplat_trainer != None:
            self.gsplat_trainer.terminate()
            self.gsplat_trainer.wait()
        if self.origin_trainer != None:
            self.origin_trainer.terminate()
            self.origin_trainer.wait()
        super().close()
    
