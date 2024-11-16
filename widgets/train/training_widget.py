from argparse import ArgumentParser
import threading
from imgui_bundle import imgui, ImVec2
from imgui_bundle import implot
import numpy as np
import os
import time
import multiprocessing

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui
from lumina3D_utils.gui_utils import imgui_utils
import tkinter as tk
import torch
from gaussiansplatting.utils.general_utils import safe_state
from lumina3D_utils.gui_utils.easy_imgui import label
import subprocess
from lumina3D_utils.dict_utils import EasyDict
from widgets.widget import Widget
from tkinter import filedialog


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
        self.stop_at_value = -1
        self.stop_training = True
        self.stop_from_renderer = False
        self.training_path = os.getcwd()
        self.use_gpu = 0
        self.gpu_items = [str(i) for i in range(torch.cuda.device_count())]
        self.quiet = False
        self.detect_anomaly = False
        self.selected_option = 0
        self.gsplat_training_dir = os.getcwd()
        self.gsplat_output_dir = os.getcwd()
        self.gsplat_mode = 0
        self.MODE = ["default","mcmc"]
        self.origin_trainer = None
        self.gsplat_trainer = None


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
                    self.training_path = self.training_path if isinstance(trainning_folder, tuple) else trainning_folder

                imgui.same_line()
                imgui.text(f"Training Path: {self.training_path}")
                
                imgui.set_next_item_width(viz.button_w)
                changed, self.use_gpu = imgui.combo(
                    "choose gpu",
                    self.use_gpu,
                    self.gpu_items
                )

                clicked, self.quiet = imgui.checkbox("Quiet",self.quiet)

                clicked, self.detect_anomaly = imgui.checkbox("Detect Anomaly",self.detect_anomaly)

                imgui.new_line()
                if self.stop_training or self.stop_from_renderer:
                    if imgui.button("Start Training", ImVec2(viz.label_w_large, 0)):
                        self.stop_training = False
                        self.origin_trainer =subprocess.Popen([
                            "python",
                            "trainer/origin/train.py",
                            "-s", self.training_path,
                            "--gpu", self.gpu_items[self.use_gpu]
                        ])
                        if self.stop_from_renderer:
                            self.stop_at_value = -1
                else:
                    if imgui.button("Pause Training", ImVec2(viz.label_w_large, 0)):
                        self.origin_trainer.terminate()
                        self.origin_trainer.wait()
                        self.stop_training = True
            elif self.selected_option == 1:
                imgui.set_next_item_width(viz.button_w)
                changedm, self.gsplat_mode = imgui.combo(
                                            "Mode",
                                            self.gsplat_mode,
                                            self.MODE
                                            )
                if imgui_utils.button("Data dir", width=viz.button_w):
                    gsplat_trainning_folder = self._select_folder()
                    self.gsplat_training_dir = self.gsplat_training_dir if isinstance(gsplat_trainning_folder, tuple) else gsplat_trainning_folder
                    target_name = os.path.basename(self.gsplat_training_dir)
                    self.gsplat_output_dir = os.path.join(self.gsplat_output_dir,"results",target_name)
                imgui.same_line()
                imgui.text(f"Training Path: {self.gsplat_training_dir}")
                
                if imgui_utils.button("Output dir", width=viz.button_w):
                    gsplat_output_dir = self._select_folder() 
                    target_name = os.path.basename(self.gsplat_training_dir)
                    self.gsplat_output_dir = self.gsplat_training_dir if isinstance(gsplat_output_dir, tuple) else os.path.join(gsplat_output_dir,"results",target_name)
                imgui.same_line()
                imgui.text(f"Output Path: {self.gsplat_output_dir}")
                if self.stop_training or self.stop_from_renderer:
                    if imgui.button("Start Training", ImVec2(viz.label_w_large, 0)):
                        self.stop_training = False
                        self.gsplat_trainer = subprocess.Popen([
                            "python", 
                            "trainer/gsplat/train.py", 
                            self.MODE[self.gsplat_mode],
                            "--data_dir", self.gsplat_training_dir, 
                            "--data_factor", "1",
                            "--result_dir",self.gsplat_output_dir
                        ])
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

        self.iterations.append(stats["iteration"])
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
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        return folder_path
    
    def close(self):
        if self.gsplat_trainer != None:
            self.gsplat_trainer.terminate()
            self.gsplat_trainer.wait()
        if self.origin_trainer != None:
            self.origin_trainer.terminate()
            self.origin_trainer.wait()
        super().close()
    
