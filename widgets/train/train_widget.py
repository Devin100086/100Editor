from argparse import ArgumentParser
import threading
from imgui_bundle import imgui, ImVec2
from imgui_bundle import implot
import numpy as np
import os
import multiprocessing

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui
from lumina3D_utils.gui_utils import imgui_utils
import tkinter as tk
import torch
from gaussiansplatting.utils.general_utils import safe_state
from train import training_3DGS
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
        self.context = torch.multiprocessing.get_context("spawn")

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text("Training Parameter")

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
                    subprocess.Popen(["python", "trainer/train.py", "-s", self.training_path, "--gpu", self.gpu_items[self.use_gpu]])
                    if self.stop_from_renderer:
                        self.stop_at_value = -1
            else:
                if imgui.button("Pause Training", ImVec2(viz.label_w_large, 0)):
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
    
    def training(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)

        parser.add_argument("--gpu", type=str, default=self.gpu_items[self.use_gpu])
        parser.add_argument('--ip', type=str, default="127.0.0.1")
        parser.add_argument('--port', type=int, default=6009)
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument('--detect_anomaly', action='store_true', default=self.detect_anomaly)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
        parser.add_argument("--quiet", action="store_true",default=self.quiet)
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--start_checkpoint", type=str, default = None)

        args = parser.parse_args(['-s',self.training_path])
        args.save_iterations.append(args.iterations)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        # Initialize system state (RNG)
        safe_state(args.quiet)

        network_gui.init(args.ip, args.port)
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        # train_3DGS = threading.Thread(target=demo)
        train_3DGS = self.context.Process(target=training_3DGS,args=(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from))
        # training_3DGS(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
        train_3DGS.start()
        # All done
        print("\nTraining complete.")


        
