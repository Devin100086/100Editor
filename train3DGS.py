import os
import sys
sys.path.append("./gaussiansplatting")
import torch
from random import randint
from gaussiansplatting.utils.loss_utils import l1_loss, ssim
from gaussiansplatting.gaussian_renderer import render, network_gui
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.utils.general_utils import safe_state
from gaussiansplatting.train import training
import uuid
from tqdm import tqdm
from gaussiansplatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from gaussiansplatting.arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)
args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print("Optimizing " + args.model_path)

# Initialize system state (RNG)
safe_state(args.quiet)

# Start GUI server, configure and run training
network_gui.init(args.ip, args.port)
torch.autograd.set_detect_anomaly(args.detect_anomaly)
training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

# All done
print("\nTraining complete.")
