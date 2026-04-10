#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import shlex
import subprocess
from pathlib import Path

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--gpu", default=1, type=int)
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = args.colmap_executable if len(args.colmap_executable) > 0 else "colmap"
magick_command = args.magick_executable if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if args.gpu else 0

distorted_dir = os.path.join(args.source_path, "distorted")
distorted_sparse_dir = os.path.join(distorted_dir, "sparse")
database_path = os.path.join(distorted_dir, "database.db")
input_dir = os.path.join(args.source_path, "input")
runtime_colmap_log_dir = Path(__file__).resolve().parents[3] / "runtime" / "colmap"
runtime_colmap_log_dir.mkdir(parents=True, exist_ok=True)


_help_cache = {}


def _executable_prefix(executable):
    if os.name == "nt" and executable.lower().endswith((".bat", ".cmd")):
        return ["cmd", "/c", executable]
    return [executable]


colmap_prefix = _executable_prefix(colmap_command)
magick_prefix = _executable_prefix(magick_command)


def _run_command(cmd, step_name):
    logging.info("Running command: %s", " ".join(shlex.quote(c) for c in cmd))
    try:
        completed = subprocess.run(cmd, check=False)
    except OSError as exc:
        logging.error("%s failed to start: %s", step_name, exc)
        exit(1)
    if completed.returncode != 0:
        logging.error(f"{step_name} failed with code {completed.returncode}. Exiting.")
        exit(completed.returncode)


def _get_command_help(command_name):
    if command_name in _help_cache:
        return _help_cache[command_name]
    help_text = ""
    help_commands = [
        colmap_prefix + [command_name, "--log_target", "stderr", "-h"],
        colmap_prefix + [command_name, "-h"],
    ]
    for cmd in help_commands:
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except OSError:
            continue
        stdout_text = completed.stdout or ""
        if completed.returncode == 0 and stdout_text:
            help_text = stdout_text
            break
        if not help_text and stdout_text:
            help_text = stdout_text
    _help_cache[command_name] = help_text
    return help_text


def _append_colmap_log_options(command_name, cmd):
    help_text = _get_command_help(command_name)
    if "--log_target" in help_text:
        cmd.extend(["--log_target", "stderr_and_file"])
    if "--log_path" in help_text:
        cmd.extend(["--log_path", str(runtime_colmap_log_dir)])


def _resolve_gpu_option(command_name, legacy_opt, modern_opt):
    help_text = _get_command_help(command_name)
    legacy_flag = f"--{legacy_opt}"
    modern_flag = f"--{modern_opt}"
    if legacy_flag in help_text:
        return legacy_flag
    if modern_flag in help_text:
        return modern_flag
    logging.warning(
        "Cannot detect GPU option for `%s`. Checked `%s` and `%s`.",
        command_name,
        legacy_flag,
        modern_flag,
    )
    return None

if not args.skip_matching:
    os.makedirs(distorted_sparse_dir, exist_ok=True)
    # Re-running with another COLMAP version can leave an incompatible DB schema.
    if os.path.isfile(database_path):
        logging.info("Removing stale COLMAP database: %s", database_path)
        os.remove(database_path)

    ## Feature extraction
    feature_gpu_option = _resolve_gpu_option(
        "feature_extractor", "SiftExtraction.use_gpu", "FeatureExtraction.use_gpu"
    )
    feat_extracton_cmd = [
        *colmap_prefix,
        "feature_extractor",
        "--database_path",
        database_path,
        "--image_path",
        input_dir,
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        args.camera,
    ]
    if feature_gpu_option is not None:
        feat_extracton_cmd.extend([feature_gpu_option, str(use_gpu)])
    _append_colmap_log_options("feature_extractor", feat_extracton_cmd)
    _run_command(feat_extracton_cmd, "Feature extraction")

    ## Feature matching
    matching_gpu_option = _resolve_gpu_option(
        "exhaustive_matcher", "SiftMatching.use_gpu", "FeatureMatching.use_gpu"
    )
    feat_matching_cmd = [
        *colmap_prefix,
        "exhaustive_matcher",
        "--database_path",
        database_path,
    ]
    if matching_gpu_option is not None:
        feat_matching_cmd.extend([matching_gpu_option, str(use_gpu)])
    _append_colmap_log_options("exhaustive_matcher", feat_matching_cmd)
    _run_command(feat_matching_cmd, "Feature matching")

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = [
        *colmap_prefix,
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        input_dir,
        "--output_path",
        distorted_sparse_dir,
        "--Mapper.ba_global_function_tolerance=0.000001",
    ]
    _append_colmap_log_options("mapper", mapper_cmd)
    _run_command(mapper_cmd, "Mapper")

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = [
    *colmap_prefix,
    "image_undistorter",
    "--image_path",
    input_dir,
    "--input_path",
    os.path.join(distorted_sparse_dir, "0"),
    "--output_path",
    args.source_path,
    "--output_type",
    "COLMAP",
]
_append_colmap_log_options("image_undistorter", img_undist_cmd)
_run_command(img_undist_cmd, "Image undistorter")

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        _run_command([*magick_prefix, "mogrify", "-resize", "50%", destination_file], "50% resize")

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        _run_command([*magick_prefix, "mogrify", "-resize", "25%", destination_file], "25% resize")

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        _run_command([*magick_prefix, "mogrify", "-resize", "12.5%", destination_file], "12.5% resize")

print("Done.")
