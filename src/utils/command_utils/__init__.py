"""
Helper functions for Executing Command.
"""

import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable


def _script_path(*parts: str) -> str:
    return str((_REPO_ROOT.joinpath(*parts)).resolve())


def _run_python(script_path: str, *args: str):
    env = os.environ.copy()
    repo_path = str(_REPO_ROOT.resolve())
    src_path = str((_REPO_ROOT / "src").resolve())
    project_paths = f"{repo_path}{os.pathsep}{src_path}"
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{project_paths}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_paths
    return subprocess.Popen([_PYTHON, script_path, *args], cwd=str(_REPO_ROOT), env=env)

def training_text_editing_command(
    gs_source, colmap_dir, edit_cam_num, guidance_type, text_prompt, origin_prompt, edit_train_steps,
    per_editing_step, edit_begin_step, edit_until_step, lambda_l1, lambda_p,
    lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale, lambda_anchor_opacity,
    sam_option, seg_prompt, text_videoEditing, gs_lr_scaler, gs_lr_end_scaler, color_lr_scaler, 
    opacity_lr_scaler, scaling_lr_scaler, rotation_lr_scaler, camera, positive_sam_points, negative_sam_points, use_original_resolution,
    output_dir, hard_segmentation, mask_thres, earlystop, cps_patience_counter, cps_patience, cps_batch_count,
    clip_origin_prompt, clip_target_prompt
):
    process = _run_python(
        _script_path("src", "editor", "hundrededitor_gui", "edit.py"),
        "--gs_source", str(gs_source),
        "--colmap_dir", str(colmap_dir),
        "--edit_cam_num", str(edit_cam_num),
        "--guidance_type", str(guidance_type),
        "--text_prompt", str(text_prompt),
        "--origin_prompt", str(origin_prompt),
        "--edit_train_steps", str(edit_train_steps),
        "--per_editing_step", str(per_editing_step),
        "--edit_begin_step", str(edit_begin_step),
        "--edit_until_step", str(edit_until_step),
        "--lambda_l1", str(lambda_l1),
        "--lambda_p", str(lambda_p),
        "--lambda_anchor_color", str(lambda_anchor_color),
        "--lambda_anchor_geo", str(lambda_anchor_geo),
        "--lambda_anchor_scale", str(lambda_anchor_scale),
        "--lambda_anchor_opacity", str(lambda_anchor_opacity),
        "--sam_option",str(sam_option),
        "--positive_sam_points", str(positive_sam_points),
        "--negative_sam_points", str(negative_sam_points),
        "--seg_prompt",str(seg_prompt),
        "--gs_lr_scaler", str(gs_lr_scaler),
        "--gs_lr_end_scaler", str(gs_lr_end_scaler),
        "--color_lr_scaler", str(color_lr_scaler),
        "--opacity_lr_scaler", str(opacity_lr_scaler),
        "--scaling_lr_scaler", str(scaling_lr_scaler),
        "--rotation_lr_scaler", str(rotation_lr_scaler),
        "--camera", str(camera),
        "--video",str(text_videoEditing),
        "--use_original_resolution", str(use_original_resolution),
        "--output_dir", str(output_dir),
        "--hard_segmentation", str(hard_segmentation),
        "--mask_thres", str(mask_thres),
        "--earlystop", str(earlystop),
        "--cps_patience_counter", str(cps_patience_counter),
        "--cps_patience", str(cps_patience),
        "--cps_batch_count", str(cps_batch_count),
        "--clip_origin_prompt", str(clip_origin_prompt),
        "--clip_target_prompt", str(clip_target_prompt)
    )
    return process

def training_fine_adding_command(
    gs_source, colmap_dir, text_prompt, edit_train_steps, cameara_update_step,
    mask_dir, video, edit_cam_num, guidance_type, per_editing_step, edit_begin_step,
    edit_until_step, lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo,
    lambda_anchor_scale, lambda_anchor_opacity, densification_interval, densify_until_step, output_dir,
    camera
):
    process = _run_python(
        _script_path("src", "editor", "hundrededitor_gui", "add.py"),
        "--gs_source", str(gs_source),
        "--colmap_dir", str(colmap_dir),
        "--text_prompt", str(text_prompt),
        "--edit_train_steps", str(edit_train_steps),
        "--cameara_update_step", str(cameara_update_step),
        "--mask_dir", str(mask_dir),
        "--video", str(video),
        "--edit_cam_num", str(edit_cam_num),
        "--guidance_type", str(guidance_type),
        "--per_editing_step", str(per_editing_step),
        "--edit_begin_step", str(edit_begin_step),
        "--edit_until_step", str(edit_until_step),
        "--lambda_l1", str(lambda_l1),
        "--lambda_p", str(lambda_p),
        "--lambda_anchor_color", str(lambda_anchor_color),
        "--lambda_anchor_geo", str(lambda_anchor_geo),
        "--lambda_anchor_scale", str(lambda_anchor_scale),
        "--lambda_anchor_opacity", str(lambda_anchor_opacity),
        "--densification_interval", str(densification_interval),
        "--densify_until_step", str(densify_until_step),
        "--output_dir", str(output_dir),
        "--camera", str(camera)
    )
    return process

def training_delete_command(gs_source, colmap_dir, inpaint_scale, mask_dilate, edit_cam_num, delete_prompt,
    use_original_resolution, edit_train_steps, per_editing_step, edit_begin_step, edit_until_step,
    lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale,
    lambda_anchor_opacity, video, sam_type, camera, gs_lr_scaler, gs_lr_end_scaler, color_lr_scaler, 
    opacity_lr_scaler, scaling_lr_scaler, rotation_lr_scaler,  positive_sam_points, negative_sam_points, output_dir
):
    process = _run_python(
        _script_path("src", "editor", "hundrededitor_gui", "delete.py"),
        "--gs_source", str(gs_source),
        "--colmap_dir", str(colmap_dir),
        "--inpaint_scale", str(inpaint_scale),
        "--mask_dilate", str(mask_dilate),
        "--edit_cam_num", str(edit_cam_num),
        "--sam_type", str(sam_type),
        "--positive_sam_points", str(positive_sam_points),
        "--negative_sam_points", str(negative_sam_points),
        "--delete_prompt", str(delete_prompt),
        "--edit_train_steps", str(edit_train_steps),
        "--per_editing_step", str(per_editing_step),
        "--edit_begin_step", str(edit_begin_step),
        "--edit_until_step", str(edit_until_step),
        "--lambda_l1", str(lambda_l1),
        "--lambda_p", str(lambda_p),
        "--lambda_anchor_color", str(lambda_anchor_color),
        "--lambda_anchor_geo", str(lambda_anchor_geo),
        "--lambda_anchor_scale", str(lambda_anchor_scale),
        "--lambda_anchor_opacity", str(lambda_anchor_opacity),
        "--gs_lr_scaler", str(gs_lr_scaler),
        "--gs_lr_end_scaler", str(gs_lr_end_scaler),
        "--color_lr_scaler", str(color_lr_scaler),
        "--opacity_lr_scaler", str(opacity_lr_scaler),
        "--scaling_lr_scaler", str(scaling_lr_scaler),
        "--rotation_lr_scaler", str(rotation_lr_scaler),
        "--video", str(video),
        "--camera", str(camera),
        "--use_original_resolution", str(use_original_resolution),
        "--output_dir", str(output_dir)
    )
    return process

def get_3DGS_mask_command(
    gs_source, colmap_dir,sam_option, seg_prompt,
    camera, positive_sam_points, negative_sam_points
):
    process = _run_python(
        _script_path("src", "editor", "hundrededitor_gui", "generate_gs_mask.py"),
        "--gs_source", str(gs_source),
        "--colmap_dir", str(colmap_dir),
        "--sam_option",str(sam_option),
        "--positive_sam_points", str(positive_sam_points),
        "--negative_sam_points", str(negative_sam_points),
        "--seg_prompt",str(seg_prompt),
        "--camera", str(camera),
    )
    return process

def showing_colmap_command(data_path):
    process = _run_python(
        _script_path("src", "editor", "hundrededitor_gui", "show_colmap.py"),
        "--data", str(data_path)
    )
    return process

def sfm_reconstruction(source_path, colmap_executable, use_gpu):
    gpu = 1 if use_gpu == True else 0
    process = _run_python(
        _script_path("src", "editor", "gaussiansplatting", "convert.py"),
        "-s",str(source_path),
        "--colmap_executable",str(colmap_executable),
        "--gpu", str(gpu)
    )
    return process

def vggt_reconstruction(source_path):
    process = _run_python(
        _script_path("third_party", "vggt", "colmap.py"),
        "--scene_dir",str(source_path),
    )
    return process

def training_3DGS(
    training_path,
    soutput_dir,
    gpu,
    alpha,
    quiet=False,
    detect_anomaly=False,
    use_depth_loss=False,
    use_appearance_embedding=False,
    use_random_background=False,
):
    args = [
        "-s", training_path,
        "-m", soutput_dir,
        "--gpu", gpu,
        "--alpha", str(alpha),
        "--use_depth_loss", str(use_depth_loss),
        "--use_appearance_embedding", str(use_appearance_embedding),
    ]
    if use_random_background:
        args.append("--random_background")
    if quiet:
        args.append("--quiet")
    if detect_anomaly:
        args.append("--detect_anomaly")

    origin_trainer = _run_python(
        _script_path("src", "trainer", "origin", "train.py"),
        *args
    )
    return origin_trainer

def training_gsplat_3DGS(
    mode,
    gsplat_training_dir,
    gsplat_output_dir,
    alpha,
    use_bilateral_grid=False,
    use_taming_3dgs=False,
    use_depth_loss=False,
):
    args = [
        mode,
        "--data_dir", gsplat_training_dir,
        "--data_factor", "1",
        "--result_dir", gsplat_output_dir,
        "--alpha", str(alpha),
    ]
    if use_bilateral_grid:
        args.append("--use_bilateral_grid")
    if use_taming_3dgs:
        args.append("--visible_adam")
    if use_depth_loss:
        args.append("--depth_loss")

    gsplat_trainer = _run_python(
        _script_path("src", "trainer", "gsplat", "train.py"),
        *args
    )
    return gsplat_trainer
