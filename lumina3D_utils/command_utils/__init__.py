"""
Helper functions for Executing Command.
"""

import subprocess

def training_text_adding_command(
    gs_source, colmap_dir, edit_cam_num, guidance_type, text_prompt, edit_train_steps,
    per_editing_step, edit_begin_step, edit_until_step, lambda_l1, lambda_p,
    lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale, lambda_anchor_opacity,
    sam_option, seg_prompt, text_videoEditing, gs_lr_scaler, gs_lr_end_scaler, color_lr_scaler, 
    opacity_lr_scaler, scaling_lr_scaler, rotation_lr_scaler, camera, positive_sam_points, negative_sam_points
):
    process = subprocess.Popen([
        "python",
        "EditorGS/GUIEditor/train_edit.py",
        "--gs_source", str(gs_source),
        "--colmap_dir", str(colmap_dir),
        "--edit_cam_num", str(edit_cam_num),
        "--guidance_type", str(guidance_type),
        "--text_prompt", str(text_prompt),
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
        "--video",str(text_videoEditing)
    ])
    return process

def show_command(gs_source, colmap_dir, depth, cam_dir):
    process = subprocess.Popen([
        "python",
        "EditorGS/GUIEditor/show.py",
        "--gs_source", str(gs_source),
        "--colmap_dir", str(colmap_dir),
        "--depth", str(depth),
        "--cam_dir", str(cam_dir)
    ])
    return process

def training_fine_adding_command(
    gs_source, colmap_dir, text_prompt, edit_train_steps, cameara_update_step,
    seg_prompt, mask_dir, video, edit_cam_num, guidance_type, per_editing_step, edit_begin_step,
    edit_until_step, lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo,
    lambda_anchor_scale, lambda_anchor_opacity, densification_interval, densify_until_step,
):
    process = subprocess.Popen([
        "python",
        "EditorGS/GUIEditor/train_fine_add.py",
        "--gs_source", str(gs_source),
        "--colmap_dir", str(colmap_dir),
        "--text_prompt", str(text_prompt),
        "--edit_train_steps", str(edit_train_steps),
        "--cameara_update_step", str(cameara_update_step),
        "--seg_prompt", str(seg_prompt),
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
        "--densify_until_step", str(densify_until_step)
    ])
    return process

def training_delete_command(gs_source, colmap_dir, inpaint_scale, mask_dilate, edit_cam_num, delete_prompt,
    inpaint_prompt, edit_train_steps, per_editing_step, edit_begin_step, edit_until_step,
    lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale,
    lambda_anchor_opacity, video, sam_type, sam_points, camera, gs_lr_scaler, gs_lr_end_scaler, color_lr_scaler, 
    opacity_lr_scaler, scaling_lr_scaler, rotation_lr_scaler,  positive_sam_points, negative_sam_points
):
    process = subprocess.Popen([
        "python",
        "EditorGS/GUIEditor/train_delete.py",
        "--gs_source", str(gs_source),
        "--colmap_dir", str(colmap_dir),
        "--inpaint_scale", str(inpaint_scale),
        "--mask_dilate", str(mask_dilate),
        "--edit_cam_num", str(edit_cam_num),
        "--sam_type", str(sam_type),
        "--positive_sam_points", str(positive_sam_points),
        "--negative_sam_points", str(negative_sam_points),
        "--delete_prompt", str(delete_prompt),
        "--inpaint_prompt", str(inpaint_prompt),
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
    ])
    return process

def showing_colmap_command(data_path):
    process = subprocess.Popen([
        "python",
        "EditorGS/GUIEditor/show_colmap.py",
        "--data", str(data_path)
    ])
    return process

def colmap_reconstruction(source_path, colmap_executable, use_gpu):
    gpu = 1 if use_gpu == True else 0
    process = subprocess.Popen([
        "python",
        "EditorGS/gaussiansplatting/convert.py",
        "-s",str(source_path),
        "--colmap_executable",str(colmap_executable),
        "--gpu", str(gpu)
    ])
    return process


