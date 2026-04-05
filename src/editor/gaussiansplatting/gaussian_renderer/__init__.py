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

import torch
import math
import os
# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )
from acc_diff_gaussian_rasterization_editor import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)
from editor.gaussiansplatting.utils.sh_utils import eval_sh


def camera2rasterizer(viewpoint_camera, bg_color: torch.Tensor, sh_degree: int = 0):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    return rasterizer


def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    separate_sh=False
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform.float(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.float(),
        prefiltered=False,
        debug=False,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features

        shs = shs.float()
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # import pdb; pdb.set_trace()

    if separate_sh:
        rendered_image, radii, depth = rasterizer(
            means3D = means3D.float(),
            means2D = means2D.float(),
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity.float(),
            scales = scales.float(),
            rotations = rotations.float(),
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth = rasterizer(
            means3D=means3D.float(),
            means2D=means2D.float(),
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity.float(),
            scales=scales.float(),
            rotations=rotations.float(),
            cov3D_precomp=cov3D_precomp,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth_3dgs": depth,
    }


# from editor.gaussiansplatting.scene.gaussian_model import GaussianModel


def point_cloud_render(
    viewpoint_camera,
    xyz,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    screenspace_points = (
        torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = torch.ones_like(xyz[..., 0:1])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = torch.ones_like(xyz) * 0.005
    rotations = torch.zeros([xyz.shape[0], 4], dtype=xyz.dtype, device=xyz.device)
    rotations[..., 0] = 1.0

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(
    #             -1, 3, (pc.max_sh_degree + 1) ** 2
    #         )
    #         dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
    #             pc.get_features.shape[0], 1
    #         )
    #         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         shs = pc.get_features

    #     shs = shs.float()
    # else:
    #     colors_precomp = override_color
    colors_precomp = torch.ones_like(xyz[..., 0:1]).repeat(1, 3)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # import pdb; pdb.set_trace()
    rendered_image, radii, depth = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity.float(),
        scales=scales.float(),
        rotations=rotations.float(),
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth_3dgs": depth,
    }


class _CompatPipe:
    """Fallback render config for legacy callers of render_simple/render_drag."""

    compute_cov3D_python = False
    convert_SHs_python = False
    antialiasing = False


def _alpha_from_depth(depth: torch.Tensor) -> torch.Tensor:
    return (depth > 0).to(depth.dtype)


def render_simple(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    debug=False,
):
    out = render(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=_CompatPipe(),
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
    )
    depth = out["depth_3dgs"]
    return {
        "render": out["render"],
        "viewspace_points": out["viewspace_points"],
        "visibility_filter": out["visibility_filter"],
        "radii": out["radii"],
        "depth": depth,
        "alpha": _alpha_from_depth(depth),
    }


def render_drag(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    d_xyz=None,
    d_rotation=None,
    d_scaling=None,
    d_opacity=None,
    d_color=None,
    scale_const=None,
    d_rotation_bias=None,
    scaling_modifier=1.0,
    override_color=None,
    debug=False,
):
    screenspace_points = (
        torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform.float(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.float(),
        prefiltered=False,
        debug=debug,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    drag_mask_path = os.getenv(
        "HUNDREDEDITOR_DRAG_MASK_PATH", "runtime/cache/drag/mask.pt"
    )
    if os.path.exists(drag_mask_path):
        mask = torch.load(drag_mask_path, map_location=pc.get_xyz.device)
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=pc.get_xyz.device)
        mask = mask.bool()
    else:
        mask = torch.ones_like(pc.get_xyz[..., 0], dtype=torch.bool, device=pc.get_xyz.device)

    if mask.shape[0] != pc.get_xyz.shape[0]:
        mask = torch.ones_like(pc.get_xyz[..., 0], dtype=torch.bool, device=pc.get_xyz.device)

    mask_f = mask.unsqueeze(-1).to(pc.get_xyz.dtype)

    means3D = pc.get_xyz if d_xyz is None else pc.get_xyz + d_xyz * mask_f
    means2D = screenspace_points
    opacity = pc.get_opacity if d_opacity is None else pc.get_opacity + d_opacity * mask_f

    scales = pc.get_scaling if d_scaling is None else pc.get_scaling + d_scaling * mask_f
    if scale_const is not None:
        scales = torch.ones_like(scales) * scale_const

    rotations = pc.get_rotation if d_rotation is None else pc.get_rotation + d_rotation * mask_f
    if d_rotation_bias is not None:
        # Keep backward-compatible arg without breaking current renderer.
        rotations = rotations + d_rotation_bias * mask_f

    if override_color is None:
        if d_color is None:
            shs = pc.get_features
        else:
            shs = torch.cat(
                [pc.get_features[:, :1] + d_color[:, None], pc.get_features[:, 1:]],
                dim=1,
            )
        colors_precomp = None
    else:
        shs = None
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity.float(),
        scales=scales.float(),
        rotations=rotations.float(),
        cov3D_precomp=None,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "alpha": _alpha_from_depth(depth),
        "bg_color": bg_color,
    }
