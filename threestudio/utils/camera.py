import torch

from EditorGS.gaussiansplatting.utils.graphics_utils import fov2focal
from EditorGS.gaussiansplatting.scene.cameras import Simple_Camera
import torch.nn.functional as F
from threestudio.utils.typing import *
import numpy as np


def camera_ray_sample_points(
    camera,
    scene_radius: float,
    n_points: int = 256,
    mask: Bool[Tensor, "H W"] = None,
    sampling_method: str = "inbound",
) -> Float[Tensor, "N n_points 3"]:
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)

    # Fuck this shit transpose
    c2w = torch.inverse(camera.world_view_transform.T)
    # c2w = camera.world_view_transform
    R = c2w[:3, :3]
    T = c2w[:3, 3]

    camera_space_ij = torch.meshgrid(
        torch.arange(camera.image_width, dtype=torch.float32),
        torch.arange(camera.image_height, dtype=torch.float32),
        indexing="xy",
    )
    camera_space_ij = torch.stack(camera_space_ij, dim=-1)

    if mask is None:
        camera_space_ij = camera_space_ij.reshape(-1, 2)
    else:
        camera_space_ij = camera_space_ij[mask]

    assert camera_space_ij.ndim == 2

    camera_space_ij = (
        camera_space_ij
        - torch.tensor([[camera.image_width, camera.image_height]], dtype=torch.float32)
        / 2
    )

    view_space_xy = camera_space_ij * torch.tensor([[1 / fx, 1 / fy]])
    view_space_xyz = torch.cat(
        [view_space_xy, torch.ones_like(view_space_xy[..., 0:1])], dim=-1
    )

    view_space_directions = torch.bmm(
        R[None, ...].repeat(view_space_xyz.shape[0], 1, 1), view_space_xyz[..., None]
    )[..., 0]
    view_space_xyz = view_space_directions + T[None, ...]

    distances = None
    if sampling_method == "inbound":
        distances = torch.linspace(0, scene_radius * 2, n_points)
    elif sampling_method == "segmented":
        # linear inside scene radius, linear disparity outside, I forget the exact name of this sampling strategy
        distances_inside = torch.linspace(0, scene_radius, n_points // 2)
        distances_outside = torch.linspace(0, 1, n_points // 2)
    else:
        raise ValueError(f"Unknown sampling method {sampling_method}")

    return (
        view_space_directions[..., None, :] * distances[None, ..., None]
        + view_space_xyz[..., None, :]
    )


def project(camera: Simple_Camera, points3d):
    # TODO: should be equivalent to full_proj_transform.T
    if isinstance(points3d, list):
        points3d = torch.stack(points3d, dim=0)
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
    xy = points3d_camera[..., :2] / points3d_camera[..., 2:]
    ij = (
        xy
        * torch.tensor(
            [
                fov2focal(camera.FoVx, camera.image_width),
                fov2focal(camera.FoVy, camera.image_height),
            ],
            dtype=torch.float32,
            device=xy.device,
        )
        + torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=xy.device,
        )
        / 2
    ).to(torch.long)

    return ij


def unproject(camera: Simple_Camera, points2d, depth):
    origin = camera.camera_center
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3].T

    if isinstance(points2d, (list, tuple)):
        points2d = torch.stack(points2d, dim=0)

    points2d[0] *= camera.image_width
    points2d[1] *= camera.image_height
    points2d = points2d.to(w2c.device)
    points2d = points2d.to(torch.long)

    directions = (
        points2d
        - torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=w2c.device,
        )
        / 2
    ) / torch.tensor(
        [
            fov2focal(camera.FoVx, camera.image_width),
            fov2focal(camera.FoVy, camera.image_height),
        ],
        dtype=torch.float32,
        device=w2c.device,
    )
    padding = torch.ones_like(directions[..., :1])
    directions = torch.cat([directions, padding], dim=-1)
    if directions.ndim == 1:
        directions = directions[None, ...]
    directions = torch.einsum("ij,bj->bi", R, directions)
    directions = F.normalize(directions, dim=-1)

    points3d = (
        directions * depth[0][points2d[..., 1], points2d[..., 0]] + origin[None, ...]
    )

    return points3d


def get_point_depth(points3d, camera: Simple_Camera):
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
    depth = points3d_camera[..., 2:]
    return depth

def pixel_to_3d(pixel, camera, depth):
    extrinsic = camera.extr.cpu().numpy() if hasattr(camera.extr, 'cpu') else camera.extr
    
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    
    cx = camera.image_width / 2
    cy = camera.image_height / 2
    
    u, v = pixel
    
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth
    
    point_camera = np.array([x_cam, y_cam, z_cam, 1.0]) 
    
    if extrinsic.shape == (4, 4):
        point_world = extrinsic @ point_camera
    else:
        point_world = np.vstack([extrinsic, [0, 0, 0, 1]]) @ point_camera 
    
    return point_world[:3]

# def project_3d_to_2d(point_3d, camera):

#     extrinsic = camera.world_view_transform.inverse().T
#     # extrinsic = camera.extr.cpu().numpy() if hasattr(camera.extr, 'cpu') else camera.extr
    
#     if point_3d.shape[-1] == 3:
#         point_camera = point_3d = np.concatenate([point_3d, np.ones(1)])
    
#     fx = fov2focal(camera.FoVx, camera.image_width)
#     fy = fov2focal(camera.FoVy, camera.image_height)
#     cx = camera.image_width / 2
#     cy = camera.image_height / 2
    
#     if extrinsic.shape == (4, 4):
#         point_camera = np.linalg.inv(extrinsic.cpu().numpy()) @ point_3d
#     else:
#         extrinsic_4x4 = np.eye(4)
#         extrinsic_4x4[:3, :] = extrinsic.cpu().numpy()
#         point_camera = np.linalg.inv(extrinsic_4x4) @ point_3d
    
#     x = (point_camera[0] / point_camera[2]) * fx + cx
#     y = (point_camera[1] / point_camera[2]) * fy + cy
    
#     return np.array([x, y])

def project_3d_to_2d(points_3d, camera):
    # 处理输入点并转换为齐次坐标
    points_3d = np.asarray(points_3d)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, 3)
    
    # 添加齐次坐标维度
    if points_3d.shape[1] == 3:
        homogeneous = np.ones((points_3d.shape[0], 1))
        points_homogeneous = np.hstack([points_3d, homogeneous])
    else:
        points_homogeneous = points_3d

    # 获取并处理外参矩阵
    extrinsic = camera.world_view_transform.inverse().T
    if hasattr(extrinsic, 'cpu'):
        extrinsic = extrinsic.cpu().numpy()
    
    # 构建4x4外参矩阵
    if extrinsic.shape == (4, 4):
        extrinsic_4x4 = extrinsic
    else:
        extrinsic_4x4 = np.eye(4)
        extrinsic_4x4[:3, :] = extrinsic[:3]  # 假设外参提供前三行

    # 计算世界到相机的变换矩阵
    world_to_cam = np.linalg.inv(extrinsic_4x4)

    # 转换所有点到相机坐标系
    points_camera = (world_to_cam @ points_homogeneous.T).T

    # 获取相机内参
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    cx = camera.image_width / 2
    cy = camera.image_height / 2

    # 投影计算（带防零除保护）
    z = points_camera[:, 2]
    z = np.where(z == 0, 1e-10, z)  # 避免除以零
    x = (points_camera[:, 0] / z) * fx + cx
    y = (points_camera[:, 1] / z) * fy + cy

    return np.column_stack((x, y))