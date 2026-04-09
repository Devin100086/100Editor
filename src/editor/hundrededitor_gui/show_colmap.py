import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap


def _load_reconstruction(data_path: Path):
    """Load COLMAP sparse model with new/old pycolmap API compatibility."""
    model_path = str(data_path)

    # Newer API
    reconstruction_cls = getattr(pycolmap, "Reconstruction", None)
    if reconstruction_cls is not None:
        try:
            return reconstruction_cls(model_path), "reconstruction"
        except Exception:
            pass

    # Older API
    scene_manager_cls = getattr(pycolmap, "SceneManager", None)
    if scene_manager_cls is not None:
        manager = scene_manager_cls(model_path)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        return manager, "scene_manager"

    raise RuntimeError(
        "Unsupported pycolmap API: neither `Reconstruction` nor `SceneManager` is available."
    )

def draw_colmap_geometries(data_path):
    print(data_path)
    recon, backend = _load_reconstruction(Path(data_path))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    # add axis
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # vis.add_geometry(axis)

    # add points
    pointxyz = []
    pointcolor = []
    if backend == "reconstruction":
        for point3D in recon.points3D.values():
            pointxyz.append([point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]])
            # Different versions may expose either `color` or `rgb`.
            color = getattr(point3D, "color", getattr(point3D, "rgb", None))
            if color is None:
                pointcolor.append([1.0, 1.0, 1.0])
            else:
                pointcolor.append(np.asarray(color) / 255.0)
    else:
        # SceneManager stores points/colors as arrays.
        if hasattr(recon, "points3D"):
            pointxyz = np.asarray(recon.points3D).tolist()
        if hasattr(recon, "point3D_colors"):
            pointcolor = (np.asarray(recon.point3D_colors) / 255.0).tolist()
        else:
            pointcolor = [[1.0, 1.0, 1.0] for _ in range(len(pointxyz))]

    if len(pointxyz) == 0:
        raise ValueError(f"No 3D points found in sparse model: {data_path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointxyz)
    pcd.colors = o3d.utility.Vector3dVector(pointcolor)
    vis.add_geometry(pcd)
    # add camera
    # for id, image in images.items():
    #     T_w2c = image.cam_from_world.matrix()
    #     # add 0,0,0,1 to T_w2c
    #     T_w2c = np.concatenate([T_w2c, np.array([[0,0,0,1]])], axis=0)
    #     T_c2w = np.linalg.inv(T_w2c)
    #     cam = cameras[image.camera_id]
    #     h, w, f= cam.height, cam.width, cam.params[0]
    #     camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, w/2, h/2)
    #     cam_lineset = o3d.geometry.LineSet.create_camera_visualization(camera_intrinsic, T_w2c, scale=0.2)
    #     cam_coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #     cam_coor.transform(T_c2w)
    #     vis.add_geometry(cam_lineset)
    #     vis.add_geometry(cam_coor)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                    help='input colmap scene directory')
    args = parser.parse_args()
    draw_colmap_geometries(Path(args.data))
