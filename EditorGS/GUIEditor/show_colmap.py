from pathlib import Path
import open3d as o3d

import os, argparse,sys
import numpy as np

here_path = Path(__file__).resolve().parent
sys.path.append(str(here_path))
import pycolmap
# "CameraModel", ["model_id", "model_name", "num_params"])
# "Camera", ["id", "model", "width", "height", "params"])
# "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
# "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

import numpy as np
import open3d as o3d
import pycolmap
import matplotlib.pyplot as plt

def draw_colmap_geometries(data_path):
    print(data_path)
    recon = pycolmap.Reconstruction(data_path)

    cameras, images, points3D = recon.cameras, recon.images, recon.points3D

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    # add axis
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # vis.add_geometry(axis)

    # add points
    pointxyz = []
    pointcolor = []
    for id, point3D in points3D.items():
        pointxyz.append([point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]])
        pointcolor.append(point3D.color/255)

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

