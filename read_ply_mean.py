from plyfile import PlyData
import numpy as np
import sys

def main():

    ply = PlyData.read("assets/model/face.ply")
    v = ply["vertex"].data

    mean_x = np.mean(v["x"])
    mean_y = np.mean(v["y"])
    mean_z = np.mean(v["z"])

    print(mean_x, mean_y, mean_z)

if __name__ == "__main__":
    main()