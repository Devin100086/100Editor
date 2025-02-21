from lumina3D  import Lumina3D
import argparse 

def main():
    parser = argparse.ArgumentParser(description="webui")
    parser.add_argument("--data_path", help="root path for .ply files", default="/home/wucunqi/Desktop/code/3D/gaussian-splatting/output/depth+appearance_pearson(total)/point_cloud/iteration_7000/point_cloud.ply")
    # parser.add_argument("--data_path", help="root path for .ply files", default="./resources/sample_scenes/truck/compression_config.yml")    
    parser.add_argument("--mode", help="[default, decoder, attach]", default="default")
    parser.add_argument("--host", help="host address", default="127.0.0.1")
    parser.add_argument("--port", help="port", default=6009)

    args = parser.parse_args()
    lumina3D = Lumina3D(args)
    while not lumina3D.should_close():
        lumina3D.draw_frame()
    lumina3D.close()


if __name__ == "__main__":
    main()
