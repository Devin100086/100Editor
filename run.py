import argparse
import sys
from pathlib import Path
from src.app.hundred_editor import HundredEditor

repo_root = Path(__file__).resolve().parent
src_dir = repo_root / "src"
for path in (repo_root, src_dir):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

def main():
    parser = argparse.ArgumentParser(description="webui")
    parser.add_argument("--data_path", help="root path for .ply files", default="/media/wucunqi/data/results/face/3DGS+depth/point_cloud/iteration_30000/point_cloud.ply")
    # parser.add_argument("--data_path", help="root path for .ply files", default="./resources/sample_scenes/truck/compression_config.yml")    
    parser.add_argument("--mode", help="[default, decoder, attach]", default="default")
    parser.add_argument("--host", help="host address", default="127.0.0.1")
    parser.add_argument("--port", help="port", default=6009)   

    args = parser.parse_args()      
    hundred_editor = HundredEditor(args)
    while not hundred_editor.should_close():
        hundred_editor.draw_frame()
    hundred_editor.close()


if __name__ == "__main__":
    main()