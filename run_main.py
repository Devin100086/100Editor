from HundredEditor  import HundredEditor
import argparse 

def main():
    parser = argparse.ArgumentParser(description="webui")
    parser.add_argument("--data_path", help="root path for .ply files", default="outputs/Turn_his_hair_blonde@2026_03_13_20_03/result.ply")
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