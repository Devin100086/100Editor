import copy
import torch
import traceback
import socket
import json
import math
import time
from gsplat import rasterization, rasterization_2dgs
from tqdm import tqdm

class FittingNetwork:
    def __init__(self, host="127.0.0.1", port=7090):
        self.slider = None
        self.edit_text = None
        self.custom_cam = None
        self.scaling_modifier = None
        self.keep_alive = None
        self.do_rot_scale_python = None
        self.do_shs_python = None
        self.do_training = None
        self.host = host
        self.port = port
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)
        self.conn = None
        self.addr = None
        tqdm.write(f"Creating 100Editor network connector for host={host} and port={port}")
        self.stop_at_value = -1

    def wait_for_connection(self, timeout_s: float = 5.0, sleep_s: float = 0.01) -> bool:
        start_time = time.time()
        while self.conn is None and (time.time() - start_time) < timeout_s:
            self.try_connect()
            if self.conn is None:
                time.sleep(sleep_s)
        return self.conn is not None

    def try_connect(self):
        try:
            self.conn, self.addr = self.listener.accept()
            tqdm.write(f"Connected to {self.addr}")
            self.conn.settimeout(None)
        except Exception as inst:
            pass

    def read(self):
        messageLength = self.conn.recv(4)
        expected_bytes = int.from_bytes(messageLength, 'little')

        current_bytes = 0
        try_counter = 50
        counter = 0
        message = bytes()
        while current_bytes < expected_bytes:
            message += self.conn.recv(expected_bytes - current_bytes)
            current_bytes = len(message)
            counter += 1
            if counter > try_counter:
                tqdm.write("Package loss")
                break
        return json.loads(message.decode("utf-8"))

    def send(self, message_bytes, training_stats):
        if message_bytes != None:
            self.conn.sendall(message_bytes)
        self.conn.sendall(len(training_stats).to_bytes(4, 'little'))
        self.conn.sendall(training_stats.encode())

    def receive(self):
        message = self.read()
        try:
            self.do_training = bool(message["train"])
            self.slider = message["slider"]
            self.stop_at_value = message["stop_at_value"]
            self.single_training_step = message["single_training_step"]
        except Exception as e:
            traceback.print_exc()
            raise e

    def render(self, runner, iteration, loss, max_steps, model_type):
        if self.conn == None:
            if not self.wait_for_connection(timeout_s=5.0):
                return
        while self.conn != None:
            edit_error = ""
            try:
                net_image_bytes = None
                self.receive()
                Ks = self.get_Ks(runner)
                if model_type == "3dgs":
                    rasterize_fnc = rasterization
                elif model_type == "2dgs":
                    rasterize_fnc = rasterization_2dgs

                with torch.no_grad():
                    renders = rasterize_fnc(
                            runner.means,
                            runner.quats / runner.quats.norm(dim=-1, keepdim=True),
                            runner.scales,
                            torch.sigmoid(runner.opacities),
                            torch.sigmoid(runner.rgbs),
                            runner.viewmat[None],
                            Ks[None],
                            runner.W,
                            runner.H,
                            packed=False,
                        )[0]
                    out_img = renders[0]
                    net_image_bytes = memoryview((torch.clamp(out_img, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy())

                training_stats = json.dumps({
                    "loss": loss,
                    "iteration": iteration,
                    "error": edit_error,
                    "paused": self.stop_at_value == iteration
                })
                self.send(net_image_bytes, training_stats)
                if self.do_training and ((iteration < int(max_steps)) or not self.keep_alive) and self.stop_at_value != iteration:
                    break
                if self.single_training_step:
                    break

            except Exception as e:
                tqdm.write(str(e))
                self.conn = None
    
    def get_Ks(self,runner):
        KS = torch.tensor(
            [
                [runner.focal, 0, runner.W / 2],
                [0, runner.focal, runner.H / 2],
                [0, 0, 1],
            ],
            device=runner.device,
        )
        return KS


class EasyDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
