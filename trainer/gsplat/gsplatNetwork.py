import copy
import torch
import traceback
import socket
from render import rasterize_splats
import json
import math

class GsplatNetwork:
    def __init__(self, host="127.0.0.1", port=6009):
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
        print(f"Creating splatviz network connector for host={host} and port={port}")
        self.stop_at_value = -1

    def try_connect(self):
        try:
            self.conn, self.addr = self.listener.accept()
            print(f"\nConnected to  at {self.addr}")
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
                print("Package loss")
                break
        return json.loads(message.decode("utf-8"))

    def send(self, message_bytes, training_stats):
        if message_bytes != None:
            self.conn.sendall(message_bytes)
        self.conn.sendall(len(training_stats).to_bytes(4, 'little'))
        self.conn.sendall(training_stats.encode())

    def receive(self):
        message = self.read()
        width = message["resolution_x"]
        height = message["resolution_y"]
        if width != 0 and height != 0:
            try:
                self.do_training = bool(message["train"])
                fovy = message["fov_y"]
                fovx = message["fov_x"]
                znear = message["z_near"]
                zfar = message["z_far"]
                self.do_shs_python = bool(message["shs_python"])
                self.do_rot_scale_python = bool(message["rot_scale_python"])
                self.keep_alive = bool(message["keep_alive"])
                self.scaling_modifer = message["scaling_modifier"]
                world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
                world_view_transform[:, 1] = -world_view_transform[:, 1]
                world_view_transform[:, 2] = -world_view_transform[:, 2]
                full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
                full_proj_transform[:, 1] = -full_proj_transform[:, 1]
                self.custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
                self.edit_text = message["edit_text"]
                self.slider = message["slider"]
                self.stop_at_value = message["stop_at_value"]
                self.single_training_step = message["single_training_step"]
            except Exception as e:
                traceback.print_exc()
                raise e

    def render(self, sh_degree, gsplat, loss, world_size, iteration, device, opt):
        if self.conn == None:
            self.try_connect()
        while self.conn != None:
            edit_error = ""
            try:
                net_image_bytes = None
                self.receive()
                if self.custom_cam != None:
                    with torch.no_grad():
                        renders_network, _, _ = rasterize_splats(
                            splats=gsplat,
                            cfg=opt,
                            world_size=world_size,
                            camtoworlds=self.custom_cam.view_inv[None],
                            Ks=self.custom_cam.Ks[None].to(device),
                            width=self.custom_cam.image_width,
                            height=self.custom_cam.image_height,
                            sh_degree=sh_degree,
                            near_plane=self.custom_cam.znear,
                            far_plane=self.custom_cam.zfar,
                            render_mode="RGB+ED" if opt.depth_loss else "RGB",
                        )
                        if renders_network.shape[-1] == 4:
                            colors, depths = renders_network[..., 0:3], renders_network[..., 3:4]
                        else:
                            colors, depths = renders_network, None
                        net_image = colors.reshape(-1, *colors.shape[2:])
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy())

                opt_copy = copy.copy(opt)
                del opt_copy.strategy
                training_stats = json.dumps({
                    "loss": loss,
                    "iteration": iteration,
                    "num_gaussians": len(gsplat["means"]),
                    "sh_degree": sh_degree,
                    "train_params": vars(opt_copy),
                    "error": edit_error,
                    "paused": self.stop_at_value == iteration
                })
                self.send(net_image_bytes, training_stats)
                if self.do_training and ((iteration < int(opt.max_steps)) or not self.keep_alive) and self.stop_at_value != iteration:
                    break
                if self.single_training_step:
                    break

            except Exception as e:
                print(e)
                self.conn = None


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


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.view_inv = torch.inverse(self.world_view_transform).T
        self.camera_center = self.view_inv[3][:3]
        self.Ks = self.compute_intrinsics()
    
    def compute_intrinsics(self):

        f_x = self.image_width / (2 * math.tan(self.FoVx / 2))
        f_y = self.image_height / (2 * math.tan(self.FoVy / 2))
        
        c_x = self.image_width / 2
        c_y = self.image_height / 2
        
        K = torch.tensor([
            [f_x, 0,  c_x],
            [0,  f_y, c_y],
            [0,  0,  1]
        ], dtype=torch.float32)
        
        return K