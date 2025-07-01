import socket
from threading import Thread
import time
from typing import List
import numpy as np
import torch
import torch.nn
import json

from renderer.base_renderer import Renderer
from PIL import Image
import torchvision.transforms as transforms


class AsyncConnector(Thread):
    def __init__(self, delay, host, port):
        super(AsyncConnector, self).__init__()
        self.delay = delay
        self.host = host
        self.port = port
        self._socket = None
        self.socket = None
        self.finished = False
        self.running = True
        self.start()

    def run(self):
        while self.socket is None and self.running:
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.port))
                self.socket = self._socket
                self.finished = True
                return
            except Exception as e:
                self._socket = None
                self.socket = None
                time.sleep(self.delay)

    def restart(self):
        self._socket = None
        self.socket = None
        self.run()


class FittingRenderer(Renderer):
    def __init__(self,host,port):
        super().__init__()
        self.connector = AsyncConnector(1, host, port)
        self.host = host
        self.port = port
        self.socket = self.connector.socket
        self.next_bytes = bytes()
        self.transform = transforms.ToTensor()

    def restart_connector(self):
        self.connector = AsyncConnector(1, self.host, self.port)
    
    def read(self, resolution):
        try:
            current_bytes = 0
            expected_bytes = resolution * resolution * 3
            try_counter = 100
            counter = 0
            message = bytes()
            while current_bytes < expected_bytes:
                message += self.socket.recv(expected_bytes - current_bytes)
                current_bytes = len(message)
                counter += 1
                if counter > try_counter:
                    print("Package loss")
                    break

            verify_len = self.socket.recv(4)
            verify_len = int.from_bytes(verify_len, "little")
            verify_data = self.socket.recv(verify_len)
            try:
                verify_dict = json.loads(verify_data)
            except Exception:
                verify_dict = {}
            image = np.frombuffer(message, dtype=np.uint8).reshape(resolution, resolution, 3)
            image = torch.from_numpy(np.array(image)) / 255.0
            image = image.permute(2, 0, 1)
            return image, verify_dict
        except Exception as e:
            print("Read Error", e)
            self.restart_connector()
            return torch.zeros([3, resolution, resolution]), {}

    def send(self, message):
        try:
            message_encode = json.dumps(message).encode()
            message_len_bytes = len(message_encode).to_bytes(4, "little")
            self.socket.sendall(message_len_bytes + bytes(message_encode))
        except Exception as e:
            self.restart_connector()
            print("Send Error", e)

    def _render_impl(
        self,
        res,
        fitting_path,
        do_training,
        img_size,
        stop_at_value=-1,
        single_training_step=False,
        img_normalize=False,
        slider={},
        **other_args,
    ):
        
        self.socket = self.connector.socket
        if self.socket is None:
            if self.connector.finished:
                self.restart_connector()
            res.message = f"Waiting for connection\n{self.host}:{self.port}"
            return
        images = []
        for image_path in fitting_path:
            image = Image.open(image_path).convert('RGB') 
            image = self.transform(image)
            images.append(image)
        message = {
            "train": do_training,
            "slider": slider,
            "single_training_step": single_training_step,
            "stop_at_value": stop_at_value, 
        }
        self.send(message)
        image, stats = self.read(img_size)
        if len(stats.keys()) > 0:
            res.training_stats = stats
            res.error = res.training_stats["error"]
        
        images.append(image)
        if len(images) > 0:
            self._return_image(
                images,
                res,
                normalize=img_normalize,
            )
        
    def close(self):
        self.connector.running = False
