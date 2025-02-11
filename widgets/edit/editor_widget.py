import os
import subprocess
import pickle
from imgui_bundle import imgui
from EditorGS.gaussiansplatting.scene.cameras import CustomCam             
from EditorGS.GUIEditor.train_coarse_add import TrainCoarseAdd                                        
from lumina3D_utils.gui_utils import imgui_utils
from lumina3D_utils.gui_utils.easy_imgui import label

from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
from diffusers.image_processor  import VaeImageProcessor
import torch
import cv2
import sys

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh

from widgets.widget import Widget
import glfw
from PIL import Image
import numpy as np
from OpenGL.GL import *

def BrushEdit_Pipeline(pipe, 
                    prompts,
                    mask_np,
                    original_image, 
                    generator,
                    num_inference_steps,
                    guidance_scale,
                    control_strength,
                    negative_prompt,
                    blending):
    if mask_np.ndim != 3:
        mask_np = mask_np[:, :, np.newaxis]

    mask_np = mask_np / 255
    height, width = mask_np.shape[0], mask_np.shape[1]
    ## resize the mask and original image to the same size which is divisible by vae_scale_factor
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor, do_convert_rgb=True)
    height_new, width_new = image_processor.get_default_height_width(original_image, height, width)
    mask_np = cv2.resize(mask_np, (width_new, height_new))[:,:,np.newaxis]
    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:, :, np.newaxis]

    original_image = cv2.resize(original_image, (width_new, height_new))

    init_image = original_image * (1 - mask_np)
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray((mask_np.repeat(3, -1) * 255).astype(np.uint8)).convert("RGB")

    brushnet_conditioning_scale = float(control_strength)
    
    images = pipe(
        prompts, 
        init_image, 
        mask_image, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale,
        negative_prompt=negative_prompt,
        height=height_new,
        width=width_new,
    ).images

    ## convert to vae shape format, must be divisible by 8
    original_image_pil = Image.fromarray(original_image).convert("RGB")
    init_image_np = np.array(image_processor.preprocess(original_image_pil, height=height_new, width=width_new).squeeze())
    init_image_np = ((init_image_np.transpose(1,2,0) + 1.) / 2.) * 255
    init_image_np = init_image_np.astype(np.uint8)
    if blending:
        mask_blurred = mask_blurred * 0.5 + 0.5
        image_all = []
        for image_i in images:
            image_np = np.array(image_i)
            ## blending
            image_pasted = init_image_np * (1 - mask_blurred) + mask_blurred * image_np
            image_pasted = image_pasted.astype(np.uint8)
            image = Image.fromarray(image_pasted)
            image_all.append(image)
    else:
        image_all = images


    return image_all, mask_image, mask_np, init_image_np

class Config:
    def __init__(self, ply_file_path, data_source, mask_prompt, edit_train_steps, left_up, right_down, zoom):
        self.gs_source = ply_file_path
        self.colmap_dir = data_source
        self.text_prompt = mask_prompt
        self.edit_train_steps = edit_train_steps
        self.left_up = [-1,-1] if left_up == None else (int(left_up[0]), int(left_up[1]))
        self.right_down = [-1,-1] if right_down == None else (int(right_down[0]), int(right_down[1]))
        self.zoom = -1 if zoom == None else zoom

 
class EditorWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Editor")
        # Option
        self.lambda_l1 = 10
        self.lambda_p = 10
        self.lambda_anchor_color = 0
        self.lambda_anchor_geo = 50
        self.lambda_anchor_scale = 50
        self.lambda_anchor_opacity = 50
        self.edit_until_step = 1000
        self.per_editing_step = 10
        self.edit_begin_step = 0
        self.edit_cam_num = 48
        self.edit_train_steps = 1500
        self.cameara_update_step = 500
        
        # text-edit
        self.guidance_type = ["InstructPix2Pix","ControlNet-Pix2Pix"]
        self.guidance_item = 0
        self.text_prompt = "turn him a clown"
        self.text_edit_trainer = None

        # mask-edit
        self.mask_prompt = "add a red hat"
        self.mask = False
        self.start_rec_pos = None
        self.end_rec_pos = None
        self.rec_start = None 
        self.rec_end = None
        self.drawing_rect = False
        self.mask_edit_trainer = None
        self.traincoarseadd = None
        self.depth = 1

        # adding
        self.editing_option = 0
        self.points = []
        self.current_color = [1.0, 1.0, 1.0, 1.0]
        self.line_width = 2.0
        self.sketch_prompt = "a man wear a wreath on head"
        self.negative_prompt = "ugly, low quality"
        self.generate_3D_prompt = "a red hat"
        self.is_drawing = False
        self.turn_camera = False
        self.sketch_edit_trainer = None
        self.seed = 1
        self.single_image = None
        self.edit_single = False
        self.segmentation_prompt = "hat"
        self.video_editing = True

        self.draw_image = False
        self.edit3D = False
        self.text_change = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        edit_image = False
        if show:
            if imgui.begin_tab_bar("EditBar"):
                if imgui.begin_tab_item("Option")[0]:
                    label("Camera Num", viz.label_w)
                    _, self.edit_cam_num = imgui.slider_int("##Camera Num", self.edit_cam_num, 12, 200, format="%d")
                    label("Total Step", viz.label_w)
                    _, self.edit_train_steps = imgui.slider_int("##Total Step", self.edit_train_steps, 0, 5000, format="%d")
                    label("Cameara Update Step", viz.label_w)
                    _, self.cameara_update_step = imgui.slider_int("##Cameara Update Step", self.cameara_update_step, 0, 5000, format="%d")
                    label("Lambda L1", viz.label_w)
                    _, self.lambda_l1 = imgui.slider_int("##Lambda L1", self.lambda_l1, 0, 100, format="%d")
                    label("Lambda Perceptual", viz.label_w)
                    _, self.lambda_p = imgui.slider_int("##Lambda Perceptual", self.lambda_p, 0, 100, format="%d")
                    label("Lambda Anchor Color", viz.label_w)
                    _, self.lambda_anchor_color = imgui.slider_int("##Lambda Anchor Color", self.lambda_anchor_color, 0, 500, format="%d")
                    label("Lambda Anchor Geo", viz.label_w)
                    _, self.lambda_anchor_geo = imgui.slider_int("##Lambda Anchor Geo", self.lambda_anchor_geo, 0, 500, format="%d")
                    label("Lambda Anchor Scale", viz.label_w)
                    _, self.lambda_anchor_scale = imgui.slider_int("##Lambda Anchor Scale", self.lambda_anchor_scale, 0, 500, format="%d")
                    label("Lambda Anchor Opacity", viz.label_w)
                    _, self.lambda_anchor_opacity = imgui.slider_int("##Lambda Anchor Opacity", self.lambda_anchor_opacity, 0, 500, format="%d")
                    label("Edit Until Step", viz.label_w)
                    _, self.edit_until_step = imgui.slider_int("##Edit Until Step", self.edit_until_step, 0, 5000, format="%d")
                    label("Edit Begining", viz.label_w)
                    _, self.edit_begin_step = imgui.slider_int("##Edit Begining", self.edit_begin_step, 0, 5000, format="%d")
                    label("Edit Interval", viz.label_w)
                    _, self.per_editing_step = imgui.slider_int("##Edit Interval", self.per_editing_step, 4, 48, format="%d")
                    imgui.end_tab_item()

                if imgui.begin_tab_item("text")[0]:
                    label("guidance_type", viz.label_w)
                    clicked, self.guidance_item = imgui.combo(
                        "##guidance_type",                
                        self.guidance_item,           
                        self.guidance_type                   
                    )
                    label("prompt", viz.label_w)
                    changed, self.text_prompt = imgui.input_text("##Prompt", self.text_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    if not self.edit3D:
                        if imgui_utils.button("Edit", width=viz.button_w):
                            self.edit3D = True
                            self.text_edit_trainer = subprocess.Popen([
                                "python", 
                                "EditorGS/GUIEditor/train_edit.py", 
                                "--gs_source",str(viz.args.ply_file_paths[0]),
                                "--colmap_dir",str(viz.args.data_source),
                                "--edit_cam_num", str(self.edit_cam_num),
                                "--guidance_type", str(self.guidance_type[self.guidance_item]),
                                "--text_prompt", str(self.text_prompt),
                                "--edit_train_steps", str(self.edit_train_steps),
                                "--per_editing_step", str(self.per_editing_step),
                                "--edit_begin_step", str(self.edit_begin_step),
                                "--edit_until_step", str(self.edit_until_step),
                                "--lambda_l1", str(self.lambda_l1),
                                "--lambda_p", str(self.lambda_p),
                                "--lambda_anchor_color", str(self.lambda_anchor_color),
                                "--lambda_anchor_geo", str(self.lambda_anchor_geo),
                                "--lambda_anchor_scale", str(self.lambda_anchor_scale),
                                "--lambda_anchor_opacity", str(self.lambda_anchor_opacity)
                            ])
                    else:
                        if imgui_utils.button("Stop", width=viz.button_w):
                            self.edit3D = False
                            self.text_edit_trainer.terminate()
                            self.text_edit_trainer.wait()
                    imgui.end_tab_item()
                
                if imgui.begin_tab_item("mask")[0]:
                    self.points = []
                    if imgui.button("clear"):
                        self.rec_start = None
                        self.rec_end = None
                    label("Paint Mask", viz.label_w)
                    changed, self.mask = imgui.checkbox("##Mask", self.mask)
                    label("prompt", viz.label_w)
                    changed, self.mask_prompt = imgui.input_text("##Prompt", self.mask_prompt, 256)
                    self.text_change = True if imgui.is_item_active() else False
                    if not self.mask and not self.text_change and self.judge_move():  
                        self.draw_image = False
                        self.rec_start, self.rec_end = None, None
                    if self.mask:
                        self.draw_mask()
                        self.draw_image = True
                        imgui.begin_disabled()
                        if imgui_utils.button("Add", width=viz.button_w):
                            pass
                        imgui.end_disabled()    
                    else:
                        edit_image = True
                        if imgui_utils.button("Add", width=viz.button_w):
                            cache_dir = "tmp_edit"
                            os.makedirs("tmp_edit", exist_ok=True)
                            mask = self.get_mask(viz.origin_image, viz.edit_image)
                            mask.save(f"{cache_dir}/mask.png")
                            origin = Image.fromarray(viz.result.image).convert("RGB")
                            origin.save(f"{cache_dir}/origin.png")
                            left_up = [self.rec_start[0]-viz.pane_w, self.rec_start[1]]
                            right_down = [self.rec_end[0]-viz.pane_w, self.rec_end[1]]
                            zoom = min((viz.content_width - viz.pane_w) / viz._tex_obj.width, viz.content_height / viz._tex_obj.height)
                            R = viz.extr.inverse()[:3, :3].T.numpy()
                            T = viz.extr.inverse()[:3, 3].numpy()
                            fov_rad = viz.fov / 360 * 2 * np.pi
                            cam = CustomCam(origin.size[0], origin.size[1], fov_rad, fov_rad, R, T, viz.extr.cuda())
                            with open(f'{cache_dir}/camera.pkl', 'wb') as f:
                                pickle.dump(cam, f)     
                            
                            cfg = Config(
                                ply_file_path=viz.args.ply_file_paths[0],
                                data_source=viz.args.data_source,
                                mask_prompt=self.mask_prompt,
                                edit_train_steps=self.edit_train_steps,
                                left_up=left_up,
                                right_down=right_down,
                                zoom=zoom,
                            )
                            self.traincoarseadd = TrainCoarseAdd(cfg=cfg)  
                            self.traincoarseadd.add(cam)
                            self.rec_start, self.rec_end = None, None

                    imgui.separator()
                    label("Depth", viz.label_w)
                    _, self.depth = imgui.slider_float("##Depth", self.depth, 0, 10, format="%.1f")
                    if os.path.exists("tmp_add/inpaint_gs.obj"):
                        if imgui_utils.button("Show", width=viz.button_w): 
                            self.edit3D = True 
                            origin = Image.fromarray(viz.result.image).convert("RGB")
                            R = viz.extr.inverse()[:3, :3].T.numpy()
                            T = viz.extr.inverse()[:3, 3].numpy()
                            fov_rad = viz.fov / 360 * 2 * np.pi
                            cam = CustomCam(origin.size[0], origin.size[1], fov_rad, fov_rad, R, T, viz.extr.cuda())
                            with open(f'tmp_edit/camera.pkl', 'wb') as f:
                                pickle.dump(cam, f)    

                            if self.mask_edit_trainer != None:
                                self.mask_edit_trainer.terminate()
                                self.mask_edit_trainer.wait()   
        
                            self.mask_edit_trainer = subprocess.Popen([
                                "python", 
                                "EditorGS/GUIEditor/train_coarse_add.py", 
                                "--gs_source",str(viz.args.ply_file_paths[0]),
                                "--colmap_dir",str(viz.args.data_source),
                                "--depth", str(self.depth),
                                "--text_prompt", "",
                                "--edit_train_steps", "-1",
                                "--left_up", "-1","-1",
                                "--right_down", "-1","-1",
                                "--zoom", "-1",
                                "--cam_dir",str(f"tmp_edit/camera.pkl"),
                            ])
                    else:
                        imgui.begin_disabled()
                        if imgui_utils.button("Show", width=viz.button_w):
                            pass
                        imgui.end_disabled()

                    imgui.end_tab_item()
                
                if imgui.begin_tab_item("add")[0]:
                    self.rec_start = None
                    self.rec_end = None
                    label("Add Option", viz.label_w)
                    if imgui.radio_button("Sketch", self.editing_option == 0):
                        self.editing_option = 0
                    imgui.same_line()
                    if imgui.radio_button("Text", self.editing_option == 1):
                        self.editing_option = 1

                    if self.editing_option == 0:
                        label("Painting", viz.label_w)
                        _, self.turn_camera = imgui.checkbox("##painting", self.turn_camera)
                        if not self.turn_camera and self.judge_move():  
                            self.draw_image = False
                            self.points = []

                    elif self.editing_option == 1:
                        label("Generate 3D Prompt", viz.label_w)
                        _, self.generate_3D_prompt = imgui.input_text("##3D Prompt", self.generate_3D_prompt, 256)
                        if imgui_utils.button("Generate", width=viz.button_w):
                            self.generate3D()

                    if not self.turn_camera or self.editing_option == 1:   
                        label("Depth", viz.label_w)
                        _, self.depth = imgui.slider_float("##Depth", self.depth, 0, 10, format="%.2f")
                        if os.path.exists("tmp_add/inpaint_gs.obj") and os.path.exists("tmp_edit/camera.pkl"):
                            if imgui_utils.button("Show", width=viz.button_w): 
                                self.edit_single = False
                                self.edit3D = True 
                                origin = Image.fromarray(viz.result.image).convert("RGB")
                                R = viz.extr.inverse()[:3, :3].T.numpy()
                                T = viz.extr.inverse()[:3, 3].numpy()
                                fov_rad = viz.fov / 360 * 2 * np.pi
                                cam = CustomCam(origin.size[0], origin.size[1], fov_rad, fov_rad, R, T, viz.extr.cuda())
                                with open(f'tmp_edit/camera.pkl', 'wb') as f:
                                    pickle.dump(cam, f)    

                                if self.mask_edit_trainer != None:
                                    self.mask_edit_trainer.terminate()
                                    self.mask_edit_trainer.wait()   
            
                                self.mask_edit_trainer = subprocess.Popen([
                                    "python", 
                                    "EditorGS/GUIEditor/train_coarse_add.py", 
                                    "--gs_source",str(viz.args.ply_file_paths[0]),
                                    "--colmap_dir",str(viz.args.data_source),
                                    "--depth", str(self.depth),
                                    "--text_prompt", "",
                                    "--edit_train_steps", "-1",
                                    "--left_up", "-1","-1",
                                    "--right_down", "-1","-1",
                                    "--zoom", "-1",
                                    "--cam_dir",str(f"tmp_edit/camera.pkl"),
                                ])
                        else:
                            imgui.begin_disabled()
                            if imgui_utils.button("Show", width=viz.button_w):
                                pass
                            imgui.end_disabled()

                    else:
                        _, self.line_width = imgui.slider_float("width", self.line_width, 1.0, 10.0)
                        _, self.current_color = imgui.color_edit4("color choice", self.current_color)
                        if imgui.button("clear"):
                            self.points = []
                        imgui.separator()

                        self.handle_mouse_input()
                        self.draw_image = True
                        edit_image = True
                        label("prompt", viz.label_w)
                        _, self.sketch_prompt = imgui.input_text("##Prompt", self.sketch_prompt, 256)
                        label("Negative Prompt", viz.label_w)
                        _, self.negative_prompt = imgui.input_text("##Negative Prompt", self.negative_prompt, 256)
                        self.text_change = True if imgui.is_item_active() else False

                        label("Seed", viz.label_w)
                        _, self.seed = imgui.slider_int("##Seed", self.seed, 0, 10, format="%d")
                        if imgui_utils.button("Edit", width=viz.button_w):
                            self.edit3D = False
                            self.edit_single = True
                            mask = self.get_mask(viz.origin_image, viz.edit_image)
                            cache_dir = "tmp_edit"
                            os.makedirs("tmp_edit", exist_ok=True)
                            mask = self.get_mask(viz.origin_image, viz.edit_image)
                            mask.save(f"{cache_dir}/mask.png")
                            origin = Image.fromarray(viz.result.image).convert("RGB")
                            origin.save(f"{cache_dir}/origin.png")

                            self.single_image = self.edit_single_image(
                                                      prompts = self.sketch_prompt,
                                                      negative_prompt = self.negative_prompt,
                                                      seed = self.seed,
                                                      image_path = f"{cache_dir}/origin.png",
                                                      mask_path = f"{cache_dir}/mask.png")
                            self.points = []

                        imgui.separator()

                        label("Segmentation", viz.label_w)
                        changed, self.segmentation_prompt = imgui.input_text("##Segmentation", self.segmentation_prompt, 256)
                        if imgui_utils.button("Mesh", width=viz.button_w):
                            cache_dir = "tmp_edit"
                            os.makedirs("tmp_edit", exist_ok=True)
                            origin = Image.fromarray(viz.result.image).convert("RGB")
                            origin.save(f"{cache_dir}/origin.png")
                            cfg = Config(
                                ply_file_path=viz.args.ply_file_paths[0],
                                data_source=viz.args.data_source,
                                mask_prompt=self.mask_prompt,
                                edit_train_steps=self.edit_train_steps,
                                left_up=None,
                                right_down=None,
                                zoom=None,
                            )
                            self.traincoarseadd = TrainCoarseAdd(cfg=cfg)  
                            self.traincoarseadd.add_sketch(origin, self.segmentation_prompt)                                
                        
                        label("Depth", viz.label_w)
                        _, self.depth = imgui.slider_float("##Depth", self.depth, 0, 10, format="%.2f")
                        if os.path.exists("tmp_add/inpaint_gs.obj") and os.path.exists("tmp_edit/camera.pkl"):
                            if imgui_utils.button("Show", width=viz.button_w): 
                                self.edit_single = False
                                self.edit3D = True 
                                origin = Image.fromarray(viz.result.image).convert("RGB")
                                R = viz.extr.inverse()[:3, :3].T.numpy()
                                T = viz.extr.inverse()[:3, 3].numpy()
                                fov_rad = viz.fov / 360 * 2 * np.pi
                                cam = CustomCam(origin.size[0], origin.size[1], fov_rad, fov_rad, R, T, viz.extr.cuda())
                                with open(f'tmp_edit/camera.pkl', 'wb') as f:
                                    pickle.dump(cam, f)    

                                if self.mask_edit_trainer != None:
                                    self.mask_edit_trainer.terminate()
                                    self.mask_edit_trainer.wait()   
            
                                self.mask_edit_trainer = subprocess.Popen([
                                    "python", 
                                    "EditorGS/GUIEditor/train_coarse_add.py", 
                                    "--gs_source",str(viz.args.ply_file_paths[0]),
                                    "--colmap_dir",str(viz.args.data_source),
                                    "--depth", str(self.depth),
                                    "--text_prompt", "",
                                    "--edit_train_steps", "-1",
                                    "--left_up", "-1","-1",
                                    "--right_down", "-1","-1",
                                    "--zoom", "-1",
                                    "--cam_dir",str(f"tmp_edit/camera.pkl"),
                                ])
                        else:
                            imgui.begin_disabled()
                            if imgui_utils.button("Show", width=viz.button_w):
                                pass
                            imgui.end_disabled()
                        
                        imgui.separator()

                        label("Video", viz.label_w)
                        _, self.video_editing = imgui.checkbox("##Video", self.video_editing)
                        if imgui_utils.button("Edit3D", width=viz.button_w):
                            if self.mask_edit_trainer != None:
                                    self.mask_edit_trainer.terminate()
                                    self.mask_edit_trainer.wait()   
                            self.edit3D = True
                            origin = Image.fromarray(viz.result.image).convert("RGB")
                            cache_dir = "tmp_edit"
                            R = viz.extr.inverse()[:3, :3].T.numpy()
                            T = viz.extr.inverse()[:3, 3].numpy()
                            fov_rad = viz.fov / 360 * 2 * np.pi
                            cam = CustomCam(origin.size[0], origin.size[1], fov_rad, fov_rad, R, T, viz.extr.cuda())
                            with open(f'{cache_dir}/camera.pkl', 'wb') as f:
                                pickle.dump(cam, f)     
                            self.mask_edit_trainer = subprocess.Popen([
                                "python", 
                                "EditorGS/GUIEditor/train_fine_add.py", 
                                "--gs_source",str(viz.args.ply_file_paths[0]),
                                "--colmap_dir",str(viz.args.data_source),
                                "--text_prompt", str(self.sketch_prompt),
                                "--negative_prompt", str(self.negative_prompt),
                                "--edit_train_steps", str(self.edit_train_steps),
                                "--cameara_update_step", str(self.cameara_update_step),
                                "--mask_dir", str(f"{cache_dir}/mask.png"),
                                "--video", str(self.video_editing),
                                "--edit_cam_num", str(self.edit_cam_num),
                                "--guidance_type", str(self.guidance_type[self.guidance_item]),
                                "--per_editing_step", str(self.per_editing_step),
                                "--edit_begin_step", str(self.edit_begin_step),
                                "--edit_until_step", str(self.edit_until_step),
                                "--lambda_l1", str(self.lambda_l1),
                                "--lambda_p", str(self.lambda_p),
                                "--lambda_anchor_color", str(self.lambda_anchor_color),
                                "--lambda_anchor_geo", str(self.lambda_anchor_geo),
                                "--lambda_anchor_scale", str(self.lambda_anchor_scale),
                                "--lambda_anchor_opacity", str(self.lambda_anchor_opacity)
                            ])
                    
                    imgui.end_tab_item()

            imgui.end_tab_bar()   

        viz.args.text_change = self.text_change
        viz.args.draw_image = self.draw_image
        viz.args.edit3D = self.edit3D
        viz.args.edit_single = self.edit_single
        viz.args.single_image = self.single_image

        viz.args.rec_start = self.rec_start
        viz.args.rec_end = self.rec_end
        viz.args.edit_image = edit_image
        viz.args.turn_camera = self.turn_camera  
        viz.args.mask = self.mask     
        viz.args.points = self.points
        viz.args.current_color = self.current_color
        viz.args.line_width = self.line_width
        
    def handle_mouse_input(self):
        if glfw.get_mouse_button(self.viz._glfw_window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.viz._glfw_window)

            region_left = self.viz.pane_w
            region_right = self.viz.content_width
            region_top = 0
            region_bottom = self.viz.content_height
            if region_left <= x <= region_right and region_top <= y <= region_bottom:
                if not self.is_drawing:
                    self.is_drawing = True
                self.points.append((x, y))
        else:
            self.is_drawing = False
            if self.points:
                self.points.append(None)
    
    def judge_move(self):
        if imgui.is_mouse_dragging(0) or imgui.is_mouse_dragging(1) or imgui.is_mouse_dragging(2):
            if imgui.get_mouse_drag_delta(0).x != 0 or imgui.get_mouse_drag_delta(1).x != 0 or imgui.get_mouse_drag_delta(2).x != 0:
                if imgui.get_mouse_drag_delta(0).y != 0 or imgui.get_mouse_drag_delta(1).y != 0 or imgui.get_mouse_drag_delta(2).y != 0:
                    return True
        if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
            return True
        if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
            return True
        if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
            return True
        if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
            return True
        if imgui.get_io().mouse_wheel != 0:
            return True
        return False
    
    def get_mask(self, ori, edit):
        ori = np.array(ori)
        edit = np.array(edit)
        diff = np.abs(ori - edit) 
        mask = np.any(diff > 0, axis=-1)
        mask_image = np.zeros_like(ori[:, :, 0])
        mask_image[mask] = 255
        mask_image_pil = Image.fromarray(np.stack([mask_image] * 3, axis=-1).astype(np.uint8))
        return mask_image_pil

    def draw_mask(self):
        if imgui.is_mouse_down(0):
            if not self.drawing_rect:
                self.start_rec_pos = imgui.get_mouse_pos()
                self.drawing_rect = True
            self.end_rec_pos = imgui.get_mouse_pos()

        if not imgui.is_mouse_down(0) and self.drawing_rect:
            self.drawing_rect = False
        if self.start_rec_pos and self.end_rec_pos:
            if self.start_rec_pos[0] >= self.viz.pane_w and self.end_rec_pos[0] >= self.viz.pane_w \
                   and self.start_rec_pos[1] >= 0 and self.end_rec_pos[1] >= 0: 
                self.rec_start, self.rec_end =  [self.start_rec_pos[0], self.start_rec_pos[1]], [self.end_rec_pos[0], self.end_rec_pos[1]]
            else:
                return
        else:
            self.rec_start, self.rec_end = None, None
    
    def edit_single_image(self, prompts, negative_prompt, seed, image_path, mask_path):
        BrushEdit_path = ".cache/models/"

        base_model_path = os.path.join(BrushEdit_path, "base_model/realisticVisionV60B1_v51VAE")
        torch_dtype = torch.float16
        brushnet_path = os.path.join(BrushEdit_path, "brushnetX")


        brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype, cache_dir=None)
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                base_model_path, brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
            )
        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # remove following line if xformers is not installed or when using Torch 2.0.
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        generator = torch.Generator("cuda").manual_seed(seed)
        num_inference_steps = 50
        guidance_scale = 7.5
        control_strength = 1
        num_samples = 1
        blending = True

        image = Image.open(f"{image_path}")
        original_image = np.array(image)
        mask_image = Image.open(f"{mask_path}").convert('L')  # 'L'表示灰度模式
        # mask = np.ones((image.size[1], image.size[0]), dtype=np.uint8) * 255
        # mask_image = Image.fromarray(mask)
        mask_np = np.array(mask_image)
        negative_prompt = "ugly, low quality"

        result,_,_,_ = BrushEdit_Pipeline(pipe, 
                                        prompts,
                                        mask_np,
                                        original_image, 
                                        generator,
                                        num_inference_steps,
                                        guidance_scale,
                                        control_strength,
                                        negative_prompt,
                                        blending)  
        del brushnet
        del pipe
        return np.array(result[0])

    def generate3D(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        diffusion = diffusion_from_config(load_config('diffusion'))

        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=15.0,
            model_kwargs=dict(texts=[self.generate_3D_prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        mesh_path = "tmp_add/inpaint_mesh.obj"
        gs_path = "tmp_add/inpaint_gs.obj"

        for i, latent in enumerate(latents):
            t = decode_latent_mesh(xm, latent).tri_mesh()
            with open(mesh_path, 'w') as f:
                t.write_obj(f)
        


        del xm
        del model
        del diffusion

        p3 = subprocess.Popen(
            [
                f"{sys.prefix}/bin/python",
                "train_from_mesh.py",
                "--mesh",
                mesh_path,
                "--save_path",
                gs_path,
                "--prompt",
                "",
            ]
        )
        p3.wait()

    def close(self):
        if self.text_edit_trainer != None:
            self.text_edit_trainer.terminate()
            self.text_edit_trainer.wait()
        if self.mask_edit_trainer != None:
            self.mask_edit_trainer.terminate()
            self.mask_edit_trainer.wait()
        if self.sketch_edit_trainer != None:
            self.sketch_edit_trainer.terminate()
            self.sketch_edit_trainer.wait()
        super().close()
























