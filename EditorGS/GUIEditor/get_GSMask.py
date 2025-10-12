from argparse import ArgumentParser
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EditorGS.GUIEditor.train_base import *
from EditorGS.gaussiansplatting.gaussian_renderer import render
from EditorGS.GUIEditor.utils import *
from threestudio.utils.camera import pixel_to_3d

from torchvision.utils import save_image
try:
    from acc_diff_gaussian_rasterization_editor import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

class MaskCatcher:
    def __init__(self, cfg):
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.mask_thres = cfg.mask_thres
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        # load
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.positive_sam_points = np.load(cfg.positive_sam_points)
        self.negative_sam_points = np.load(cfg.negative_sam_points)
        self.save_mask_tmp =  os.path.join(os.path.dirname(cfg.positive_sam_points),"mask")
        with open(args.camera, 'rb') as f:
            self.cam  = pickle.load(f)
        
        self.use_sparse_adam = cfg.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    def get_mask(self, sam_option, seg_prompt):
    
        if sam_option == 0:
            return
        elif sam_option == 1:
            self.masks, _ = self.update_mask(self.colmap_cameras, text_prompt=seg_prompt)
        elif sam_option == 2:

            positive_points3d = []
            negative_points3d = []
            # positive
            for i, sam_point in enumerate(self.positive_sam_points):
                depth = render(self.cam, self.gaussian, self.pipe ,self.background_tensor, separate_sh=self.use_sparse_adam)[
                    "depth_3dgs"
                ]
                # depth = render_simple(self.cam[i], gaussian_copy, self.background_tensor)["depth"]
                depth = (1/depth).detach().cpu().numpy()
                sam_point = sam_point * np.array([self.cam.image_width, self.cam.image_height])
                unprojected_points3d = pixel_to_3d(sam_point, self.cam, depth[0][int(sam_point[1]), int(sam_point[0])])
                # point2d = project_3d_to_2d(unprojected_points3d, self.cam[i])
                positive_points3d.append(unprojected_points3d)
            
            # negative
            for i, sam_point in enumerate(self.negative_sam_points):
                depth = render(self.cam, self.gaussian, self.pipe ,self.background_tensor, separate_sh=self.use_sparse_adam)[
                    "depth_3dgs"
                ]
                # depth = render_simple(self.cam[i], gaussian_copy, self.background_tensor)["depth"]
                depth = (1/depth).detach().cpu().numpy()
                sam_point = sam_point * np.array([self.cam.image_width, self.cam.image_height])
                unprojected_points3d = pixel_to_3d(sam_point, self.cam, depth[0][int(sam_point[1]), int(sam_point[0])])
                # point2d = project_3d_to_2d(unprojected_points3d, self.cam[i])
                negative_points3d.append(unprojected_points3d)
            
            positive_points3d = np.array(positive_points3d)
            negative_points3d = np.array(negative_points3d)
            self.masks, _ = self.update_sam_mask_with_point_prompt(self.colmap_cameras, positive_points3d, negative_points3d)

        elif sam_option == 3:

            self.positive_sam_points = np.empty((0,2)) if self.positive_sam_points.shape[0] == 0 else self.positive_sam_points * np.array([self.cam.image_width, self.cam.image_height])
            self.negative_sam_points = np.empty((0,2)) if self.negative_sam_points.shape[0] == 0 else self.negative_sam_points * np.array([self.cam.image_width, self.cam.image_height])
            render_folder = os.path.join(os.path.dirname(self.save_mask_tmp), "render")
            init_render = render(self.cam, self.gaussian, self.pipe ,self.background_tensor, separate_sh=self.use_sparse_adam)["render"]
            save_image(init_render[None], f"{render_folder}/{0:05d}" + ".jpg")
            self.masks, _ = self.update_sam2_mask_with_point_prompt(self.colmap_cameras, 
                                                                    self.positive_sam_points ,
                                                                    self.negative_sam_points)

        torch.save(self.gaussian.mask, os.path.join(os.path.dirname(self.save_mask_tmp),"mask.pt"))
    
    def update_mask(self, edit_cameras, text_prompt = "hat", type = "default") -> None:

        from threestudio.utils.sam import LangSAMTextSegmentor
        lang_sam = LangSAMTextSegmentor().to(get_device())

        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
        kernel =  np.ones((5,5),np.uint8)

        for cam in tqdm(edit_cameras):
            cur_cam = cam
            if type == "default":
                this_frame = render(
                    cur_cam, self.gaussian, self.pipe, self.background_tensor
                )["render"]
            else:
                this_frame = render(
                    cur_cam, self.gaussian2, self.pipe, self.background_tensor
                )["render"]
            
            mask = lang_sam(this_frame.unsqueeze(0).permute(0,2,3,1), text_prompt)[
                    0
                ].to(get_device())

            masks.append(mask.cpu().numpy().astype(np.uint8) * 255)
            self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

        weights /= weights_cnt + 1e-7
        selected_mask = weights > self.mask_thres
        selected_mask = selected_mask[:, 0]
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)

        return masks, selected_mask
    
    def update_sam_mask_with_point_prompt(
        self, edit_cameras, positive_points3d=None, negative_points3d=None
    ):
        os.makedirs(self.save_mask_tmp, exist_ok=True)
        os.system(f"rm -rf {self.save_mask_tmp}/*")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_checkpoint = "./.cache/models/sam2/sam2.1_hiera_large.pt"
        sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_predictor = SAM2ImagePredictor(build_sam2(sam2_model_cfg, sam2_checkpoint))
        positive_points3d = positive_points3d if positive_points3d is not None else self.positive_points3d
        negative_points3d = negative_points3d if negative_points3d is not None else self.negative_points3d
        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
        for cam in tqdm(edit_cameras):
            cur_cam = cam
            assert len(positive_points3d) > 0
            positive_points2ds = project_3d_to_2d(positive_points3d, cur_cam) if len(positive_points3d) > 0 else np.empty((0,2))
            negative_points2ds = project_3d_to_2d(negative_points3d, cur_cam) if len(negative_points3d) > 0 else np.empty((0,2))
            img = render(cur_cam, self.gaussian, self.pipe, self.background_tensor, separate_sh=self.use_sparse_adam)[
                "render"
            ]
            sam2_predictor.set_image(
                np.asarray(to_pil_image(img.cpu())),
            )

            positive_points2ds = np.empty((0,2)) if positive_points2ds.shape[0] == 0 else positive_points2ds
            negative_points2ds = np.empty((0,2)) if negative_points2ds.shape[0] == 0 else negative_points2ds
            positive_label = np.empty((0), dtype=np.int64) if positive_points2ds.shape[0] == 0 else np.array([1] * positive_points2ds.shape[0], dtype=np.int64) 
            negative_label = np.empty((0), dtype=np.int64) if negative_points2ds.shape[0] == 0 else np.array([0] * negative_points2ds.shape[0], dtype=np.int64)
            
            point_coords = np.concatenate((positive_points2ds, negative_points2ds), axis=0)
            point_labels = np.concatenate((positive_label, negative_label), axis=0) 
            
            mask, _, _ = sam2_predictor.predict(
                point_coords= point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=False,
            )
            mask = torch.from_numpy(mask).to(get_device())
            torchvision.utils.save_image(mask.unsqueeze(0).to(torch.float16), f"{self.save_mask_tmp}/mask_{cam.image_name}" + ".png")
            self.gaussian.apply_weights(
                cur_cam, weights, weights_cnt, mask.to(torch.float32)
            )
            masks.append(mask)

        weights /= weights_cnt + 1e-7
        selected_mask = weights > self.mask_thres
        selected_mask = selected_mask[:, 0]
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)
        del sam2_predictor
        gc.collect()    
        torch.cuda.empty_cache()

        return masks, selected_mask
    
    def update_sam2_mask_with_point_prompt(
        self, edit_cameras, positive_sam_points=None, negative_sam_points=None, type = "default"
    ):
        os.makedirs(self.save_mask_tmp, exist_ok=True)
        os.system(f"rm -rf {self.save_mask_tmp}/*")
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        sam2_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
        render_folder = os.path.join(os.path.dirname(self.save_mask_tmp), "render")
        for i, cam in tqdm(enumerate(edit_cameras)):
            cur_cam = cam
            if type == "default":
                img = render(cur_cam, self.gaussian, self.pipe, self.background_tensor)["render"]
            else:
                img = render(cur_cam, self.gaussian2, self.pipe, self.background_tensor)["render"]
            save_image(img[None], f"{render_folder}/{i+1:05d}" + ".jpg")

        frame_names = [
            p for p in os.listdir(render_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]

        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        state = sam2_predictor.init_state(video_path=render_folder)
        sam2_predictor.reset_state(state)

        ann_frame_idx = 0  
        ann_obj_id = 1  

        positive_points2ds = np.empty((0,2)) if positive_sam_points.shape[0] == 0 else positive_sam_points
        negative_points2ds = np.empty((0,2)) if negative_sam_points.shape[0] == 0 else negative_sam_points
        positive_label = np.empty((0), dtype=np.int64) if positive_sam_points.shape[0] == 0 else np.array([1] * positive_sam_points.shape[0], dtype=np.int64) 
        negative_label = np.empty((0), dtype=np.int64) if negative_sam_points.shape[0] == 0 else np.array([0] * negative_sam_points.shape[0], dtype=np.int64)
        
        point_coords = np.concatenate((positive_points2ds, negative_points2ds), axis=0)
        point_labels = np.concatenate((positive_label, negative_label), axis=0) 

        sam2_predictor.add_new_points(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=point_coords,
            labels=point_labels,
        )

        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        for out_frame_idx, _, out_mask_logits in sam2_predictor.propagate_in_video(state):
            if out_frame_idx == 0:
                continue
            mask = out_mask_logits[0] > 0.0
            cur_cam = edit_cameras[out_frame_idx-1]
            save_image(mask.unsqueeze(0).to(torch.float16), f"{self.save_mask_tmp}/mask_{cur_cam.image_name}" + ".png")
            self.gaussian.apply_weights(
                cur_cam, weights, weights_cnt, mask.to(torch.float32)
            )
            masks.append(mask.to(torch.float16))

        weights /= weights_cnt + 1e-7
        selected_mask = weights > self.mask_thres
        selected_mask = selected_mask[:, 0]
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)
        del sam2_predictor,state
        gc.collect()    
        torch.cuda.empty_cache()

        return masks, selected_mask

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--optimizer_type", type=str, default="sparse_adam", help="default or sparse_adam")
    parser.add_argument("--sam_option", type=int, default=-1, help="Sam Option.")
    parser.add_argument("--seg_prompt", type=str, default="face", help="seg Prompt.")
    parser.add_argument("--positive_sam_points", type=str, default="/", help="the path of the positive sam points.")
    parser.add_argument("--negative_sam_points", type=str, default="/", help="the path of the negative sam points.")
    parser.add_argument("--camera", type=str, default="tmd_delete/camera.pkl", help="camera.")
    parser.add_argument("--mask_thres", type=float, default=0.5, help="mask threshold.")

    args = parser.parse_args()
    if args.gs_source.endswith(".ply"):
        trainer = MaskCatcher(args)
        trainer.gaussian.update_anchor_term(
            anchor_weight_init_g0=0.05,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=1.3,
        )
        # trainer.configure_optimizers()
        trainer.get_mask(sam_option=args.sam_option, seg_prompt=args.seg_prompt)