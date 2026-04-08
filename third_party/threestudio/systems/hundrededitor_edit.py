from dataclasses import dataclass, field
import os
import random
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import threestudio
from torchvision.utils import save_image
from tqdm import tqdm

from threestudio.utils.clip_metrics import ClipSimilarity
from threestudio.utils.typing import *

from .hundrededitor import HundredEditor

@threestudio.register("100editor-system-edit")
class HundredEditorEdit(HundredEditor):
    @dataclass
    class Config(HundredEditor.Config):
        local_edit: bool = False

        seg_prompt: str = ""

        second_guidance_type: str = "dds"
        second_guidance: dict = field(default_factory=dict)
        dds_target_prompt_processor: dict = field(default_factory=dict)
        dds_source_prompt_processor: dict = field(default_factory=dict)

        clip_prompt_origin: str = ""
        clip_prompt_target: str = ""  # only for metrics

        use_masked_image: bool = False
        batch: bool = False
        camera_update_per_step: int = 500
        added_noise_schedule: List[int] = field(default_factory=[999, 200, 200, 21])

        enable_cps: bool = False
        cps_patience_counter: int = 0
        cps_patience: int = 4
        cps_batch_count: int = 3
        cps_eval_interval: int = 20
        cps_min_delta: float = 0.006
        cps_num_test_views: int = 6

    cfg: Config

    def configure(self) -> None:
        super().configure()
        cache_root = Path(__file__).resolve().parents[3] / "runtime" / "cache" / "edit"
        cache_root.mkdir(parents=True, exist_ok=True)
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = (cache_root / self.cfg.cache_dir).as_posix()
        else:
            self.cache_dir = (cache_root / "edit-n2n").as_posix()
        self.best_metric = float("-inf")
        self.initial_patience_counter = 0
        self.patience_counter = 0
        self.pending_batch_refresh = False
        self.batch_done_count = 0
        self.test_view_indices = []
        self.train_view_pool = []
        self.cps_clip_metrics = None
        self._warned_missing_cps_prompt = False

    def _refresh_train_views(self, random_seed: int, use_train_pool: bool) -> None:
        train_dataset = self.trainer.datamodule.train_dataset

        if use_train_pool and len(self.train_view_pool) > 0:
            candidate_pool = self.train_view_pool
        else:
            candidate_pool = list(range(train_dataset.total_view_num))

        sample_size = min(len(candidate_pool), train_dataset.cfg.max_view_num)
        if sample_size <= 0:
            return

        rng = random.Random(random_seed)
        sampled_views = rng.sample(candidate_pool, sample_size)
        train_dataset.n2n_view_index = sampled_views
        train_dataset.view_index_stack = sampled_views.copy()

        self.view_list = sampled_views
        sorted_train_view_list = sorted(self.view_list)
        if len(sorted_train_view_list) == 0:
            return

        selected_views = torch.linspace(
            0,
            len(sorted_train_view_list) - 1,
            self.trainer.datamodule.val_dataset.n_views,
            dtype=torch.int,
        )
        self.trainer.datamodule.val_dataset.selected_views = [
            sorted_train_view_list[int(idx)] for idx in selected_views
        ]

    def _init_cps_state(self) -> None:
        train_dataset = self.trainer.datamodule.train_dataset
        total_view_num = train_dataset.total_view_num

        self.initial_patience_counter = max(0, int(self.cfg.cps_patience_counter))
        self.patience_counter = self.initial_patience_counter
        self.best_metric = float("-inf")
        self.pending_batch_refresh = False
        self.batch_done_count = 0
        self.test_view_indices = []
        self.train_view_pool = list(range(total_view_num))
        self._warned_missing_cps_prompt = False

        if self.cfg.batch and self.cfg.enable_cps and total_view_num > 1:
            num_test_views = min(
                max(1, int(self.cfg.cps_num_test_views)),
                total_view_num - 1,
            )
            rng = random.Random(0)
            self.test_view_indices = sorted(rng.sample(self.train_view_pool, num_test_views))
            test_set = set(self.test_view_indices)
            self.train_view_pool = [idx for idx in self.train_view_pool if idx not in test_set]
            self._refresh_train_views(random_seed=0, use_train_pool=True)
        else:
            self.view_list = train_dataset.n2n_view_index

    def _zero_loss(self) -> Dict[str, Tensor]:
        return {"loss": self.gaussian.get_xyz.sum() * 0.0}

    def compute_cps_metric(self) -> Optional[float]:
        if not (self.cfg.batch and self.cfg.enable_cps):
            return None
        if len(self.test_view_indices) == 0:
            return None
        if len(self.cfg.clip_prompt_origin) == 0 or len(self.cfg.clip_prompt_target) == 0:
            if not self._warned_missing_cps_prompt:
                threestudio.warn(
                    "CPS is enabled but clip prompts are empty, skipping CPS metric."
                )
                self._warned_missing_cps_prompt = True
            return None

        total_cos = 0.0
        with torch.no_grad():
            for idx in self.test_view_indices:
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[idx]
                cur_batch = {
                    "index": idx,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out = self(cur_batch)["comp_rgb"]
                _, _, cos_sim, _ = self.cps_clip_metrics(
                    self.origin_frames[idx].permute(0, 3, 1, 2),
                    out.permute(0, 3, 1, 2),
                    self.cfg.clip_prompt_origin,
                    self.cfg.clip_prompt_target,
                )
                total_cos += abs(cos_sim.item())

        avg_cos = total_cos / len(self.test_view_indices)
        self.log("train/cps_metric", avg_cos, on_step=True, on_epoch=False)
        return avg_cos

    def edit_all_view(self, original_render_name, cache_name, update_camera=False, global_step=0):
        # if self.true_global_step >= self.cfg.camera_update_per_step * 2:
        #     self.guidance.use_normal_unet()

        self.edited_cams = []
        if update_camera:
            self._refresh_train_views(
                random_seed=global_step + 1,
                use_train_pool=self.cfg.batch and self.cfg.enable_cps,
            )

        self.edit_frames = {}
        cache_dir = os.path.join(self.cache_dir, cache_name)
        original_render_cache_dir = os.path.join(self.cache_dir, original_render_name)
        os.makedirs(cache_dir, exist_ok=True)

        cameras = []
        images = []
        original_frames = []
        t_max_step = self.cfg.added_noise_schedule
        self.guidance.max_step = t_max_step[
            min(len(t_max_step) - 1, self.true_global_step // self.cfg.camera_update_per_step)
        ]
        with torch.no_grad():
            for id in self.view_list:
                cameras.append(self.trainer.datamodule.train_dataset.scene.cameras[id])
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]
            cams_sorted = [cameras[idx] for idx in sorted_cam_idx]

            for id in view_sorted:
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                original_image_path = os.path.join(original_render_cache_dir, "{:0>4d}.png".format(id))
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out_pkg = self(cur_batch)
                out = out_pkg["comp_rgb"]
                if self.cfg.use_masked_image:
                    out = out * out_pkg["masks"].unsqueeze(-1)
                images.append(out)
                assert os.path.exists(original_image_path)
                cached_image = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)
                self.origin_frames[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]
                original_frames.append(self.origin_frames[id])
            images = torch.cat(images, dim=0)
            original_frames = torch.cat(original_frames, dim=0)

            edited_images = self.guidance(
                images,
                original_frames,
                self.prompt_processor(),
                cams=cams_sorted,
            )
            # save_image(
            #     edited_images["edit_images"].permute(0, 3, 1, 2),
            #     os.path.join(self.cache_dir, f"batch_image_{global_step}.png"),
            #     nrow=4,
            # )
            for view_index_tmp in range(len(self.view_list)):
                self.edit_frames[view_sorted[view_index_tmp]] = (
                    edited_images["edit_images"][view_index_tmp].unsqueeze(0).detach().clone()
                )

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render")

        if len(self.cfg.seg_prompt) > 0:
            self.update_mask()

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if len(self.cfg.dds_target_prompt_processor) > 0:
            self.dds_target_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_target_prompt_processor)
        if len(self.cfg.dds_source_prompt_processor) > 0:
            self.dds_source_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_source_prompt_processor)
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            self.cfg.guidance["video"] = self.cfg.batch
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.loss.lambda_dds > 0:
            self.second_guidance = threestudio.find(self.cfg.second_guidance_type)(
                self.cfg.second_guidance
            )
        self.cps_clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        self._init_cps_state()

    def training_step(self, batch, batch_idx):
        if self.cfg.batch:
            if self.cfg.enable_cps:
                eval_interval = max(1, int(self.cfg.cps_eval_interval))
                if self.true_global_step % eval_interval == 0:
                    metric = self.compute_cps_metric()
                    if metric is not None:
                        if metric > self.best_metric + float(self.cfg.cps_min_delta):
                            self.best_metric = metric
                            self.patience_counter = self.initial_patience_counter
                        else:
                            self.patience_counter += 1
                        self.log(
                            "train/cps_patience_counter",
                            float(self.patience_counter),
                            on_step=True,
                            on_epoch=False,
                        )
                        self.log(
                            "train/cps_best_metric",
                            float(self.best_metric),
                            on_step=True,
                            on_epoch=False,
                        )
                        if self.patience_counter >= max(1, int(self.cfg.cps_patience)):
                            self.patience_counter = self.initial_patience_counter
                            self.pending_batch_refresh = True
                            return self._zero_loss()

                if self.true_global_step == 0 or self.pending_batch_refresh:
                    if self.batch_done_count >= max(1, int(self.cfg.cps_batch_count)):
                        threestudio.info(
                            f"CPS reached batch limit ({self.batch_done_count}), stopping training."
                        )
                        self.trainer.should_stop = True
                        return self._zero_loss()

                    self.edit_all_view(
                        original_render_name="origin_render",
                        cache_name="edited_views",
                        update_camera=self.true_global_step >= self.cfg.camera_update_per_step,
                        global_step=self.true_global_step,
                    )
                    self.pending_batch_refresh = False
                    self.batch_done_count += 1
                    self.log(
                        "train/cps_batch_done_count",
                        float(self.batch_done_count),
                        on_step=True,
                        on_epoch=False,
                    )
            elif self.true_global_step % self.cfg.camera_update_per_step == 0:
                self.edit_all_view(
                    original_render_name="origin_render",
                    cache_name="edited_views",
                    update_camera=self.true_global_step >= self.cfg.camera_update_per_step,
                    global_step=self.true_global_step,
                )

        self.gaussian.update_learning_rate(self.true_global_step)

        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]
        if self.cfg.batch:
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames:
                    batch_index[img_index] = self.view_list[img_index]
        out = self(batch, local=self.cfg.local_edit)


        images = out["comp_rgb"]

        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames or (
                    self.cfg.per_editing_step > 0
                    and self.cfg.edit_begin_step
                    < self.global_step
                    < self.cfg.edit_until_step
                    and self.global_step % self.cfg.per_editing_step == 0 
                ):
                    result = self.guidance(
                        images[img_index][None],
                        self.origin_frames[cur_index],
                        prompt_utils,
                    )

                    self.edit_frames[cur_index] = result["edit_images"].detach().clone()

                    # print("edited image index", cur_index)

                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)

            guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                "loss_p": self.perceptual_loss(
                    images.permute(0, 3, 1, 2).contiguous(),
                    gt_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # dds loss
        if self.cfg.loss.lambda_dds > 0:
            dds_target_prompt_utils = self.dds_target_prompt_processor()
            dds_source_prompt_utils = self.dds_source_prompt_processor()

            second_guidance_out = self.second_guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                dds_target_prompt_utils,
                dds_source_prompt_utils,
            )
            for name, value in second_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        if (
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.cfg.clip_prompt_target) > 0:
            self.compute_clip()

    def compute_clip(self):
        clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        total_cos = 0
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out = self(cur_batch)["comp_rgb"]
                _, _, cos_sim, _ = clip_metrics(self.origin_frames[id].permute(0, 3, 1, 2), out.permute(0, 3, 1, 2),
                                                self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target)
                total_cos += abs(cos_sim.item())
        print(self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target, total_cos / len(self.view_list))
        self.log("train/clip_sim", total_cos / len(self.view_list))

    def gaussian_blur(self, mask, kernel_size=21, sigma=1.0):

        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        x = torch.exp(-x**2 / (2 * sigma**2))
        kernel1d = x / x.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.expand(mask.size(1), 1, kernel_size, kernel_size)
        kernel2d = kernel2d.to(mask.device)

        blurred_mask = F.conv2d(mask, kernel2d, padding=kernel_size // 2, groups=mask.size(1))
        return blurred_mask
