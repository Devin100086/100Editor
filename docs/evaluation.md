## Evaluation
All runnable scripts under `scripts/train`, `scripts/eval`, and `scripts/render` are listed below.

You can append `--dry-run` to any script to preview the resolved command without actually executing it.

### scripts/train
Train 3DGS:
```bash
bash scripts/train/train_3dgs.sh \
  <scene_path> \
  <output_dir> \
  <checkpoint_iter>
```
Example:
```bash
bash scripts/train/train_3dgs.sh \
  /path/to/scene \
  output/scene_name \
  7000
```

### scripts/render
Render train/test views from a trained 3DGS model:
```bash
bash scripts/render/render_3dgs.sh \
  <scene_path> \
  <model_path> \
  <iteration(-1=latest)> <skip_train(0|1)> <skip_test(0|1)>
```
Parameter meaning:
- `-1` (`iteration`): use latest available iteration under model path; use values like `7000` to render a specific iteration.
- `0` (`skip_train`): `0` means render train views, `1` means skip train views.
- `0` (`skip_test`): `0` means render test views, `1` means skip test views.

Render edited 3DGS result to image folder:
```bash
bash scripts/render/render_edit_3dgs.sh \
  <edited_gs_source.ply> \
  <colmap_dir> \
  <save_dir> \
  <use_original_resolution(0|1)>
```
Parameter meaning:
- `1` (`use_original_resolution`): `1` means original resolution, `0` means fixed `512x512`.

Render edited 3DGS result to video:
```bash
bash scripts/render/render_video.sh \
  <edited_gs_source.ply> \
  <colmap_dir> \
  <output_video.mp4> \
  <use_original_resolution(0|1)> <render_path_mode(0|1)>
```
Parameter meaning:
- `1` (`use_original_resolution`): `1` means original resolution, `0` means fixed `512x512`.
- `0` (`render_path_mode`): `0` means spiral camera trajectory, `1` means path mode (`--render_path True`).

### scripts/eval
Evaluate edited image folder with CLIP:
```bash
bash scripts/eval/eval_image.sh \
  <origin_image_dir> \
  <edited_image_dir> \
  "<origin_prompt>" \
  "<target_prompt>"
```

Evaluate origin vs edited PLY with CLIP:
```bash
bash scripts/eval/eval_ply.sh \
  <origin_gs_source.ply> \
  <edited_gs_source.ply> \
  <colmap_dir> \
  "<origin_prompt>" \
  "<target_prompt>" \
  <use_original_resolution(0|1)>
```
Parameter meaning:
- `1` (`use_original_resolution`): `1` means original resolution, `0` means fixed `512x512`.

Evaluate temporal consistency (MEt3R) on rendered frames:
```bash
bash scripts/eval/eval_met3r.sh \
  <image_dir> \
  <distance(cosine|lpips|rmse|psnr|mse|ssim)> \
  <img_size(0=original)>
```
Parameter meaning:
- `cosine` (`distance`): metric type, e.g. `cosine`, `lpips`, `rmse`, `psnr`, `mse`, `ssim`.
- `256` (`img_size`): resize images to `256x256`; use `0` to keep original resolution.