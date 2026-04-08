## Render & Evaluation
All runnable scripts under `scripts/train`, `scripts/eval`, and `scripts/render` are listed below.

You can append `--dry-run` to any script to preview the resolved command without actually executing it.

### render
Render train/test views from a trained 3DGS model:
```bash
bash scripts/render/render_3dgs.sh \
  <scene_path> \
  <model_path> \
  <iteration(-1=latest)> <skip_train(0|1)> <skip_test(0|1)>
```
**Arguments**:

| Argument | Description | Required |
| --- | --- | --- |
| `<scene_path>` | Path to the scene directory. | Yes |
| `<model_path>` | Path to the trained 3DGS model directory. | Yes |
| `<iteration(-1=latest)>` | Iteration to render. Use `-1` to select the latest available iteration. | Yes |
| `<skip_train(0\|1)>` | Whether to skip rendering train views. `0`: render train views; `1`: skip. | Yes |
| `<skip_test(0\|1)>` | Whether to skip rendering test views. `0`: render test views; `1`: skip. | Yes |

Render edited 3DGS result to image folder:
```bash
bash scripts/render/render_edit_3dgs.sh \
  <edited_gs_source.ply> \
  <colmap_dir> \
  <save_dir> \
  <use_original_resolution(0|1)>
```
**Arguments**:

| Argument | Description | Required |
| --- | --- | --- |
| `<edited_gs_source.ply>` | Path to the edited 3DGS PLY file. | Yes |
| `<colmap_dir>` | Path to the COLMAP reconstruction directory. | Yes |
| `<save_dir>` | Output directory for rendered images. | Yes |
| `<use_original_resolution(0\|1)>` | Use original image resolution (`1`) or fixed `512x512` (`0`). | Yes |

Render edited 3DGS result to video:
```bash
bash scripts/render/render_video.sh \
  <edited_gs_source.ply> \
  <colmap_dir> \
  <output_video.mp4> \
  <use_original_resolution(0|1)> <render_path_mode(0|1)>
```
**Arguments**:

| Argument | Description | Required |
| --- | --- | --- |
| `<edited_gs_source.ply>` | Path to the edited 3DGS PLY file. | Yes |
| `<colmap_dir>` | Path to the COLMAP reconstruction directory. | Yes |
| `<output_video.mp4>` | Output path for the rendered video. | Yes |
| `<use_original_resolution(0\|1)>` | Use original image resolution (`1`) or fixed `512x512` (`0`). | Yes |
| `<render_path_mode(0\|1)>` | Camera path mode. `0`: spiral trajectory; `1`: path mode (`--render_path True`). | Yes |

### eval
Evaluate edited image folder with CLIP:
```bash
bash scripts/eval/eval_image.sh \
  <origin_image_dir> \
  <edited_image_dir> \
  "<origin_prompt>" \
  "<target_prompt>"
```
**Arguments**:

| Argument | Description | Required |
| --- | --- | --- |
| `<origin_image_dir>` | Path to the original image directory. | Yes |
| `<edited_image_dir>` | Path to the edited image directory. | Yes |
| `"<origin_prompt>"` | Text prompt describing the original content. | Yes |
| `"<target_prompt>"` | Text prompt describing the target edited content. | Yes |

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
**Arguments**:

| Argument | Description | Required |
| --- | --- | --- |
| `<origin_gs_source.ply>` | Path to the original 3DGS PLY file. | Yes |
| `<edited_gs_source.ply>` | Path to the edited 3DGS PLY file. | Yes |
| `<colmap_dir>` | Path to the COLMAP reconstruction directory. | Yes |
| `"<origin_prompt>"` | Text prompt describing the original content. | Yes |
| `"<target_prompt>"` | Text prompt describing the target edited content. | Yes |
| `<use_original_resolution(0\|1)>` | Use original image resolution (`1`) or fixed `512x512` (`0`). | Yes |

Evaluate temporal consistency (MEt3R) on rendered frames:
```bash
bash scripts/eval/eval_met3r.sh \
  <image_dir> \
  <distance(cosine|lpips|rmse|psnr|mse|ssim)> \
  <img_size(0=original)>
```
**Arguments**:

| Argument | Description | Required |
| --- | --- | --- |
| `<image_dir>` | Path to the rendered image directory. | Yes |
| `<distance(cosine\|lpips\|rmse\|psnr\|mse\|ssim)>` | Distance metric used by MEt3R. | Yes |
| `<img_size(0=original)>` | Resize images to a fixed size (e.g. `256`) or keep original size with `0`. | Yes |
