<p align="center">
  <img src="resources/assets/logo.png" alt="100Editor Logo" width="100"/>
  <br>
</p>
  
<h3 align="center"><strong>[CVPR 2026(Findings)] 100Editor: 100+ Views per Batch and Minute-Scale View-Consistent 3D Editing</strong></h3>

<p align="center">
    <a href="https://github.com/Devin100086">Cunqi Wu</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=Hv0M87UAAAAJ">Peng Zhou</a><sup>†,1</sup>,</span>
    <a href="https://scholar.google.com/citations?user=mhPGcuwAAAAJ&hl=zh-CN">Jie Qin</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=61b6eYkAAAAJ">Qi Tian</a><sup>2</sup>,
    <br>
    <sup>†</sup>Corresponding author.
    <br>
    <sup>1</sup>Nanjing University of Aeronautics and Astronautics,
    <br>
    <sup>2</sup>Huawei Inc.
    <br>
</p>

<div align="center">

 <a href='https://arxiv.org/abs/2311.14521'><img src='https://img.shields.io/badge/arXiv-2311.14521-b31b1b.svg?style=for-the-badge'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://devin100086.github.io/100Editor/'><img src='https://img.shields.io/badge/Project-Page-Green?style=for-the-badge'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://www.youtube.com/watch?v=TdZIICSFqsU&ab_channel=YiwenChen'><img src='https://img.shields.io/badge/Bilibili-00A1D6?logo=bilibili&logoColor=white&style=for-the-badge'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='你的链接'><img src='https://img.shields.io/badge/Software-Document-8A2BE2.svg?style=for-the-badge'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</div>

## :loudspeaker:News

 **[2026-04]** :fire: We release the code of 100Editor and the 3D editing software of 100Editor！
 
 **[2026-02]** :tada: 100Editor has been accepted to CVPR 2026(Findings)! Code coming soon!
 
## :wrench: Installation
> Our environment has been tested on an NVIDIA RTX 4090 GPU with Ubuntu 22.04 and CUDA 12.4.
1. Clone our repository
```
git clone https://github.com/Devin100086/100Editor.git
```
2. create environment and install dependencies
```
# Create an environment
conda create -n 100Editor python=3.11

# Install dependencies
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e third_party/BrushNet
pip install -e third_party/LeftRefill
pip install -e third_party/sam2

pip install -e src/trainer/origin/submodules/diff-gaussian-rasterization
pip install -e src/editor/gaussiansplatting/submodules/acc-diff-gaussian-rasterization-editor
pip install -e third_party/gaussiansplatting/submodules/diff-gaussian-rasterization
pip install -e third_party/dreamgaussian/add_diff-gaussian-rasterization
pip install -e src/trainer/origin/submodules/fused-ssim
pip install -e src/trainer/origin/submodules/simple-knn
```
3. 下载所需的模型权重
```
sh ./scripts/download.sh
```
## :fire:Train 3DGS
### 脚本训练
用户可以通过执行以下命令实现对目标3dgs场景的重建
```bash
bash scripts/train/train_3dgs.sh \
  [--use-depth-loss] \
  [--use-appearance-embedding] \
  <scene_path> \
  <output_dir> \
  <checkpoint_iter>
  
Example (默认关闭外观嵌入和 depth loss):
bash scripts/train/train_3dgs.sh \
  /path/to/scene \
  output/scene_name \
  7000

Example (开启外观嵌入 + depth loss，和 GUI 对齐):
bash scripts/train/train_3dgs.sh \
  --use-depth-loss \
  --use-appearance-embedding \
  /path/to/scene \
  output/scene_name \
  7000
```
### GUI中训练3DGS
我们的软件同样可以去支持3dgs训练，并且其中可以支持gsplat库进行训练，并且提供多种训练策略选择，更多关于这部分的使用可以见这里。
## :art: Editing
[API Guide](https://help.aliyun.com/zh/model-studio/get-api-key),
[API Platform](https://bailian.console.aliyun.com)
## :pencil: Evaluation



## :pray: Acknowledgments
We sincerely appreciate these excellent open-source projects.

<center>
<table>
  <tr>
    <th>Project</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>Gaussian Splatting</td>
    <td><a href="https://github.com/graphdeco-inria/gaussian-splatting">graphdeco-inria/gaussian-splatting</a></td>
  </tr>
  <tr>
    <td>GaussianEditor</td>
    <td><a href="https://github.com/buaacyw/GaussianEditor">buaacyw/GaussianEditor</a></td>
  </tr>
  <tr>
    <td>DGE</td>
    <td><a href="https://github.com/silent-chen/DGE">silent-chen/DGE</a></td>
  </tr>
  <tr>
    <td>InstructPix2Pix</td>
    <td><a href="https://github.com/timothybrooks/instruct-pix2pix">timothybrooks/instruct-pix2pix</a></td>
  </tr>
  <tr>
    <td>BrushEdit</td>
    <td><a href="https://github.com/TencentARC/BrushEdit">TencentARC/BrushEdit</a></td>
  </tr>
  <tr>
    <td>dreamgaussian</td>
    <td><a href="https://github.com/dreamgaussian/dreamgaussian">dreamgaussian/dreamgaussian</a></td>
  </tr>
</table>
</center>

## :pushpin: Citation
```bibtex
11
``` 

---
<div align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/refs/heads/master/Emojis/Smilies/Beaming%20Face%20with%20Smiling%20Eyes.png" alt="Thanks" width="50" height="50" />

**We hope that 100Editor provides you with a distinctive editing experience！**

**If you find this repository helpful, please give it a star ⭐**
</div>
