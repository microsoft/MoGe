# MoGe: Accurate Monocular Geometry Estimation

MoGe is a powerful model for recovering 3D geometry from monocular open-domain images, including metric point maps, metric depth maps, normal maps and camera FOV. ***Check our websites ([MoGe-1](https://wangrc.site/MoGePage), [MoGe-2](https://wangrc.site/MoGe2Page), [MoGe-3](https://qft-333.github.io/moge3page/)) for videos and interactive results!***

## 📖 Publications

### MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement
<div align="center">
  <a href="https://arxiv.org/abs/2607.17967"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://qft-333.github.io/moge3page/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</div>

https://github.com/user-attachments/assets/50e908f1-dd63-4182-8ad8-788f7b56ef2c

### MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details

<div align="center">
  <a href="https://arxiv.org/abs/2507.02546"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://wangrc.site/MoGe2Page/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/spaces/Ruicheng/MoGe-2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo_(MoGe_v2)-blue'></a>
</div>

<details><summary>Video</summary>
https://github.com/user-attachments/assets/8f9ae680-659d-4f7f-82e2-b9ed9d6b988a
</details>

### MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision

<div align="center">
  <a href="https://arxiv.org/abs/2410.19115"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://wangrc.site/MoGePage/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/spaces/Ruicheng/MoGe'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo_(MoGe_v1)-blue'></a>
</div>

<details><summary>Overview</summary>
<img src="./assets/overview_simplified.png" width="100%" alt="Method overview" align="center">
</details>

## 🌟 Features

* **Accurate 3D geometry estimation**: Estimate point maps & depth maps & [normal maps](docs/normal.md) from open-domain single images with high precision -- all capabilities in one model, one forward pass.
* **Optional ground-truth FOV input**: Enhance model accuracy further by providing the true field of view.
* **Flexible resolution support**: Works seamlessly with various resolutions and aspect ratios, from 2:1 to 1:2.
* **Optimized for speed**: Achieves 60ms latency per image (A100 or RTX3090, FP16, ViT-L). Adjustable inference resolution for even faster speed.

## ✨ News

***(2026-07-21)***
* ❗Published **MoGe-3**, featuring significantly improved **fine-grained point map geometry**.
  * _The new code and pretrained models are coming soon. Please stay tuned!_

## 📦 Installation

### Install via pip
  
```bash
pip install git+https://github.com/microsoft/MoGe.git
```

### Or clone this repository

```bash
git clone https://github.com/microsoft/MoGe.git
cd MoGe
pip install -r requirements.txt   # install the requirements
```

Note: MoGe should be compatible with most requirements versions. Please check the `requirements.txt` for more details if you encounter any dependency issues.

## 🤗 Pretrained Models

Our pretrained models are available on the huggingface hub:

<table>
  <thead>
    <tr>
      <th>Version</th>
      <th>Hugging Face Model</th>
      <th>Metric scale</th>
      <th>Normal</th>
      <th>#Params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MoGe-1</td>
      <td><a href="https://huggingface.co/Ruicheng/moge-vitl" target="_blank"><code>Ruicheng/moge-vitl</code><a></td>
      <td>-</td>
      <td>-</td>
      <td>314M</td>
    </tr>
    <tr>
      <td rowspan="4">MoGe-2</td>
      <td><a href="https://huggingface.co/Ruicheng/moge-2-vitl" target="_blank"><code>Ruicheng/moge-2-vitl</code></a></td>
      <td>✅</td>
      <td>-</td>
      <td>326M</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Ruicheng/moge-2-vitl-normal" target="_blank"><code>Ruicheng/moge-2-vitl-normal</code></a></td>
      <td>✅</td>
      <td>✅</td>
      <td>331M</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Ruicheng/moge-2-vitb-normal" target="_blank"><code>Ruicheng/moge-2-vitb-normal</code></a></td>
      <td>✅</td>
      <td>✅</td>
      <td>104M</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Ruicheng/moge-2-vits-normal" target="_blank"><code>Ruicheng/moge-2-vits-normal</code></a></td>
      <td>✅</td>
      <td>✅</td>
      <td>35M</td>
    </tr>
    <tr>
      <td rowspan="2">MoGe-3</td>
      <td>moge-3-vitg (coming soon)</td>
      <td>✅</td>
      <td>✅</td>
      <td>-</td>
    </tr>
    <tr>
      <td>moge-3-vitl (coming soon)</td>
      <td>✅</td>
      <td>✅</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

You may import the `MoGeModel` class of the matched version, then load the pretrained weights via `MoGeModel.from_pretrained("HUGGING_FACE_MODEL_REPO_NAME")` with automatic downloading.
If loading a local checkpoint, replace the model name with the local path.

For ONNX support, please refer to [docs/onnx.md](docs/onnx.md).

## 💡 Minimal Code Example 

Here is a minimal example for loading the model and inferring on a single image. 

```python
import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)                             

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread("PATH_TO_IMAGE.jpg"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer 
output = model.infer(input_image)
"""
`output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
The maps are in the same size as the input image. 
{
    "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    "depth": (H, W),        # depth map
    "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
    "mask": (H, W),         # a binary mask for valid pixels. 
    "intrinsics": (3, 3),   # normalized camera intrinsics
}
"""
```
For more usage details, see the `MoGeModel.infer()` docstring.

## 💡 Usage

### Gradio demo | `moge app`

> The demo for MoGe-1 is also available at our [Hugging Face Space](https://huggingface.co/spaces/Ruicheng/MoGe).

```bash
# Using the command line tool
moge app        # will run MoGe-2 demo by default.

# In this repo
python moge/scripts/app.py   # --share for Gradio public sharing
```

See also [`moge/scripts/app.py`](moge/scripts/app.py) 


### Inference | `moge infer`

Run the script `moge/scripts/infer.py` via the following command:

```bash
# Save the output [maps], [glb] and [ply] files
moge infer -i IMAGES_FOLDER_OR_IMAGE_PATH --o OUTPUT_FOLDER --maps --glb --ply

# Show the result in a window (requires pyglet < 2.0, e.g. pip install pyglet==1.5.29)
moge infer -i IMAGES_FOLDER_OR_IMAGE_PATH --o OUTPUT_FOLDER --show
```

For detailed options, run `moge infer --help`:

```
Usage: moge infer [OPTIONS]

  Inference script

Options:
  -i, --input PATH            Input image or folder path. "jpg" and "png" are
                              supported.
  --fov_x FLOAT               If camera parameters are known, set the
                              horizontal field of view in degrees. Otherwise,
                              MoGe will estimate it.
  -o, --output PATH           Output folder path
  --pretrained TEXT           Pretrained model name or path. If not provided,
                              the corresponding default model will be chosen.
  --version [v1|v2]           Model version. Defaults to "v2"
  --device TEXT               Device name (e.g. "cuda", "cuda:0", "cpu").
                              Defaults to "cuda"
  --fp16                      Use fp16 precision for much faster inference.
  --resize INTEGER            Resize the image(s) & output maps to a specific
                              size. Defaults to None (no resizing).
  --resolution_level INTEGER  An integer [0-9] for the resolution level for
                              inference. Higher value means more tokens and
                              the finer details will be captured, but
                              inference can be slower. Defaults to 9. Note
                              that it is irrelevant to the output size, which
                              is always the same as the input size.
                              `resolution_level` actually controls
                              `num_tokens`. See `num_tokens` for more details.
  --num_tokens INTEGER        number of tokens used for inference. A integer
                              in the (suggested) range of `[1200, 2500]`.
                              `resolution_level` will be ignored if
                              `num_tokens` is provided. Default: None
  --threshold FLOAT           Threshold for removing edges. Defaults to 0.01.
                              Smaller value removes more edges. "inf" means no
                              thresholding.
  --maps                      Whether to save the output maps (image, point
                              map, depth map, normal map, mask) and fov.
  --glb                       Whether to save the output as a.glb file. The
                              color will be saved as a texture.
  --ply                       Whether to save the output as a.ply file. The
                              color will be saved as vertex colors.
  --show                      Whether show the output in a window. Note that
                              this requires pyglet<2 installed as required by
                              trimesh.
  --help                      Show this message and exit.
```

See also [`moge/scripts/infer.py`](moge/scripts/infer.py)

### 360° panorama images | `moge infer_panorama` 

> *NOTE: This is an experimental extension of MoGe.*

The script will split the 360-degree panorama image into multiple perspective views and infer on each view separately. 
The output maps will be combined to produce a panorama depth map and point map. 

Note that the panorama image must have spherical parameterization (e.g., environment maps or equirectangular images). Other formats must be converted to spherical format before using this script. Run `moge infer_panorama --help` for detailed options.


<div align="center">
  <img src="./assets/panorama_pipeline.png" width="80%">

The photo is from [this URL](https://commons.wikimedia.org/wiki/Category:360%C2%B0_panoramas_with_equirectangular_projection#/media/File:Braunschweig_Sankt-%C3%84gidien_Panorama_02.jpg)
</div>

See also [`moge/scripts/infer_panorama.py`](moge/scripts/infer_panorama.py)

## 🏋️‍♂️ Training & Finetuning

See [docs/train.md](docs/train.md)

## 🧪 Evaluation

See [docs/eval.md](docs/eval.md)

## ⚖️ License

MoGe code is released under the MIT license, except for DINOv2 code in `moge/model/dinov2` which is released by Meta AI under the Apache 2.0 license. 
See [LICENSE](LICENSE) for more details.


## 📜 Citation

If you find our work useful in your research, we gratefully request that you consider citing our paper:

```
@inproceedings{wang2025moge,
  title={Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision},
  author={Wang, Ruicheng and Xu, Sicheng and Dai, Cassie and Xiang, Jianfeng and Deng, Yu and Tong, Xin and Yang, Jiaolong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5261--5271},
  year={2025}
}

@misc{wang2025moge2,
      title={MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details}, 
      author={Ruicheng Wang and Sicheng Xu and Yue Dong and Yu Deng and Jianfeng Xiang and Zelong Lv and Guangzhong Sun and Xin Tong and Jiaolong Yang},
      year={2025},
      eprint={2507.02546},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.02546}, 
}
```
