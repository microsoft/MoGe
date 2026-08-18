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

***(2026-08-18)***
* ❗Released **MoGe-3**, featuring significantly improved **fine-grained point map geometry**.

## 📦 Installation

Requires Python 3.10 or newer. Dependencies are declared in `pyproject.toml`, which works with both [uv](https://docs.astral.sh/uv/) and pip.

> macOS is not supported: MoGe-3 depends on [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM), which builds on Triton, and Triton publishes no macOS wheels.

The following optional extras are available:

| Extra | Contents |
| --- | --- |
| `train` | `accelerate`, `wandb`, `tensorboard`, `mlflow`, … — see [`docs/train.md`](docs/train.md) |

### Using uv (recommended)

```bash
git clone https://github.com/microsoft/MoGe.git
cd MoGe
uv sync                             # inference, all model versions
# uv sync --extra train             # ... plus the training dependencies
```

This creates a `.venv/` and installs MoGe into it in editable mode. Prefix commands with `uv run` (e.g. `uv run moge infer ...`), or activate the environment with `source .venv/bin/activate`.

### Using pip

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

Or from a clone, which is what you want if you intend to edit the code:

```bash
git clone https://github.com/microsoft/MoGe.git
cd MoGe
pip install -e .
```

Extras work the same way here: `pip install -e ".[train]"`.

### Choosing a PyTorch build

With uv there is nothing to choose: `pyproject.toml` pins `torch` and `torchvision` to the **CUDA 13.0** wheel index. To target a different CUDA version, either edit the index URL in `pyproject.toml`, or reinstall PyTorch into the synced environment:

```bash
uv pip install --torch-backend=cu128 torch torchvision --reinstall
# --torch-backend=auto picks a build matching your installed driver
```

pip does not read uv's index configuration, so a plain `pip install` takes whatever PyPI serves. Pass the index you want explicitly:

```bash
pip install -e . --index-url https://download.pytorch.org/whl/cu130
```

Note: MoGe should be compatible with most dependency versions — the bounds in `pyproject.toml` are deliberately loose. Please check them for details if you encounter any dependency issues.

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
      <td><a href="https://huggingface.co/Ruicheng/moge-3-vitg" target="_blank"><code>Ruicheng/moge-3-vitg</code></a></td>
      <td>✅</td>
      <td>✅</td>
      <td>1.25B</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Ruicheng/moge-3-vitl" target="_blank"><code>Ruicheng/moge-3-vitl</code></a></td>
      <td>✅</td>
      <td>✅</td>
      <td>370M</td>
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
# from moge.model.v2 import MoGeModel
from moge.model.v3 import MoGeModel # Let's try MoGe-3

device = torch.device("cuda")

# Load the model
model = MoGeModel.from_pretrained("PATH_TO_CKPT.pt").to(device)

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread("PATH_TO_IMAGE.jpg"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer
# Three refinement steps are applied by default. Set `refine_steps` to change this.
output = model.infer(input_image)
"""
`output` contains the final prediction. Pass `return_per_step=True` to also return every refinement step.
All maps have the same height and width as the input image.
{
  "points": (H, W, 3),                  # final metric point map in OpenCV camera coordinates (x right, y down, z forward)
  "depth": (H, W),                      # final metric depth map
  "intrinsics": (3, 3),                 # normalized camera intrinsics for the final prediction
  "mask": (H, W),                       # binary mask for valid pixels
  "normal": (H, W, 3),                 # normal map in OpenCV camera coordinates (optional)
}
With `return_per_step=True`, `points_per_step`, `depth_per_step`, and `intrinsics_per_step`
contain `refine_steps + 1` entries, including the initial prediction.
"""
```
For more usage details, see the `MoGeModel.infer()` docstring.

## 💡 Usage

### Gradio demo
> The demo for MoGe-1 is available at our [Hugging Face Space](https://huggingface.co/spaces/Ruicheng/MoGe).
> The demo for MoGe-2 is available at our [Hugging Face Space](https://huggingface.co/spaces/Ruicheng/MoGe-2).

```bash
# Using the command line tool
moge app --version v1
moge app --version v2
moge app --version v3 --pretrained PATH_TO_CKPT.pt

# In this repo
python -m moge.scripts.app  # --share for Gradio public sharing
```

See also [`moge/scripts/app.py`](moge/scripts/app.py) 


### Inference | `moge infer`

Run the script `moge/scripts/infer.py` via the following command:

```bash
# Save the output [maps], [glb] and [ply] files
moge infer -i IMAGES_FOLDER_OR_IMAGE_PATH --version v2 --o OUTPUT_FOLDER --maps --glb --ply

# MoGe-3 requires an explicit checkpoint and supports sparse refinement
moge infer -i IMAGES_FOLDER_OR_IMAGE_PATH --version v3 --pretrained PATH_TO_CKPT.pt --refine_steps 3 --o OUTPUT_FOLDER --maps --glb --ply

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
  --pretrained TEXT           Pretrained model name or path. Optional for v1/v2
                              and required for v3.
  --version [v1|v2|v3]        Model version. Defaults to "v3"
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
  --refine_steps INTEGER RANGE
                              Number of sparse refinement steps for v3.
                              Defaults to 3. [x>=0]
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

MoGe code is released under the MIT license, except for DINOv2 code in `moge/model/modules/dinov2` which is released by Meta AI under the Apache 2.0 license. 
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

@misc{kong2026finedetailmonoculargeometryestimation,
      title={Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement},
      author={Lingyu Kong and Ruicheng Li and Ruicheng Wang and Sicheng Xu and Chengtang Yao and Jianfeng Xiang and Jiaolong Yang},
      year={2026},
      eprint={2607.17967},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.17967},
}
```
