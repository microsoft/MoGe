<div align="center">

# MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision

<a href="https://arxiv.org/abs/2410.19115"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://wangrc.site/MoGePage/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/Ruicheng/MoGe'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>

</div>

<img src="./assets/overview_simplified.png" width="100%" alt="Method overview" align="center">

MoGe is a powerful model for recovering 3D geometry from monocular open-domain images. The model consists of a ViT encoder and a convolutional decoder. It directly predicts an affine-invariant point map as well as a mask that excludes regions with undefined geometry (e.g., sky), from which the camera shift, camera focal length and depth map can be further derived. 

***Check our [website](https://wangrc.site/MoGePage) for videos and interactive results!***

## Features

* **Accurately** estimate 3D geometry in point map or mesh format from a **single** image.
* Support various image resolutions and aspect ratios, ranging from **2:1** to **1:2**.
* Capable of producing an extensive depth range, with distances from nearest to farthest reaching up to **1000x**.
* **Fast** inference, typically **0.2s** for a single image on an A100 or RTX 3090 GPU.

## TODO List

- [x] Release inference code & ViT-Large model.
- [ ] Release ViT-Base and ViT-Giant models.
- [ ] Release evaluation and training code.

🌟*Updated on 2024/11/28* - [CHANGELOG](CHANGELOG.md): 
  * Supported user-provided camera FOV. 
  * Added the script for panorama images [scripts/infer_panorama.py](scripts/infer_panorama.py).

## Usage

### Prerequisite

- Clone this repository. 

    ```bash
    git clone https://github.com/microsoft/MoGe.git
    cd MoGe
    ```

- Python (>= 3.10) environment:
  - torch (>= 2.0) and torchvision (compatible with the torch version).
  - other requirements
    ```bash
    pip install -r requirements.txt
    ```
    MoGe should be compatible with most requirements versions. Please check the `requirements.txt` for more details if you have concerns.

### Pretrained model

The ViT-Large model has been uploaded to Hugging Face hub at [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl). 
You may load the model via `MoGeModel.from_pretrained("Ruicheng/moge-vitl")` without manually downloading.

If loading the model from a local file is preferred, you may manually download the model from the huggingface hub and load it via `MoGeModel.from_pretrained("PATH_TO_LOCAL_MODEL.pt")`.

### Minimal example 

Here is a minimal example for loading the model and inferring on a single image. 

```python
import cv2
import torch
from moge.model import MoGeModel

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             

# Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
input_image = cv2.cvtColor(cv2.imread("PATH_TO_IMAGE.jpg"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer 
output = model.infer(input_image)
# `output` has keys "points", "depth", "mask" and "intrinsics",
# The maps are in the same size as the input image. 
# {
#     "points": (H, W, 3),    # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
#     "depth": (H, W),        # scale-invariant depth map
#     "mask": (H, W),         # a binary mask for valid pixels. 
#     "intrinsics": (3, 3),   # normalized camera intrinsics
# }
# For more usage details, see the `MoGeModel.infer` docstring.
```

### Web demo

Make sure that `gradio` is installed and then run the following command to start the web demo:
 
```bash
python scripts/app.py   # --share for Gradio public sharing
```

The web demo is also available at our [Hugging Face space](https://huggingface.co/spaces/Ruicheng/MoGe).


### Using `scripts/infer.py`

Run the script `scripts/infer.py` via the following command:

```bash
# Save the output [maps], [glb] and [ply] files
python scripts/infer.py --input IMAGES_FOLDER_OR_IMAGE_PATH --output OUTPUT_FOLDER --maps --glb --ply

# Show the result in a window (requires pyglet < 2.0, e.g. pip install pyglet==1.5.29)
python scripts/infer.py --input IMAGES_FOLDER_OR_IMAGE_PATH --output OUTPUT_FOLDER --show
```

For detailed options, run `python scripts/infer.py --help`:

```
Usage: infer.py [OPTIONS]

  Inference script for the MoGe model.

Options:
  --input PATH                Input image or folder path. "jpg" and "png" are
                              supported.
  --fov_x FLOAT               If camera parameters are known, set the
                              horizontal field of view in degrees. Otherwise,
                              MoGe will estimate it.
  --output PATH               Output folder path
  --pretrained TEXT           Pretrained model name or path. Default is
                              "Ruicheng/moge-vitl"
  --device TEXT               Device name (e.g. "cuda", "cuda:0", "cpu").
                              Default is "cuda"
  --resize INTEGER            Resize the image(s) & output maps to a specific
                              size. Default is None (no resizing).
  --resolution_level INTEGER  An integer [0-9] for the resolution level of
                              inference. The higher, the better but slower.
                              Default is 9. Note that it is irrelevant to the
                              output resolution.
  --threshold FLOAT           Threshold for removing edges. Default is 0.03.
                              Smaller value removes more edges. "inf" means no
                              thresholding.
  --maps                      Whether to save the output maps and fov(image,
                              depth, mask, points, fov).
  --glb                       Whether to save the output as a.glb file. The
                              color will be saved as a texture.
  --ply                       Whether to save the output as a.ply file. The
                              color will be saved as vertex colors.
  --show                      Whether show the output in a window. Note that
                              this requires pyglet<2 installed as required by
                              trimesh.
  --help                      Show this message and exit.
```

### Using `scripts/infer_panorama.py` for 360-degree panorama images

> *NOTE: This is an experimental extension of MoGe.*

The script will split the 360-degree panorama image into multiple perspective views and infer on each view separately. 
The output maps will be combined to produce a panorama depth map and point map. 

Note that the panorama image must have spherical parameterization (e.g., environment maps or equirectangular images). Other formats must be converted to spherical format before using this script. Run `python scripts/infer_panorama.py --help` for detailed options.


<div align="center">
  <img src="./assets/panorama_pipeline.png" width="80%">

The photo is from [this URL](https://commons.wikimedia.org/wiki/Category:360%C2%B0_panoramas_with_equirectangular_projection#/media/File:Braunschweig_Sankt-%C3%84gidien_Panorama_02.jpg)
</div>

## License

MoGe code is released under the MIT license, except for DINOv2 code in `moge/model/dinov2` which is released by Meta AI under the Apache 2.0 license. 
See [LICENSE](LICENSE) for more details.


## Citation

If you find our work useful in your research, we gratefully request that you consider citing our paper:

```
@misc{wang2024moge,
    title={MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision},
    author={Wang, Ruicheng and Xu, Sicheng and Dai, Cassie and Xiang, Jianfeng and Deng, Yu and Tong, Xin and Yang, Jiaolong},
    year={2024},
    eprint={2410.19115},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2410.19115}, 
}
```
