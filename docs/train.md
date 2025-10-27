
# Training 

This document provides instructions for training and finetuning the MoGe model.

## Additional Requirements

The following packages other than those listed in [`pyproject.toml`](../pyproject.toml) are required for training and finetuning the MoGe model:

```
accelerate
sympy
mlflow
```

## Data preparation

### Dataset format

Each dataset should be organized as follows:

```
somedataset
├── .index.txt          # A list of instance paths
├── folder1 
│   ├── instance1       # Each instance is in a folder
│   │   ├── image.jpg   # RGB image.
│   │   ├── depth.png   # 16-bit depth. See moge/utils/io.py for details
│   │   ├── meta.json   # Stores "intrinsics" as a 3x3 matrix
│   │   └── ...         # Other componests such as segmentation mask, normal map etc.
...
```

* `.index.txt` is placed at top directory to store a list of instance paths in this dataset. The dataloader will look for instances in this list. You may also use a custom split, e.g. `.train.txt`, `.val.txt` and specify it in the configuration file.

* For depth images, it is recommended to use `read_depth()` and `write_depth()` in [`moge/utils/io.py`](../moge/utils/io.py) to read and write depth images. The depth is stored in logarithmic scale in 16-bit PNG format, offering a balanced precision, dynamic range and compression ratio compared to 16-bit and 32-bit EXR and linear depth formats. It also encodes `NaN` and `Inf` values for invalid depth values.

* The `meta.json` should be a dictionary containing the key `intrinsics`, which are **normalized** camera parameters. You may put more metadata.

* We also support reading and storing segementation masks for evaluation data (see paper evaluation of local points), which are saved in PNG format with semantic labels stored in png metadata as JSON strings. See `read_segmentation()` and `write_segmentation()` in [`moge/utils/io.py`](../moge/utils/io.py) for details.


### Visual inspection

We provide a script to visualize the data and check the data quality. It will export the instance as a PLY file for visualization of point cloud.

```bash
python moge/scripts/vis_data.py PATH_TO_INSTANCE --ply [-o SOMEWHERE_ELSE_TO_SAVE_VIS]
```

### DataLoader

Our training dataloaders is customized to handle loading data, performing perspective crop, and augmentation in a multithreading pipeline. Please refer to [`moge/train/dataloader.py`](../moge/train/dataloader.py) if you have any concern.


## Configuration

See [`configs/train/v1.json`](../configs/train/v1.json) for an example configuration file. The configuration file defines the hyperparameters for training the MoGe model. 
Here is a commented configuration for reference:

```json
{
    "data": {
        "aspect_ratio_range": [0.5, 2.0],               # Range of aspect ratio of sampled images
        "area_range": [250000, 1000000],                # Range of sampled image area in pixels
        "clamp_max_depth": 1000.0,                      # Maximum far/near
        "center_augmentation": 0.5,                     # Ratio of center crop augmentation
        "fov_range_absolute": [1, 179],                 # Absolute range of FOV in degrees
        "fov_range_relative": [0.01, 1.0],              # Relative range of FOV to the original FOV
        "image_augmentation": ["jittering", "jpeg_loss", "blurring"],       # List of image augmentation techniques
        "datasets": [ 
            {
                "name": "TartanAir",                    # Name of the dataset. Name it as you like.
                "path": "data/TartanAir",               # Path to the dataset
                "label_type": "synthetic",              # Label type for this dataset. Losses will be applied accordingly. see "loss" config
                "weight": 4.8,                          # Probability of sampling this dataset
                "index": ".index.txt",                  # File name of the index file.  Defaults to .index.txt
                "depth": "depth.png",                   # File name of depth images. Defaults to depth.png
                "center_augmentation": 0.25,            # Below are dataset-specific hyperparameters. Overriding the global ones above.
                "fov_range_absolute": [30, 150],
                "fov_range_relative": [0.5, 1.0],
                "image_augmentation": ["jittering", "jpeg_loss", "blurring", "shot_noise"]
            }
        ]
    },
    "model_version": "v1",                 # Model version. If you have multiple model variants, you can use this to switch between them.
    "model": {                             # Model hyperparameters. Will be passed to Model __init__() as kwargs.
        "encoder": "dinov2_vitl14",
        "remap_output": "exp",
        "intermediate_layers": 4,
        "dim_upsample": [256, 128, 64],
        "dim_times_res_block_hidden": 2,
        "num_res_blocks": 2,
        "num_tokens_range": [1200, 2500],
        "last_conv_channels": 32,
        "last_conv_size": 1
    },
    "optimizer": {                          # Reflection-like optimizer configurations. See moge.train.utils.py build_optimizer() for details.
        "type": "AdamW",
        "params": [
            {"params": {"include": ["*"], "exclude": ["*backbone.*"]}, "lr": 1e-4},
            {"params": {"include": ["*backbone.*"]}, "lr": 1e-5}
        ]
    },
    "lr_scheduler": {                       # Reflection-like lr_scheduler configurations. See moge.train.utils.py build_lr_scheduler() for details.
        "type": "SequentialLR",
        "params": {
            "schedulers": [
                {"type": "LambdaLR", "params": {"lr_lambda": ["1.0", "max(0.0, min(1.0, (epoch - 1000) / 1000))"]}},
                {"type": "StepLR", "params": {"step_size": 25000, "gamma": 0.5}}
            ],
            "milestones": [2000]
        }
    },
    "low_resolution_training_steps": 50000, # Total number of low-resolution training steps. It makes the early stage training faster. Later stage training on varying size images will be slower.
    "loss": {
        "invalid": {},                      # invalid instance due to runtime error when loading data
        "synthetic": {                      # Below are loss hyperparameters
            "global": {"function": "affine_invariant_global_loss", "weight": 1.0, "params": {"align_resolution": 32}},
            "patch_4": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 4, "align_resolution": 16, "num_patches": 16}},
            "patch_16": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 16, "align_resolution": 8, "num_patches": 256}},
            "patch_64": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 64, "align_resolution": 4, "num_patches": 4096}},
            "normal": {"function": "normal_loss", "weight": 1.0},
            "mask": {"function": "mask_l2_loss", "weight": 1.0}
        },
        "sfm": {
            "global": {"function": "affine_invariant_global_loss", "weight": 1.0, "params": {"align_resolution": 32}},
            "patch_4": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 4, "align_resolution": 16, "num_patches": 16}},
            "patch_16": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 16, "align_resolution": 8, "num_patches": 256}},
            "mask": {"function": "mask_l2_loss", "weight": 1.0}
        },
        "lidar": {
            "global": {"function": "affine_invariant_global_loss", "weight": 1.0, "params": {"align_resolution": 32}},
            "patch_4": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 4, "align_resolution": 16, "num_patches": 16}},
            "mask": {"function": "mask_l2_loss", "weight": 1.0}
        }
    }
}
```

## Run Training 

Launch the training script [`moge/scripts/train.py`](../moge/scripts/train.py). Note that we use [`accelerate`](https://github.com/huggingface/accelerate) for distributed training. 

```bash
accelerate launch \
    --num_processes 8 \
    moge/scripts/train.py \
    --config configs/train/v1.json \
    --workspace workspace/debug \
    --gradient_accumulation_steps 2 \
    --batch_size_forward 2 \
    --checkpoint latest \
    --enable_gradient_checkpointing True \
    --vis_every 1000 \
    --enable_mlflow True
```


## Finetuning

To finetune the pre-trained MoGe model, download the model checkpoint and put it in a local directory, e.g. `pretrained/moge-vitl.pt`.

> NOTE: when finetuning pretrained MoGe model, a much lower learning rate is required. 
The suggested learning rate for finetuning is not greater than 1e-5 for the head and 1e-6 for the backbone. 
And the batch size is recommended to be 32 at least. 
The settings in default configuration are not optimal for specific datasets and may require further tuning.

```bash
accelerate launch \
    --num_processes 8 \
    moge/scripts/train.py \
    --config configs/train/v1.json \
    --workspace workspace/debug \
    --gradient_accumulation_steps 2 \
    --batch_size_forward 2 \
    --checkpoint pretrained/moge-vitl.pt \
    --enable_gradient_checkpointing True \
    --vis_every 1000 \
    --enable_mlflow True
```
