"""Training-time visualisation dumps shared by the MoGe trainers."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import json
from pathlib import Path
from typing import *

import cv2
import numpy as np
import torch
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d
from tqdm import tqdm

from ..utils.vis import colorize_depth, colorize_normal

EXR_FLOAT = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]


def _write_rgb(path: Path, image: np.ndarray, params: Optional[List[int]] = None):
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params or [])


def visualize_gt(
    batches_for_vis: List[Dict[str, Any]],
    workspace: Path,
    batch_size_forward: int,
    initial_step: int,
    logger,
):
    """Dump the ground truth of the held-out visualisation batches once."""
    save_dir = Path(workspace).joinpath('vis/gt')
    for i_batch, batch in enumerate(tqdm(batches_for_vis, desc='Visualize GT', leave=False)):
        image, gt_depth, gt_normal, gt_intrinsics, info = (
            batch['image'], batch['depth'], batch['normal'], batch['intrinsics'], batch['info']
        )
        gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
        for i_instance in range(batch['image'].shape[0]):
            idx = i_batch * batch_size_forward + i_instance
            image_i = (image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gt_depth_i = gt_depth[i_instance].numpy()
            instance_dir = save_dir.joinpath(f'{idx:04d}')
            instance_dir.mkdir(parents=True, exist_ok=True)
            _write_rgb(instance_dir / 'image.jpg', image_i)
            _write_rgb(instance_dir / 'points.exr', gt_points[i_instance].numpy(), EXR_FLOAT)
            _write_rgb(instance_dir / 'depth_vis.png', colorize_depth(gt_depth_i))
            _write_rgb(instance_dir / 'normal.png', colorize_normal(gt_normal[i_instance].numpy()))
            logger.log_images({
                f'{idx:04d}-image-gt': image_i,
                f'{idx:04d}-depth_vis-gt': colorize_depth(gt_depth_i),
            }, step=initial_step)
            with instance_dir.joinpath('info.json').open('w') as f:
                json.dump(info[i_instance], f)


def visualize_predictions(
    batches_for_vis: List[Dict[str, Any]],
    model,
    accelerator,
    workspace: Path,
    device,
    batch_size_forward: int,
    i_step: int,
    refine_steps: Optional[int],
    logger,
):
    """Run inference and dump."""
    unwrapped_model = accelerator.unwrap_model(model)
    save_dir = Path(workspace).joinpath(f'vis/step_{i_step:08d}')
    save_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i_batch, batch in enumerate(tqdm(batches_for_vis, desc=f'Visualize: {i_step:08d}', leave=False)):
            image = batch['image'].to(device)
            infer_kwargs = {'refine_steps': refine_steps, 'return_per_step': True} if refine_steps is not None else {}
            output = unwrapped_model.infer(image, **infer_kwargs)
            is_refiner_output = 'points_per_step' in output or 'depth_per_step' in output
            pred_points_all = [step.cpu().numpy() for step in output.get('points_per_step', [])]
            pred_depth_all = [step.cpu().numpy() for step in output.get('depth_per_step', [])]
            if not pred_points_all and output.get('points') is not None:
                pred_points_all = [output['points'].cpu().numpy()]
            if not pred_depth_all and output.get('depth') is not None:
                pred_depth_all = [output['depth'].cpu().numpy()]
            pred_mask = output['mask'].cpu().numpy() if output.get('mask') is not None else None
            pred_normal = output['normal'].cpu().numpy() if output.get('normal') is not None else None
            image = (image.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

            for i_instance in range(image.shape[0]):
                idx = i_batch * batch_size_forward + i_instance
                pred_mask_i = pred_mask[i_instance] if pred_mask is not None else None
                pred_mask_bool = pred_mask_i if pred_mask_i is None else pred_mask_i > 0.5
                instance_dir = save_dir.joinpath(f'{idx:04d}')
                instance_dir.mkdir(parents=True, exist_ok=True)
                _write_rgb(instance_dir / 'image.jpg', image[i_instance])
                if pred_mask_i is not None:
                    mask_name = f'mask_train_step_{i_step:08d}.png' if is_refiner_output else 'mask.png'
                    cv2.imwrite(str(instance_dir / mask_name), pred_mask_bool.astype(np.uint8) * 255)
                images_to_log = {}
                num_steps = max(len(pred_points_all), len(pred_depth_all))
                for i_refine_step in range(num_steps):
                    if is_refiner_output:
                        suffix = f'_train_step_{i_step:08d}_refine_step_{i_refine_step:02d}'
                        log_suffix = f'-refine-step-{i_refine_step:02d}'
                    else:
                        suffix = ''
                        log_suffix = ''
                    if i_refine_step < len(pred_points_all):
                        _write_rgb(instance_dir / f'points{suffix}.exr', pred_points_all[i_refine_step][i_instance], EXR_FLOAT)
                    if i_refine_step < len(pred_depth_all):
                        depth_vis = colorize_depth(pred_depth_all[i_refine_step][i_instance], pred_mask_bool)
                        _write_rgb(instance_dir / f'depth_vis{suffix}.png', depth_vis)
                        images_to_log[f'{idx:04d}-depth-vis-pred-train-step-{i_step:06d}{log_suffix}'] = depth_vis
                if pred_normal is not None:
                    normal_vis = colorize_normal(pred_normal[i_instance], pred_mask_bool)
                    _write_rgb(instance_dir / 'normal_vis.png', normal_vis)
                logger.log_images(images_to_log, step=i_step)
