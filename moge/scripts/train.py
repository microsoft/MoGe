import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
import time
import random
from typing import *
import itertools
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import io

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.version
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import utils3d
import click
from tqdm import tqdm, trange
import mlflow
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

from moge.train.dataloader import TrainDataLoaderPipeline
from moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss, 
    edge_loss,
    normal_loss, 
    mask_l2_loss, 
    mask_bce_loss,
    monitoring, 
)
from moge.train.utils import build_optimizer, build_lr_scheduler
from moge.utils.geometry_torch import intrinsics_to_fov
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.tools import key_average, recursive_replace, CallbackOnException, flatten_nested_dict
from moge.test.metrics import compute_metrics


@click.command()
@click.option('--config', 'config_path', type=str, default='configs/debug.json')
@click.option('--workspace', type=str, default='workspace/debug', help='Path to the workspace')
@click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Path to the checkpoint to load')
@click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in backbone')
@click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16')
@click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights')
@click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
@click.option('--save_every', type=int, default=10000, help='Save checkpoint every n iterations')
@click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
@click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize, must be a multiple of divided batch size')
@click.option('--enable_mlflow', type=bool, default=True, help='Log metrics to MLFlow')
@click.option('--seed', type=int, default=0, help='Random seed')
def main(
    config_path: str,
    workspace: str,
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    enable_gradient_checkpointing: bool,
    enable_mixed_precision: bool,
    enable_ema: bool,
    num_iterations: int,
    save_every: int,
    log_every: int,
    vis_every: int,
    num_vis_images: int,
    enable_mlflow: bool,
    seed: Optional[int],
):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16' if enable_mixed_precision else None,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )
    device = accelerator.device
    batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes

    # Log config
    if accelerator.is_main_process:
        if enable_mlflow:
            try:
                mlflow.log_params({
                    **click.get_current_context().params,
                    'batch_size_total': batch_size_total,
                })
            except:
                print('Failed to log config to MLFlow')
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with Path(workspace).joinpath('config.json').open('w') as f:
            json.dump(config, f, indent=4)

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
    print('Initialize model')
    with accelerator.local_main_process_first():
        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(config['model_version'])      
        model = MoGeModel(**config['model'])
    count_total_parameters = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {count_total_parameters}')

    # Set up EMA model
    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

    # Set gradient checkpointing
    if enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
    # Initalize optimizer & lr scheduler
    optimizer = build_optimizer(model, config['optimizer'])
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

    count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
    for i, count in enumerate(count_grouped_parameters):
        print(f'- Group {i}: {count} parameters')

    # Attempt to load checkpoint
    checkpoint: Dict[str, Any]
    with accelerator.local_main_process_first():
        if checkpoint_path.endswith('.pt'):
            # - Load specific checkpoint file
            print(f'Load checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        elif checkpoint_path == "latest": 
            # - Load latest
            checkpoint_path = Path(workspace, 'checkpoint', 'latest.pt')
            if checkpoint_path.exists():
                print(f'Load checkpoint: {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                i_step = checkpoint['step']
                if 'model' not in checkpoint and (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
                    print(f'Load model checkpoint: {checkpoint_model_path}')
                    checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
                if 'optimizer' not in checkpoint and (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
                    print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
                    checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
                if enable_ema and accelerator.is_main_process:
                    if 'ema_model' not in checkpoint and (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
                        print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
                        checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
            else:
                checkpoint = None
        elif checkpoint_path is not None:
            # - Load by step number
            i_step = int(checkpoint_path)
            checkpoint = {'step': i_step}
            if (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
                print(f'Load model checkpoint: {checkpoint_model_path}')
                checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
            if (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
                print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
                checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
            if enable_ema and accelerator.is_main_process:
                if (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
                    print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
                    checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
        else:
            checkpoint = None

    if checkpoint is None:
        # Initialize model weights
        print('Initialize model weights')
        with accelerator.local_main_process_first():
            model.init_weights()
        initial_step = 0
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'step' in checkpoint:
            initial_step = checkpoint['step'] + 1
        else:
            initial_step = 0
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if enable_ema and accelerator.is_main_process and 'ema_model' in checkpoint:
            ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        del checkpoint
    
    model, optimizer = accelerator.prepare(model, optimizer)
    if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # Hacking potential gradient synchronization issue in ROCm backend
        from moge.model.utils import sync_ddp_hook
        model.register_comm_hook(None, sync_ddp_hook)

    # Initialize training data pipeline
    with accelerator.local_main_process_first():
        train_data_pipe = TrainDataLoaderPipeline(config['data'], batch_size_forward)

    def _write_bytes_retry_loop(save_path: Path, data: bytes):
        while True:
            try:
                save_path.write_bytes(data)
                break
            except Exception as e:
                print('Error while saving checkpoint, retrying in 1 minute: ', e)
                time.sleep(60)

    # Ready to train
    records = []
    model.train()
    with (
        train_data_pipe,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
        ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
    ):  
        # Get some batches for visualization
        if accelerator.is_main_process:
            batches_for_vis: List[Dict[str, torch.Tensor]] = []
            num_vis_images = num_vis_images // batch_size_forward * batch_size_forward
            for _ in range(num_vis_images // batch_size_forward):
                batch = train_data_pipe.get()
                batches_for_vis.append(batch)

        # Visualize GT
        if vis_every > 0 and accelerator.is_main_process and initial_step == 0:
            save_dir = Path(workspace).joinpath('vis/gt')
            for i_batch, batch in enumerate(tqdm(batches_for_vis, desc='Visualize GT', leave=False)):
                image, gt_depth, gt_mask, gt_mask_inf, gt_intrinsics, info = batch['image'], batch['depth'], batch['depth_mask'], batch['depth_mask_inf'], batch['intrinsics'], batch['info']
                gt_points = utils3d.torch.depth_to_points(gt_depth, intrinsics=gt_intrinsics)
                gt_normal, gt_normal_mask = utils3d.torch.points_to_normals(gt_points, gt_mask)
                for i_instance in range(batch['image'].shape[0]):
                    idx = i_batch * batch_size_forward + i_instance
                    image_i = (image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt_depth_i = gt_depth[i_instance].numpy()
                    gt_mask_i = gt_mask[i_instance].numpy()
                    gt_mask_inf_i = gt_mask_inf[i_instance].numpy()
                    gt_points_i = gt_points[i_instance].numpy()
                    gt_normal_i = gt_normal[i_instance].numpy()
                    save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(gt_points_i, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/mask.png')), gt_mask_i * 255)
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(gt_depth_i, gt_mask_i), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal.png')), cv2.cvtColor(colorize_normal(gt_normal_i), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/mask_inf.png')), gt_mask_inf_i * 255)
                    with save_dir.joinpath(f'{idx:04d}/info.json').open('w') as f:
                        json.dump(info[i_instance], f)

        # Reset seed to avoid training on the same data when resuming training
        if seed is not None:
            set_seed(seed + initial_step, device_specific=True)   

        # Training loop
        for i_step in range(initial_step, num_iterations):

            i_accumulate, weight_accumulate = 0, 0
            while i_accumulate < gradient_accumulation_steps:
                # Load batch
                batch = train_data_pipe.get()
                image, gt_depth, gt_mask, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric = batch['image'], batch['depth'], batch['depth_mask'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric']
                image, gt_depth, gt_mask, gt_mask_fin, gt_mask_inf, gt_intrinsics = image.to(device), gt_depth.to(device), gt_mask.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
                current_batch_size = image.shape[0]
                if all(label == 'invalid' for label in label_type):
                    continue            # NOTE: Skip all-invalid batches to avoid messing up the optimizer.
                
                gt_points = utils3d.torch.depth_to_points(gt_depth, intrinsics=gt_intrinsics)
                gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

                with accelerator.accumulate(model):
                    # Forward
                    if i_step <= config.get('low_resolution_training_steps', 0):
                        num_tokens = config['model']['num_tokens_range'][0]
                    else:
                        num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    with torch.autocast(device_type=accelerator.device.type, dtype=torch.float16, enabled=enable_mixed_precision):
                        output = model(image, num_tokens=num_tokens)
                    pred_points, pred_mask, pred_metric_scale = output['points'], output['mask'], output.get('metric_scale', None)

                    # Compute loss (per instance)
                    loss_list, weight_list = [], []
                    for i in range(current_batch_size):
                        gt_metric_scale = None
                        loss_dict, weight_dict, misc_dict = {}, {}, {}
                        misc_dict['monitoring'] = monitoring(pred_points[i])
                        for k, v in config['loss'][label_type[i]].items():
                            weight_dict[k] = v['weight']
                            if v['function'] == 'affine_invariant_global_loss':
                                loss_dict[k], misc_dict[k], gt_metric_scale = affine_invariant_global_loss(pred_points[i], gt_points[i], gt_mask[i], **v['params'])
                            elif v['function'] == 'affine_invariant_local_loss':
                                loss_dict[k], misc_dict[k] = affine_invariant_local_loss(pred_points[i], gt_points[i], gt_mask[i], gt_focal[i], gt_metric_scale, **v['params'])
                            elif v['function'] == 'normal_loss':
                                loss_dict[k], misc_dict[k] = normal_loss(pred_points[i], gt_points[i], gt_mask[i])
                            elif v['function'] == 'edge_loss':
                                loss_dict[k], misc_dict[k] = edge_loss(pred_points[i], gt_points[i], gt_mask[i])
                            elif v['function'] == 'mask_bce_loss':
                                loss_dict[k], misc_dict[k] = mask_bce_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                            elif v['function'] == 'mask_l2_loss':
                                loss_dict[k], misc_dict[k] = mask_l2_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                            else:
                                raise ValueError(f'Undefined loss function: {v["function"]}')
                        weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                        loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                        loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
                        loss_list.append(loss_)
                        
                        if torch.isnan(loss_).item():
                            pbar.write(f'NaN loss in process {accelerator.process_index}')
                            pbar.write(str(loss_dict))

                        misc_dict = {'.'.join(k): v for k, v in flatten_nested_dict(misc_dict).items()}
                        records.append({
                            **{k: v.item() for k, v in loss_dict.items()},
                            **misc_dict,
                        })

                    loss = sum(loss_list) / len(loss_list)
                    
                    # Backward & update
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if not enable_mixed_precision and any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                            if accelerator.is_main_process:
                                pbar.write(f'NaN gradients, skip update')
                            optimizer.zero_grad()
                            continue
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            
                    optimizer.step()
                    optimizer.zero_grad()

                i_accumulate += 1

            lr_scheduler.step()

            # EMA update            
            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                ema_model.update_parameters(model)

            # Log metrics
            if i_step == initial_step or i_step % log_every == 0:
                records = [key_average(records)]
                records = accelerator.gather_for_metrics(records, use_gather_object=True)
                if accelerator.is_main_process:
                    records = key_average(records)
                    if enable_mlflow:
                        try:
                            mlflow.log_metrics(records, step=i_step)
                        except Exception as e:
                            print(f'Error while logging metrics to mlflow: {e}')
                records = []

            # Save model weight checkpoint
            if accelerator.is_main_process and (i_step % save_every == 0):
                # NOTE: Writing checkpoint is done in a separate thread to avoid blocking the main process
                pbar.write(f'Save checkpoint: {i_step:08d}')
                Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)

                # Model checkpoint
                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'model': accelerator.unwrap_model(model).state_dict(),
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(
                    _write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}.pt'), checkpoint_bytes
                )

                # Optimizer checkpoint
                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'step': i_step,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(
                    _write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt'), checkpoint_bytes
                )
                
                # EMA model checkpoint
                if enable_ema:
                    with io.BytesIO() as f:
                        torch.save({
                            'model_config': config['model'],
                            'model': ema_model.module.state_dict(),
                        }, f)
                        checkpoint_bytes = f.getvalue()
                    save_checkpoint_executor.submit(
                        _write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt'), checkpoint_bytes
                    )

                # Latest checkpoint
                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'step': i_step,
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(
                    _write_bytes_retry_loop, Path(workspace, 'checkpoint', 'latest.pt'), checkpoint_bytes
                )
            
            # Visualize
            if vis_every > 0 and accelerator.is_main_process and (i_step == initial_step or i_step % vis_every == 0):
                unwrapped_model = accelerator.unwrap_model(model)
                save_dir = Path(workspace).joinpath(f'vis/step_{i_step:08d}')
                save_dir.mkdir(parents=True, exist_ok=True)
                with torch.inference_mode():
                    for i_batch, batch in enumerate(tqdm(batches_for_vis, desc=f'Visualize: {i_step:08d}', leave=False)):
                        image, gt_depth, gt_mask, gt_intrinsics = batch['image'], batch['depth'], batch['depth_mask'], batch['intrinsics']
                        image, gt_depth, gt_mask, gt_intrinsics = image.to(device), gt_depth.to(device), gt_mask.to(device), gt_intrinsics.to(device)
                        
                        output = unwrapped_model.infer(image)
                        pred_points, pred_depth, pred_mask = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy()
                        image = image.cpu().numpy()

                        for i_instance in range(image.shape[0]):
                            idx = i_batch * batch_size_forward + i_instance
                            image_i = (image[i_instance].transpose(1, 2, 0) * 255).astype(np.uint8)
                            pred_points_i = pred_points[i_instance]
                            pred_mask_i = pred_mask[i_instance]
                            pred_depth_i = pred_depth[i_instance]
                            save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(pred_points_i, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/mask.png')), pred_mask_i * 255)
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(pred_depth_i, pred_mask_i), cv2.COLOR_RGB2BGR))

            pbar.set_postfix({'loss': loss.item()}, refresh=False)
            pbar.update(1)


if __name__ == '__main__':
    main()