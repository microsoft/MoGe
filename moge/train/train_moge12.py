import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from typing import *

import gc
import json
import random
import warnings
import click
import torch
import torch.version

try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from accelerate.utils import set_seed

from .dataloader import TrainDataLoaderPipeline
from .losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss,
    edge_loss,
    mask_bce_loss,
    mask_l2_loss,
    metric_scale_loss,
    monitoring,
    normal_loss,
    normal_map_loss,
)
from .checkpoint import (
    CheckpointSaver,
    load_checkpoint,
    restore_data_pipeline_states,
    restore_ma_buffer,
    restore_training_state,
    save_data_pipeline_states,
)
from .debug import DebugDumper
from .experiment import RunLogger, setup_accelerator
from .utils import (
    build_lr_scheduler,
    build_optimizer,
    materialize_log_records,
    to_device,
    to_log_scalar,
    write_optimizer_param_assignment_log,
)
from .visualization import visualize_gt, visualize_predictions
from ..utils.tools import flatten_nested_dict, key_average, timeit


warnings.filterwarnings('ignore', category=FutureWarning, module='torch.utils.checkpoint')
torch._dynamo.config.disable = True
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

# The cuDNN attention backend on H100 sm90 has a bug that occasionally causes NaN gradients.
if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
    torch.backends.cuda.enable_cudnn_sdp(False)


@click.command()
@click.option('--config', 'config_path', type=str, default='configs/train/v2.json')
@click.option('--name', 'experiment_name', type=str, default='MoGe-2', help='Experiment name')
@click.option('--workspace', 'workspace_path', type=str, default='workspace/moge2', help='Workspace for logs, visualizations and checkpoints')
@click.option('--initial_checkpoint', type=str, default='', help='Initial checkpoint used when the workspace has no requested checkpoint')
@click.option('--checkpoint', 'checkpoint_path', type=str, default='latest', help='Checkpoint path, step number, "latest", or "none"')
@click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in the backbone')
@click.option('--precision', type=click.Choice(['fp32', 'mixed_bf16']), default='fp32', help='Numerical precision')
@click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of model weights')
@click.option('--debug', 'debug_mode', type=bool, default=False, help='Enable additional debug dumps')
@click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
@click.option('--checkpoint_every', type=int, default=10000, help='Save a permanent checkpoint every n iterations')
@click.option('--rolling_checkpoint_every', type=int, default=500, help='Save a rolling checkpoint every n iterations')
@click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
@click.option('--vis_gt', type=bool, default=True, help='Visualize ground truth')
@click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize')
@click.option('--log_type', type=click.Choice(['mlflow', 'tensorboard', 'wandb']), multiple=True, default=(), help='Logging backend; may be specified more than once')
@click.option('--tb_log_root', type=str, default=None, help='Root directory for TensorBoard logs. Logs will be stored in a subdirectory named after the experiment.')
@click.option('--wandb_project', type=str, default='MoGe', help='Weights & Biases project name')
@click.option('--gc_every', type=int, default=1000, help='Run garbage collection every n iterations')
@click.option('--seed', type=int, default=0, help='Random seed')
@click.option('--find_unused_parameters', type=bool, default=True, help='Whether DDP should look for unused parameters')
def main(
    config_path: str,
    experiment_name: str,
    workspace_path: str,
    initial_checkpoint: str,
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    enable_gradient_checkpointing: bool,
    precision: str,
    enable_ema: bool,
    debug_mode: bool,
    num_iterations: int,
    checkpoint_every: int,
    rolling_checkpoint_every: int,
    log_every: int,
    vis_every: int,
    vis_gt: bool,
    num_vis_images: int,
    log_type: Tuple[str, ...],
    tb_log_root: Optional[str],
    wandb_project: str,
    gc_every: int,
    seed: int,
    find_unused_parameters: bool,
):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    if gradient_accumulation_steps < 1:
        raise ValueError(f'--gradient_accumulation_steps must be at least 1, got {gradient_accumulation_steps}')

    # Init
    accelerator, device, batch_size_total, workspace = setup_accelerator(
        gradient_accumulation_steps, find_unused_parameters, batch_size_forward, workspace_path,
    )
    logger = RunLogger(accelerator, log_type)
    logger.setup(
        workspace=workspace, config=config, experiment_name=experiment_name,
        tb_log_root=tb_log_root, wandb_project=wandb_project, batch_size_total=batch_size_total,
    )

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
    print('Initialize model')
    with accelerator.local_main_process_first():
        from moge.model import import_model_class_by_version
        model_class = import_model_class_by_version(config['model_version'])
        model = model_class(**config['model'])
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')

    # Set up EMA model
    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged, current, _: 0.999 * averaged + 0.001 * current
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=device, avg_fn=ema_avg_fn)

    # Set gradient checkpointing
    if enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Set precision
    if precision == 'mixed_bf16':
        if config['model_version'] == "v1":
            raise ValueError('mixed_bf16 precision is not supported for model version v1')
        model.enable_mixed_precision()

    # Initalize optimizer & lr scheduler
    optimizer = build_optimizer(model, config['optimizer'])
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])
    count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
    for i, count in enumerate(count_grouped_parameters):
        print(f'- Group {i}: {count} parameters')
    if accelerator.is_main_process:
        write_optimizer_param_assignment_log(model, optimizer, workspace)

    # Attempt to load checkpoint; fall back to the init checkpoint when the workspace has none.
    checkpoint = load_checkpoint(checkpoint_path, workspace, accelerator, enable_ema)
    if checkpoint is None:
        checkpoint = load_checkpoint(initial_checkpoint, workspace, accelerator, enable_ema)
    initial_step = restore_training_state(
        checkpoint, model, optimizer, lr_scheduler, ema_model if enable_ema and accelerator.is_main_process else None,
        accelerator, enable_ema,
    )
    del checkpoint

    model, optimizer = accelerator.prepare(model, optimizer)
    if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # Hacking potential gradient synchronization issue in ROCm backend
        from moge.model.utils import sync_ddp_hook
        model.register_comm_hook(None, sync_ddp_hook)

    # Initialize training data pipelines
    dataloader_seed = seed + accelerator.process_index
    with accelerator.local_main_process_first():
        train_data_pipe = TrainDataLoaderPipeline(
            deepcopy(config['data']),
            batch_size_forward,
            workspace=workspace,
            seed=dataloader_seed,
        )

    # Restore data pipeline RNG state if resuming
    data_pipelines = {'train': train_data_pipe}
    restore_data_pipeline_states(workspace, initial_step, accelerator, data_pipelines)

    records: List[Dict[str, Any]] = []
    ma_buffer = restore_ma_buffer(workspace, initial_step, accelerator)

    model.train()

    with (
        train_data_pipe,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
        ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
    ):
        checkpoint_saver = CheckpointSaver(
            workspace=workspace,config=config, accelerator=accelerator, model=model,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            ema_model=ema_model if enable_ema and accelerator.is_main_process else None,
            enable_ema=enable_ema, ma_buffer=ma_buffer, executor=save_checkpoint_executor,
            pbar=pbar, num_iterations=num_iterations, checkpoint_every=checkpoint_every,
            rolling_checkpoint_every=rolling_checkpoint_every, initial_step=initial_step,
        )

        # Get some batches for visualization
        batches_for_vis: List[Dict[str, torch.Tensor]] = []
        if accelerator.is_main_process and vis_every > 0:
            # Visualisation works in whole forward batches. Rounding down to zero would leave
            # --vis_every silently creating empty directories for the rest of the run, so keep
            # at least one batch.
            num_vis_batches = max(1, num_vis_images // batch_size_forward)
            if num_vis_batches * batch_size_forward != num_vis_images:
                pbar.write(
                    f'num_vis_images={num_vis_images} is not a multiple of batch_size_forward='
                    f'{batch_size_forward}; visualizing {num_vis_batches * batch_size_forward} images instead'
                )
            batches_for_vis = [train_data_pipe.get() for _ in range(num_vis_batches)]
            if vis_gt:
                visualize_gt(batches_for_vis, workspace, batch_size_forward, initial_step, logger)

        if seed is not None:
            set_seed(seed + initial_step, device_specific=True)

        # Tags starting with 'nan_' are fatal and count toward the abort threshold; others
        # (e.g. large_grad_norm_*) are informational and capped to avoid filling disk.
        dumper = DebugDumper(
            workspace, accelerator, model,
            dump_grad_norm_above=config.get('dump_grad_norm_above', 10 if debug_mode else None),
            save_model_on_first_dump=config.get('debug_save_model_on_first_dump', False),
        )

        # Training loop
        for i_step in range(initial_step, num_iterations):
            with timeit('Step', verbose=False) as timer_step:
                dumper.begin_step()
                for i_accumulate in range(gradient_accumulation_steps):
                    # Load batch
                    with timeit('Load instance', verbose=False) as timer_load:
                        batch = to_device(train_data_pipe.get(), device)
                    records.append({'time/data': timer_load.time})

                    image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric, info = batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric'], batch.get('info')

                    is_invalid_batch = all(label == 'invalid' for label in label_type)
                    if is_invalid_batch:
                        pbar.write(
                            f'Rank {accelerator.process_index} all-invalid batch at step {i_step}, '
                            f'accumulation {i_accumulate}. Batch info: {info}'
                        )

                    gt_points_raw = utils3d.pt.depth_map_to_point_map(
                        gt_depth, intrinsics=gt_intrinsics,
                    )
                    gt_points_mask = torch.isfinite(gt_points_raw).all(dim=-1)
                    gt_points = torch.where(gt_points_mask[..., None], gt_points_raw, 1)
                    gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

                    with accelerator.accumulate(model):
                        # Forward
                        if i_step <= config.get('low_resolution_training_steps', 0):
                            num_tokens = config['model']['num_tokens_range'][0]
                        else:
                            num_tokens = random.Random(f'num_tokens-{seed}-{i_step}-{i_accumulate}').randint(*config['model']['num_tokens_range'])

                        with timeit('Model forward', verbose=False) as timer_forward:
                            output = model(image, num_tokens=num_tokens)
                        records.append({'time/forward': timer_forward.time})
                        pred_points, pred_mask, pred_normal, pred_metric_scale = (
                            output.get(k) for k in ('points', 'mask', 'normal', 'metric_scale')
                        )

                        # Compute loss
                        with timeit('Loss computation', verbose=False) as timer_loss_computation:
                            if is_invalid_batch:
                                loss = torch.tensor(0.0, device=device, requires_grad=True)
                            else:
                                instance_losses = []
                                for i in range(image.shape[0]):
                                    gt_metric_scale = None
                                    loss_dict: Dict[str, Any] = {}
                                    weight_dict: Dict[str, Any] = {}
                                    misc_dict: Dict[str, Any] = {
                                        'monitoring': monitoring(pred_points[i].detach())
                                    }
                                    loss_config = config['loss'][label_type[i]]

                                    # points
                                    for name, spec in loss_config.get('points', {}).items():
                                        weight_dict[name] = spec['weight']
                                        function = spec['function']
                                        params = spec.get('params', {})
                                        if function == 'affine_invariant_global_loss':
                                            loss_dict[name], misc_dict[name], gt_metric_scale, _ = affine_invariant_global_loss(
                                                pred_points[i], gt_points[i], gt_points_mask[i], **params,
                                            )
                                            gt_metric_scale = gt_metric_scale.detach()
                                        elif function == 'affine_invariant_local_loss':
                                            if gt_metric_scale is None:
                                                raise RuntimeError(
                                                    'affine_invariant_local_loss requires a preceding global loss'
                                                )
                                            loss_dict[name], misc_dict[name] = affine_invariant_local_loss(
                                                pred_points[i], gt_points[i], gt_points_mask[i],
                                                gt_focal[i], gt_metric_scale, **params,
                                            )
                                        elif function == 'normal_loss':
                                            loss_dict[name], misc_dict[name] = normal_loss(
                                                pred_points[i], gt_points_raw[i],
                                            )
                                        elif function == 'edge_loss':
                                            loss_dict[name], misc_dict[name] = edge_loss(
                                                pred_points[i], gt_points[i], gt_points_mask[i],
                                            )
                                        else:
                                            raise ValueError(f'Undefined points loss function: {function}')

                                    # normal
                                    for name, spec in loss_config.get('normal', {}).items():
                                        weight_dict[name] = spec['weight']
                                        function = spec['function']
                                        params = spec.get('params', {})
                                        if function == 'normal_map_loss':
                                            loss_dict[name], misc_dict[name] = normal_map_loss(
                                                pred_normal[i], gt_normal[i], **params,
                                            )
                                        else:
                                            raise ValueError(f'Undefined normal loss function: {function}')

                                    # mask
                                    for name, spec in loss_config.get('mask', {}).items():
                                        weight_dict[name] = spec['weight']
                                        function = spec['function']
                                        params = spec.get('params', {})
                                        if function == 'mask_bce_loss':
                                            loss_dict[name], misc_dict[name] = mask_bce_loss(
                                                pred_mask[i], gt_mask_fin[i], gt_mask_inf[i], **params,
                                            )
                                        elif function == 'mask_l2_loss':
                                            loss_dict[name], misc_dict[name] = mask_l2_loss(
                                                pred_mask[i], gt_mask_fin[i], gt_mask_inf[i], **params,
                                            )
                                        else:
                                            raise ValueError(f'Undefined mask loss function: {function}')

                                    # metric_scale
                                    for name, spec in loss_config.get('metric_scale', {}).items():
                                        weight_dict[name] = spec['weight']
                                        function = spec['function']
                                        params = spec.get('params', {})
                                        if function == 'metric_scale_loss':
                                            if (
                                                bool(is_metric[i])
                                                and pred_metric_scale is not None
                                                and gt_metric_scale is not None
                                            ):
                                                loss_dict[name], misc_dict[name] = metric_scale_loss(
                                                    pred_metric_scale[i], gt_metric_scale, **params,
                                                )
                                        else:
                                            raise ValueError(f'Undefined metric_scale loss function: {function}')

                                    weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                                    loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                                    misc_dict = {'.'.join(k): v for k, v in flatten_nested_dict(misc_dict).items()}
                                    instance_loss = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
                                    instance_losses.append(instance_loss)

                                    # NaN loss check
                                    for name, value in loss_dict.items():
                                        if not torch.isfinite(value.detach()).all():
                                            pbar.write(
                                                f'NaN loss in process {accelerator.process_index}: {name}'
                                            )
                                            dumper.add_reason(f'nan_loss_{name}')
                                    records.append({
                                        **{f'loss/{k}': to_log_scalar(v) for k, v in loss_dict.items()},
                                        **{f'misc/{k}': to_log_scalar(v) for k, v in misc_dict.items()},
                                    })
                                loss = sum(instance_losses) / len(instance_losses)  # Average over the batch
                            records.append({'train/loss': to_log_scalar(loss)})
                        records.append({'time/loss': timer_loss_computation.time})

                        # Backward
                        with timeit('Backward', verbose=False) as timer_backward:
                            accelerator.backward(loss)
                        records.append({'time/backward': timer_backward.time})

                        # Attribute a non-finite gradient to the micro-batch that introduced it.
                        # Before the sync micro-step nothing has been all-reduced, so this pins
                        # the culprit to this rank and this batch.
                        dumper.check_grads(i_accumulate, synced=accelerator.sync_gradients)

                        # Optimizer step
                        if accelerator.sync_gradients:
                            # Clip grad norm
                            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            records.append({'train/grad_norm': to_log_scalar(grad_norm)})
                            # Step only if grad is finite
                            grad_norm_is_finite = True if accelerator.scaler is not None else torch.isfinite(grad_norm.detach()).cpu().item()
                            if grad_norm_is_finite or accelerator.scaler is not None:
                                optimizer.step()
                            else:
                                pbar.write(f'Non-finite gradient norm {grad_norm}; skip optimizer step')
                                pbar.write(f'Batch info: {info}')
                                if not dumper.grads_flagged:
                                    # Defensive: `check_grads` screens the same quantity and
                                    # should already own this, so landing here means they
                                    # disagreed. Never lose the event.
                                    dumper.add_reason('nan_grad_norm_unattributed')

                            # Extra dump trigger: large grad norm.
                            dumper.note_grad_norm(grad_norm, grad_norm_is_finite)

                        optimizer.zero_grad()

                        dumper.flush(i_step, i_accumulate, batch, output, meta={'num_tokens': num_tokens})

            records.append({'time/step': timer_step.time})
            lr_scheduler.step()

            # EMA update  
            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                ema_model.update_parameters(model)

            if log_every > 0:
                if i_step == initial_step or i_step % log_every == 0:
                    records = logger.log_metrics(records, ma_buffer, lr_scheduler, i_step, initial_step)
            else:
                # Logging disabled. `records` is only ever drained by `log_metrics`, and every
                # entry holds live CUDA scalars, so it has to be dropped here or it grows without
                # bound for the whole run.
                records = []

            # Save checkpoint
            due = checkpoint_saver.save_if_due(i_step)
            if due:
                save_data_pipeline_states(workspace, i_step, accelerator, data_pipelines)

            # Print data pipeline profile every 100 steps
            if accelerator.is_main_process and i_step > 0 and i_step % 100 == 0:
                pbar.write(f'[Step {i_step}] train data pipeline profile:\n{train_data_pipe.profile()}')


            # Visualize
            if (
                vis_every > 0
                and accelerator.is_main_process
                and (i_step == initial_step or i_step % vis_every == 0 or i_step == num_iterations - 1)
            ):
                visualize_predictions(
                    batches_for_vis, model, accelerator, workspace, device,
                    batch_size_forward, i_step, refine_steps=None, logger=logger,
                )

            pbar.update(1)

            # Garbage collection to reduce peak memory
            if gc_every > 0 and i_step % gc_every == 0 and i_step != initial_step:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
