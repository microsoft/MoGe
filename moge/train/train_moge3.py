import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from typing import *

import gc
import json
import math
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
    radial_partition_local_loss,
    radial_partition_local_loss_rand_partition,
    edge_loss,
    normal_map_loss,
    mask_bce_loss,
    metric_scale_loss,
    monitoring,
    monitor_delta,
    RadialPartitionLocalLossCache,
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
    accumulate_step_transitions,
    build_optimizer,
    refine_step_pairs,
    build_lr_scheduler,
    to_device,
    materialize_log_records,
    split_step_suffix,
    to_log_scalar,
    write_optimizer_param_assignment_log,
    write_refine_monitor_table,
)
from .visualization import visualize_gt, visualize_predictions
from ..utils.tools import key_average, flatten_nested_dict, timeit


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
torch._dynamo.config.disable = True
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

# The cuDNN attention backend on H100 sm90 has a bug that occasionally causes NaN gradients.
if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
    torch.backends.cuda.enable_cudnn_sdp(False)


def get_refine_accumulation_schedule(
    step: int,
    gradient_accumulation_steps: int,
    refine_ratio: float,
) -> List[bool]:
    start_quota = math.floor(step * gradient_accumulation_steps * refine_ratio + 1e-9)
    end_quota = math.floor((step + 1) * gradient_accumulation_steps * refine_ratio + 1e-9)
    num_refine_accumulations = end_quota - start_quota
    if num_refine_accumulations <= 0:
        return [False] * gradient_accumulation_steps
    if num_refine_accumulations >= gradient_accumulation_steps:
        return [True] * gradient_accumulation_steps
    num_norefine_accumulations = gradient_accumulation_steps - num_refine_accumulations
    return [i >= num_norefine_accumulations for i in range(gradient_accumulation_steps)]


@click.command()
@click.option('--config', 'config_path', type=str, default='configs/train/v3.json')
@click.option('--name', 'experiment_name', type=str, default='MoGe-3', help='Experiment name')
@click.option('--workspace', 'workspace_path', type=str, default='workspace/moge3', help='Workspace for logs, visualizations and checkpoints')
@click.option('--initial_checkpoint', type=str, default='', help='Initial checkpoint used when the workspace has no requested checkpoint')
@click.option('--checkpoint', 'checkpoint_path', type=str, default='latest', help='Checkpoint path, step number, "latest", or "none"')
@click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in the backbone')
@click.option('--precision', type=click.Choice(['fp32', 'mixed_bf16']), default='mixed_bf16', help='Numerical precision')
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
    gc_every: int,
    seed: int,
    wandb_project: str,
    find_unused_parameters: bool,
):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config['model_version'] != 'v3':
        raise ValueError(f"train_moge3_refiner.py is only compatible with model_version 'v3', got {config['model_version']}")
    if gradient_accumulation_steps < 1:
        raise ValueError(f'--gradient_accumulation_steps must be at least 1, got {gradient_accumulation_steps}')
    refine_ratio = config['refine_ratio']
    monitor_pairs = refine_step_pairs(config['refine_steps'])
    if not 0.0 <= refine_ratio <= 1.0:
        raise ValueError(f"config['refine_ratio'] must be in [0, 1], got {refine_ratio}")

    # Init
    accelerator, device, batch_size_total, workspace = setup_accelerator(
        gradient_accumulation_steps, find_unused_parameters, batch_size_forward, workspace_path,
    )
    logger = RunLogger(accelerator, log_type)
    logger.setup(
        workspace=workspace, config=config, experiment_name=experiment_name,
        tb_log_root=tb_log_root, wandb_project=wandb_project, batch_size_total=batch_size_total,
        extra_params={'refine_ratio': refine_ratio},
    )

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
    print('Initialize model')
    with accelerator.local_main_process_first():
        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(config['model_version'])      
        model = MoGeModel(**config['model'])
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')

    # Set up EMA model
    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged, current, _: 0.999 * averaged + 0.001 * current
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

    # Set gradient checkpointing
    if enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Set precision
    if precision == 'mixed_bf16':
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
        refine_data_pipeline = TrainDataLoaderPipeline(deepcopy(config['refine_data']),
            batch_size_forward,
            workspace=workspace,
            seed=dataloader_seed
        )
        norefine_data_pipeline = TrainDataLoaderPipeline(
            deepcopy(config['norefine_data']),
            batch_size_forward,
            workspace=workspace,
            seed=dataloader_seed
        )

    # Restore data pipeline RNG state if resuming
    data_pipelines = {'refine': refine_data_pipeline, 'norefine': norefine_data_pipeline}
    restore_data_pipeline_states(workspace, initial_step, accelerator, data_pipelines)

    records: List[Dict[str, Any]] = []
    ma_buffer = restore_ma_buffer(workspace, initial_step, accelerator)

    model.train()

    with (
        refine_data_pipeline,
        norefine_data_pipeline,
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
            for _ in range(num_vis_batches // 2):
                batch = norefine_data_pipeline.get()
                batches_for_vis.append(batch)
            for _ in range(num_vis_batches - len(batches_for_vis)):
                batch = refine_data_pipeline.get()
                batches_for_vis.append(batch)

        # Visualize GT
        if vis_every > 0 and accelerator.is_main_process and vis_gt:
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

        # Trackers and logs for refiner training monitoring.
        loss_decrease_tracker: Dict[Tuple, List[int]] = {}
        loss_decrease_log: Dict[str, float] = {}
        delta_increase_tracker: Dict[Tuple, List[int]] = {}
        delta_increase_log: Dict[str, float] = {}
        error_decrease_tracker: Dict[Tuple, List[int]] = {}
        error_decrease_log: Dict[str, float] = {}

        # Training loop
        for i_step in range(initial_step, num_iterations):
            with timeit('Step', verbose=False) as timer_step:
                dumper.begin_step()
                refine_accumulation_schedule = get_refine_accumulation_schedule(
                    i_step, gradient_accumulation_steps, refine_ratio,
                )
                for i_accumulate in range(gradient_accumulation_steps):
                    use_refine_pipeline = refine_accumulation_schedule[i_accumulate]
                    active_data_pipeline = refine_data_pipeline if use_refine_pipeline else norefine_data_pipeline
                    refine_steps = config['refine_steps'] if use_refine_pipeline else 0
                    # Load batch
                    with timeit('Load instance', verbose=False) as timer_load:
                        batch = to_device(active_data_pipeline.get(), device)
                    records.append({'time/data': timer_load.time})

                    image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric, info = batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric'], batch['info']
                    current_batch_size = image.shape[0]

                    is_invalid_batch = all(label == 'invalid' for label in label_type)
                    if is_invalid_batch:
                        pbar.write(
                            f'Rank {accelerator.process_index} all-invalid batch at step {i_step}, '
                            f'accumulation {i_accumulate}. Batch info: {info}'
                        )

                    gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                    gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

                    with accelerator.accumulate(model):
                        # Forward
                        if i_step <= config.get('low_resolution_training_steps', 0):
                            num_tokens = config['model']['num_tokens_range'][0]
                        else:
                            num_tokens = random.Random(f'num_tokens-{seed}-{i_step}-{i_accumulate}').randint(*config['model']['num_tokens_range'])
                        
                        _detach_backbone = i_step < config['refiner_detach_backbone_until']
                        with timeit('Model forward', verbose=False) as timer_forward:
                            output = model(
                                image,
                                num_tokens=num_tokens,
                                refine_steps=refine_steps,
                                refiner_detach_backbone=_detach_backbone,
                                return_per_step=True,
                            )
                        records.append({'time/forward': timer_forward.time})
                        pred_points_all = output.get('points_per_step', None)
                        pred_normal, pred_mask, pred_metric_scale = (output.get(k, None) for k in ['normal', 'mask', 'metric_scale'])

                        # Compute loss
                        with timeit('Loss computation', verbose=False) as timer_loss_computation:
                            if is_invalid_batch:
                                loss = torch.tensor(0.0, device=device, requires_grad=True)
                            else:
                                loss_list, weight_list = [], []

                                for i in range(current_batch_size):
                                    mask_i = torch.isfinite(gt_points[i]).all(dim=-1)
                                    gt_points_i = torch.where(mask_i[..., None], gt_points[i], 1)
                                    gt_metric_scale = None
                                    step0_gt_metric_scale = None
                                    refine_step0_scale = None
                                    loss_dict, weight_dict, misc_dict, refine_stat_dict = {}, {}, {}, {}
                                    is_refine_instance = label_type[i] == "D"
                                    radial_loss_cache: Optional[RadialPartitionLocalLossCache] = None

                                    # points
                                    pred_points_i = [p[i] for p in pred_points_all]
                                    for pred_step, pred_points_iter in enumerate(pred_points_i):
                                        with torch.no_grad():
                                            misc_dict[f'monitoring_step_{pred_step}'] = monitoring(pred_points_iter.detach())
                                        if pred_step > 0:
                                            delta_stats = monitor_delta(pred_points_i[pred_step - 1], pred_points_iter)
                                            refine_stat_dict[f'delta_z_step_{pred_step}_'] = delta_stats
                                        for k, v in config['loss'][label_type[i]].get('points', {}).items():
                                            if pred_step not in v['apply_steps']:
                                                continue
                                            iter_key = f'{k}_step_{pred_step}' if pred_step > 0 else k

                                            if not is_refine_instance:
                                                weight_dict[iter_key] = v['weight']
                                            else:
                                                if _detach_backbone:
                                                    if pred_step == 0:
                                                        weight_dict[iter_key] = v['weight']
                                                    else:
                                                        weight_dict[iter_key] = v['weight'] / config['refine_steps']
                                                else:
                                                    weight_dict[iter_key] = v['weight'] / len(v['apply_steps'])

                                            if v['function'] == 'affine_invariant_global_loss':
                                                # For refine steps (pred_step > 0), reuse the step-0 scale so the
                                                # alignment only solves the z-shift and the loss stays scale-sensitive.
                                                _fixed_scale = refine_step0_scale if pred_step > 0 else None
                                                loss_dict[iter_key], misc_dict[iter_key], gt_metric_scale, _ = affine_invariant_global_loss(
                                                    pred_points_iter,
                                                    gt_points_i,
                                                    mask_i,
                                                    fixed_scale=_fixed_scale,
                                                    **v['params'],
                                                )
                                                if pred_step == 0:
                                                    step0_gt_metric_scale = gt_metric_scale
                                                    # Detached so later refine steps pin their scale to step 0 without
                                                    # back-propagating step-k losses into the step-0 point map.
                                                    refine_step0_scale = gt_metric_scale.detach()
                                            elif v['function'] in {'radial_partition_local_loss', 'radial_partition_local_loss_rand_partition'}:
                                                scale_to_use = gt_metric_scale
                                                if scale_to_use is None:
                                                    raise RuntimeError(f"{v['function']} requires a preceding global scale loss in the same points config")
                                                if scale_to_use.dim() == 0:
                                                    scale_to_use = scale_to_use.unsqueeze(0)
                                                if radial_loss_cache is None:
                                                    radial_loss_cache = RadialPartitionLocalLossCache.from_inputs(gt_points_i, mask_i)
                                                radial_loss_fn = radial_partition_local_loss_rand_partition if v['function'] == 'radial_partition_local_loss_rand_partition' else radial_partition_local_loss
                                                loss_dict[iter_key], misc_dict[iter_key] = radial_loss_fn(
                                                    pred_points_iter,
                                                    gt_points_i,
                                                    mask_i,
                                                    scale_to_use,
                                                    packed=radial_loss_cache,
                                                    **v['params'],
                                                )
                                            elif v['function'] == 'edge_loss':
                                                loss_dict[iter_key], misc_dict[iter_key] = edge_loss(
                                                    pred_points_iter,
                                                    gt_points_i,
                                                    mask_i,
                                                )
                                            else:
                                                raise ValueError(f"Undefined points loss function: {v['function']}")

                                    # normal
                                    for k, v in config['loss'][label_type[i]].get('normal', {}).items():
                                        weight_dict[k] = v['weight']
                                        if v['function'] == 'normal_map_loss':
                                            loss_dict[k], misc_dict[k] = normal_map_loss(
                                                pred_normal[i], gt_normal[i], **v.get('params', {}),
                                            )
                                        else:
                                            raise ValueError(f"Undefined normal loss function: {v['function']}")

                                    # mask
                                    for k, v in config['loss'][label_type[i]].get('mask', {}).items():
                                        weight_dict[k] = v['weight']
                                        if v['function'] == 'mask_bce_loss':
                                            loss_dict[k], misc_dict[k] = mask_bce_loss(
                                                pred_mask[i], gt_mask_fin[i], gt_mask_inf[i], **v.get('params', {}),
                                            )
                                        else:
                                            raise ValueError(f"Undefined mask loss function: {v['function']}")

                                    # metric_scale
                                    for k, v in config['loss'][label_type[i]].get('metric_scale', {}).items():
                                        weight_dict[k] = v['weight']
                                        if v['function'] == 'metric_scale_loss':
                                            if is_metric[i] and pred_metric_scale is not None and step0_gt_metric_scale is not None:
                                                loss_dict[k], misc_dict[k] = metric_scale_loss(
                                                    pred_metric_scale[i], step0_gt_metric_scale.detach(), **v.get('params', {}),
                                                )
                                        else:
                                            raise ValueError(f"Undefined metric_scale loss function: {v['function']}")

                                    weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                                    loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                                    loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
                                    loss_list.append(loss_)
                                    
                                    # NaN loss check
                                    loss_finite_names, loss_finite_flags = [], []
                                    for loss_name, loss_value in loss_dict.items():
                                        loss_finite_names.append(loss_name)
                                        loss_finite_flags.append(torch.isfinite(loss_value.detach()).all())
                                    if loss_finite_flags:
                                        loss_finite_values = torch.stack(loss_finite_flags).cpu().tolist()
                                        for loss_name, is_finite in zip(loss_finite_names, loss_finite_values):
                                            if not is_finite:
                                                pbar.write(f'NaN loss in process {accelerator.process_index}, loss name: {loss_name}')
                                                dumper.add_reason(f'nan_loss_{loss_name}')

                                    misc_dict = {'.'.join(k): v for k, v in flatten_nested_dict(misc_dict).items()}
                                    refine_stat_dict = {'.'.join(k): v for k, v in flatten_nested_dict(refine_stat_dict).items()}
                                    loss_log_dict = {f"loss/{k}": v_ for k, v in loss_dict.items() if (v_ := to_log_scalar(v)) is not None}
                                    misc_log_dict = {f"misc/{k}": v_ for k, v in misc_dict.items() if (v_ := to_log_scalar(v)) is not None}
                                    refine_log_dict = {f"refine_stat/{k}": v_ for k, v in refine_stat_dict.items() if (v_ := to_log_scalar(v)) is not None}
                                    records.append({
                                        **loss_log_dict,
                                        **misc_log_dict,
                                        **refine_log_dict,
                                    })

                                    # Monitor refine step regression (only for refine instances)
                                    if is_refine_instance and accelerator.is_main_process:
                                        loss_dict_for_monitor, misc_dict_for_monitor = materialize_log_records([loss_dict, misc_dict])

                                        monitor_loss_by_step: Dict[str, Dict[int, float]] = {}
                                        for lk, lv in loss_dict_for_monitor.items():
                                            base, step_num = split_step_suffix(lk)
                                            monitor_loss_by_step.setdefault(base, {})[step_num] = lv

                                        # Misc keys are '<name>[_step_N].<metric>'; only delta* and
                                        # truncated_error are tracked across refine steps.
                                        misc_delta_by_step: Dict[str, Dict[int, float]] = {}
                                        misc_error_by_step: Dict[str, Dict[int, float]] = {}
                                        for mk, mv in misc_dict_for_monitor.items():
                                            dot_pos = mk.rfind('.')
                                            if dot_pos < 0:
                                                continue
                                            base_name, step_num = split_step_suffix(mk[:dot_pos])
                                            metric_name = mk[dot_pos + 1:]
                                            if metric_name == 'delta' or metric_name.startswith('delta_'):
                                                delta_key = base_name if metric_name == 'delta' else f'{base_name}.{metric_name}'
                                                misc_delta_by_step.setdefault(delta_key, {})[step_num] = mv
                                            elif metric_name == 'truncated_error':
                                                misc_error_by_step.setdefault(base_name, {})[step_num] = mv

                                        # Counted predicate must match how each table reports it.
                                        accumulate_step_transitions(monitor_loss_by_step, loss_decrease_tracker, lambda to, fr: to > fr, monitor_pairs)
                                        accumulate_step_transitions(misc_delta_by_step, delta_increase_tracker, lambda to, fr: to > fr, monitor_pairs)
                                        accumulate_step_transitions(misc_error_by_step, error_decrease_tracker, lambda to, fr: to < fr, monitor_pairs)

                                loss = sum(loss_list) / len(loss_list)  # Average over the batch
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
                                pbar.write(f'Non-finite gradient norm {grad_norm} encountered in process {accelerator.process_index}, skip optimizer step.')
                                pbar.write(f'Batch info: {info}')
                                if not dumper.grads_flagged:
                                    # Defensive: `check_grads` screens the same quantity and
                                    # should already own this, so landing here means they
                                    # disagreed. Never lose the event.
                                    dumper.add_reason('nan_grad_norm_unattributed')

                            # Extra dump trigger: large grad norm.
                            dumper.note_grad_norm(grad_norm, grad_norm_is_finite)

                        optimizer.zero_grad()

                        dumper.flush(i_step, i_accumulate, batch, output, meta={
                            'num_tokens': num_tokens,
                            'refine_steps': refine_steps,
                            'use_refine_pipeline': use_refine_pipeline,
                            'refiner_detach_backbone': _detach_backbone,
                        })

            records.append({'time/step': timer_step.time})
            lr_scheduler.step()

            # EMA update            
            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                ema_model.update_parameters(model)

            # Print refine loss regression stats every 100 steps
            if accelerator.is_main_process and i_step % 100 == 0 and i_step != initial_step:
                write_refine_monitor_table(
                    pbar, i_step, loss_decrease_tracker, loss_decrease_log,
                    title='Refine loss decrease% monitor (bigger=better):',
                    label='loss', log_prefix='loss_decrease', pairs=monitor_pairs, invert=True,
                )
                write_refine_monitor_table(
                    pbar, i_step, delta_increase_tracker, delta_increase_log,
                    title='Misc delta increase% monitor (bigger=better):',
                    label='metric', log_prefix='delta_increase', pairs=monitor_pairs,
                )
                write_refine_monitor_table(
                    pbar, i_step, error_decrease_tracker, error_decrease_log,
                    title='Misc error decrease% monitor (bigger=better):',
                    label='metric', log_prefix='error_decrease', pairs=monitor_pairs,
                )

            # Log metrics
            if log_every > 0:
                if i_step == initial_step or i_step % log_every == 0:
                    _extra_scalars = {}
                    for _log in (loss_decrease_log, delta_increase_log, error_decrease_log):
                        _extra_scalars.update(_log)
                        _log.clear()
                    records = logger.log_metrics(
                        records, ma_buffer, lr_scheduler, i_step, initial_step, extra_scalars=_extra_scalars,
                    )
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
                pbar.write(f'[Step {i_step}] refine data pipeline profile:\n{refine_data_pipeline.profile()}')
                pbar.write(f'[Step {i_step}] norefine data pipeline profile:\n{norefine_data_pipeline.profile()}')

            # Visualize
            if (
                vis_every > 0
                and accelerator.is_main_process
                and (i_step == initial_step or i_step % vis_every == 0 or i_step == num_iterations - 1)
            ):
                visualize_predictions(
                    batches_for_vis, model, accelerator, workspace, device,
                    batch_size_forward, i_step, refine_steps=config['refine_steps'], logger=logger,
                )

            pbar.update(1)

            # Garbage collection to reduce peak memory
            if gc_every > 0 and i_step % gc_every == 0 and i_step != initial_step:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
