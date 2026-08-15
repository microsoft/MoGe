from typing import *
import fnmatch
import time
from pathlib import Path
from numbers import Number
from collections import Counter
import json
import numpy as np
import sympy
import torch
import torch.nn as nn

from ..utils.tools import flatten_nested_dict


def any_match(s: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)


T = TypeVar('T')
def to_device(data: T, device: Union[str, torch.device]) -> T:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data  # For other types (e.g., int, float, str), return as is


def write_bytes_retry_loop(save_path: Path, data: bytes):
    while True:
        try:
            save_path.write_bytes(data)
            break
        except Exception as e:
            print('Error while saving checkpoint, retrying in 1 minute: ', e)
            time.sleep(60)


def to_log_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            return value.detach().reshape(())
        return value.detach().float().mean()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.mean())
    if isinstance(value, Number):
        return float(value)
    return None


def materialize_log_records(log_records: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    materialized: List[Dict[str, float]] = []
    cuda_scalars: List[torch.Tensor] = []
    cuda_slots: List[Tuple[Dict[str, float], str]] = []

    for record in log_records:
        out: Dict[str, float] = {}
        materialized.append(out)
        for key, value in record.items():
            scalar = to_log_scalar(value)
            if scalar is None:
                continue
            if isinstance(scalar, torch.Tensor):
                scalar = scalar.detach()
                if scalar.numel() != 1:
                    scalar = scalar.float().mean()
                scalar = scalar.reshape(())
                if scalar.is_cuda:
                    cuda_slots.append((out, key))
                    cuda_scalars.append(scalar.float())
                else:
                    out[key] = float(scalar.float().item())
            else:
                out[key] = float(scalar)

    if cuda_scalars:
        values = torch.stack(cuda_scalars).cpu().tolist()
        for (out, key), value in zip(cuda_slots, values):
            out[key] = float(value)

    return materialized


def group_loss_values(loss_value: torch.Tensor, group_size: int) -> torch.Tensor:
    if loss_value.numel() == 1:
        return loss_value.reshape(()).expand(group_size)
    if loss_value.numel() == group_size:
        return loss_value.reshape(group_size)
    raise ValueError(f'Expected scalar or {group_size} loss values, got shape {tuple(loss_value.shape)}')


def select_group_log_value(value: Any, position: int, group_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == group_size:
            return value.reshape(group_size)[position]
        return to_log_scalar(value)
    if isinstance(value, np.ndarray):
        if value.size == group_size:
            return value.reshape(group_size)[position]
        return to_log_scalar(value)
    return to_log_scalar(value)


def append_group_log_value(
    group_records: List[Dict[str, Any]],
    key: str,
    value: Any,
    positions: Optional[List[int]] = None,
):
    if positions is None:
        positions = list(range(len(group_records)))
    for local_position, group_position in enumerate(positions):
        scalar = select_group_log_value(value, local_position, len(positions))
        if scalar is not None:
            group_records[group_position][key] = scalar


def append_group_log_dict(
    group_records: List[Dict[str, Any]],
    prefix: str,
    values: Dict[str, Any],
    positions: Optional[List[int]] = None,
):
    for key_tuple, value in flatten_nested_dict(values).items():
        key = '.'.join(key_tuple)
        append_group_log_value(group_records, f'{prefix}.{key}' if key else prefix, value, positions=positions)


def detach_to_cpu(x):
    """Recursively detach tensors and move them to CPU (for debug pickle dumps)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return type(x)(detach_to_cpu(item) for item in x)
    elif isinstance(x, dict):
        return {k: detach_to_cpu(v) for k, v in x.items()}
    return x


def filter_outliers(values: List[float], sigma: float = 5.0) -> List[float]:
    """Filter outlier values using median + MAD (robust to outliers)."""
    if len(values) < 10:
        return values
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = sorted_vals[n // 2]
    mad = sorted([abs(v - median) for v in values])[n // 2]
    if mad < 1e-12:
        return values
    threshold = sigma * mad * 1.4826  # MAD to std conversion for normal distribution
    return [v for v in values if abs(v - median) <= threshold]


ROLLING_CKPT_MANIFEST = 'rolling_ckpts.json'


def cleanup_old_rolling_ckpts(workspace: Path, current_step: int):
    """Keep only `current_step` among the rolling checkpoints this workspace has written.

    Rolling checkpoints are tracked in `checkpoint/rolling_ckpts.json` rather than inferred
    from the step number. Inferring them (e.g. "any step not divisible by checkpoint_every")
    also matches checkpoints written by another run or another training script that used a
    different cadence, and silently deletes them. Only steps this workspace recorded as
    rolling are ever removed; anything else found in the directory is left alone.

    The manifest is written before the deletions so a crash mid-cleanup leaves stale files
    to be collected next time rather than losing track of them.
    """
    ckpt_dir = Path(workspace, 'checkpoint')
    if not ckpt_dir.exists():
        return
    manifest_path = Path(ckpt_dir, ROLLING_CKPT_MANIFEST)

    try:
        tracked = set(json.loads(manifest_path.read_text())) if manifest_path.exists() else set()
    except (json.JSONDecodeError, OSError, TypeError):
        tracked = set()

    stale = sorted(step for step in tracked if step != current_step)
    manifest_path.write_text(json.dumps(sorted(tracked - set(stale) | {current_step})))

    for step in stale:
        for suffix in ('', '_optimizer', '_ema'):
            Path(ckpt_dir, f'{step:08d}{suffix}.pt').unlink(missing_ok=True)


def record_rolling_ckpt(workspace: Path, step: int):
    """Add `step` to the rolling-checkpoint manifest, so a later cleanup may remove it."""
    ckpt_dir = Path(workspace, 'checkpoint')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(ckpt_dir, ROLLING_CKPT_MANIFEST)
    try:
        tracked = set(json.loads(manifest_path.read_text())) if manifest_path.exists() else set()
    except (json.JSONDecodeError, OSError, TypeError):
        tracked = set()
    manifest_path.write_text(json.dumps(sorted(tracked | {step})))


_OPTIMIZER_CONFIG_METADATA_KEYS = {'params', 'type', 'optimizer', 'optimizer_type', 'name'}


def _get_param_group_optimizer_type(optimizer_config: Dict[str, Any], param_group_config: Dict[str, Any]) -> str:
    optimizer_type = param_group_config.get(
        'type',
        param_group_config.get(
            'optimizer_type',
            param_group_config.get('optimizer', optimizer_config.get('type', 'AdamW')),
        ),
    )
    if not isinstance(optimizer_type, str):
        raise TypeError(f'Optimizer type must be a string, got {type(optimizer_type)}')
    return optimizer_type


def _get_torch_optimizer_cls(optimizer_type: str) -> Type[torch.optim.Optimizer]:
    if hasattr(torch.optim, optimizer_type):
        return getattr(torch.optim, optimizer_type)
    optimizer_type_lower = optimizer_type.lower()
    for attr_name in dir(torch.optim):
        if attr_name.lower() == optimizer_type_lower:
            optimizer_cls = getattr(torch.optim, attr_name)
            if isinstance(optimizer_cls, type) and issubclass(optimizer_cls, torch.optim.Optimizer):
                return optimizer_cls
    raise AttributeError(f'torch.optim has no optimizer named {optimizer_type}')


def _optimizer_option_items(config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in config.items() if k not in _OPTIMIZER_CONFIG_METADATA_KEYS}


def _build_named_param_groups(
    model: nn.Module,
    optimizer_config: Dict[str, Any],
) -> Tuple[Dict[str, nn.Parameter], List[Dict[str, nn.Parameter]]]:
    named_parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
    named_param_groups: List[Dict[str, nn.Parameter]] = []
    param_to_group: Dict[str, int] = {}
    duplicated_params: List[str] = []

    for group_idx, param_group_config in enumerate(optimizer_config['params']):
        group_params = {
            k: p
            for k, p in named_parameters.items()
            if any_match(k, param_group_config['params']['include'])
            and not any_match(k, param_group_config['params'].get('exclude', []))
        }
        named_param_groups.append(group_params)
        for name in group_params:
            if name in param_to_group:
                duplicated_params.append(name)
            param_to_group[name] = group_idx

    excluded_params = [k for k in named_parameters if k not in param_to_group]
    assert len(duplicated_params) == 0, f'The following parameters are included in multiple optimizer groups: {duplicated_params}'
    assert len(excluded_params) == 0, f'The following parameters require grad but are excluded from the optimizer: {excluded_params}'
    return named_parameters, named_param_groups


def _normalize_torch_optimizer_options(options: Dict[str, Any]) -> Dict[str, Any]:
    options = dict(options)
    if 'wd' in options:
        options.setdefault('weight_decay', options.pop('wd'))
    if 'adamw_betas' in options:
        options.setdefault('betas', options.pop('adamw_betas'))
    if 'adamw_eps' in options:
        options.setdefault('eps', options.pop('adamw_eps'))
    return options


def _optimizer_group_metadata(param_group_config: Dict[str, Any], optimizer_type: str) -> Dict[str, Any]:
    metadata = {'optimizer_type': optimizer_type}
    if 'name' in param_group_config:
        metadata['name'] = param_group_config['name']
    return metadata


def build_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    named_parameters, named_param_groups = _build_named_param_groups(model, optimizer_config)
    param_group_optimizer_types = [
        _get_param_group_optimizer_type(optimizer_config, param_group_config)
        for param_group_config in optimizer_config['params']
    ]

    unique_optimizer_types = {optimizer_type.lower() for optimizer_type in param_group_optimizer_types}
    if len(unique_optimizer_types) != 1:
        raise ValueError(f'Mixing multiple torch optimizer types in one optimizer is not supported: {param_group_optimizer_types}')

    optimizer_cls = _get_torch_optimizer_cls(param_group_optimizer_types[0])
    optimizer_defaults = _normalize_torch_optimizer_options(_optimizer_option_items(optimizer_config))
    param_groups = [
        {
            **_normalize_torch_optimizer_options(_optimizer_option_items(param_group_config)),
            **_optimizer_group_metadata(param_group_config, optimizer_type),
            'params': list(params.values()),
        }
        for param_group_config, params, optimizer_type in zip(
            optimizer_config['params'],
            named_param_groups,
            param_group_optimizer_types,
        )
    ]
    return optimizer_cls(param_groups, **optimizer_defaults)


def _is_dino_backbone_parameter(name: str) -> bool:
    return name == 'backbone' or name.startswith('backbone.') or '.backbone.' in name


def _is_head_parameter(name: str) -> bool:
    first_name = name.split('.', 1)[0]
    return first_name in {'head', 'neck'} or first_name.endswith('_head')


def _is_refiner_parameter(name: str) -> bool:
    return name == 'refiner' or name.startswith('refiner.') or '.refiner.' in name


def _optimizer_assignment_name(optimizer: torch.optim.Optimizer) -> str:
    return optimizer.__class__.__name__.lower()


def write_optimizer_param_assignment_log(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    workspace: Path,
) -> None:
    group_by_param_id: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        group_options = {k: v for k, v in group.items() if k != 'params'}
        for p in group['params']:
            group_by_param_id[id(p)] = (group_idx, group_options)

    records = []
    summary = Counter()
    numel_summary = Counter()
    for name, parameter in model.named_parameters():
        group_info = group_by_param_id.get(id(parameter))
        if group_info is None:
            assignment = 'not_optimized'
            group_idx = None
            group_options = {}
        else:
            group_idx, group_options = group_info
            assignment = _optimizer_assignment_name(optimizer)

        numel = parameter.numel()
        summary[assignment] += 1
        numel_summary[assignment] += numel
        records.append({
            'name': name,
            'assignment': assignment,
            'requires_grad': parameter.requires_grad,
            'shape': list(parameter.shape),
            'numel': numel,
            'param_group': group_idx,
            'optimizer_type': group_options.get('optimizer_type', optimizer.__class__.__name__),
            'param_group_name': group_options.get('name'),
            'lr': group_options.get('lr'),
            'wd': group_options.get('wd', group_options.get('weight_decay')),
            'is_dino_backbone': _is_dino_backbone_parameter(name),
            'is_head': _is_head_parameter(name),
            'is_refiner': _is_refiner_parameter(name),
        })

    log = {
        'optimizer': optimizer.__class__.__name__,
        'summary': {
            assignment: {
                'parameter_count': summary[assignment],
                'numel': numel_summary[assignment],
            }
            for assignment in sorted(summary)
        },
        'parameters': records,
    }

    workspace.mkdir(parents=True, exist_ok=True)
    json_path = workspace / 'optimizer_param_assignments.json'
    tsv_path = workspace / 'optimizer_param_assignments.tsv'
    with json_path.open('w') as f:
        json.dump(log, f, indent=4)
    with tsv_path.open('w') as f:
        f.write('name\tassignment\trequires_grad\tshape\tnumel\tparam_group\toptimizer_type\tparam_group_name\tlr\twd\tis_dino_backbone\tis_head\tis_refiner\n')
        for record in records:
            f.write(
                f"{record['name']}\t{record['assignment']}\t{record['requires_grad']}\t"
                f"{record['shape']}\t{record['numel']}\t{record['param_group']}\t"
                f"{record['optimizer_type']}\t{record['param_group_name']}\t"
                f"{record['lr']}\t{record['wd']}\t{record['is_dino_backbone']}\t"
                f"{record['is_head']}\t{record['is_refiner']}\n"
            )
    print(f'Optimizer parameter assignment log: {json_path}')
    print(f'Optimizer parameter assignment table: {tsv_path}')


def parse_lr_lambda(s: str) -> Callable[[int], float]:
    epoch = sympy.symbols('epoch')
    lr_lambda = sympy.sympify(s)
    return sympy.lambdify(epoch, lr_lambda, 'math')


def build_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    if scheduler_config['type'] == "SequentialLR":
        child_schedulers = [
            build_lr_scheduler(optimizer, child_scheduler_config)
                for child_scheduler_config in scheduler_config['params']['schedulers']
        ]
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=child_schedulers, milestones=scheduler_config['params']['milestones'])
    elif scheduler_config['type'] == "LambdaLR":
        lr_lambda = scheduler_config['params']['lr_lambda']
        if isinstance(lr_lambda, str):
            lr_lambda = parse_lr_lambda(lr_lambda)
        elif isinstance(lr_lambda, list):
            lr_lambda = [parse_lr_lambda(l) for l in lr_lambda]
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
    else:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config['type'])
        scheduler = scheduler_cls(optimizer, **scheduler_config.get('params', {}))
    return scheduler


def refine_step_pairs(refine_steps: int) -> List[Tuple[int, int]]:
    """The refine-step transitions the monitor tables report, for a run of `refine_steps` steps.

    Every consecutive transition, then the two spans 0->n (what refinement
    achieved overall), and 1->n (the refiner's incremental contribution). 
    """
    pairs = [(i, i + 1) for i in range(refine_steps)]
    for span in ((0, refine_steps), (1, refine_steps)):
        if span[0] < span[1] and span not in pairs:
            pairs.append(span)
    return pairs


def split_step_suffix(key: str) -> Tuple[str, int]:
    """Split a logged key into its base name and refine step: 'global_step_2' -> ('global', 2).

    A key with no suffix is step 0, which is how step-0 losses are logged.
    """
    base, sep, step = key.rpartition('_step_')
    return (base, int(step)) if sep else (key, 0)


def accumulate_step_transitions(
    values_by_step: Dict[str, Dict[int, float]],
    tracker: Dict[Tuple, List[int]],
    count_when: Callable[[float, float], bool],
    pairs: Sequence[Tuple[int, int]],
) -> None:
    """Tally, per (name, step_from, step_to), how many instances satisfy `count_when`.

    `tracker` accumulates `[count, total]`. What the count *means* is decided by
    `count_when(value_at_to, value_at_from)` and must match how the corresponding
    table reports it -- the loss tracker counts instances that got *worse* and its
    table inverts, while the delta and error trackers count the outcome they name.
    """
    for name, step_vals in values_by_step.items():
        for step_from, step_to in pairs:
            if step_from in step_vals and step_to in step_vals:
                entry = tracker.setdefault((name, step_from, step_to), [0, 0])
                entry[1] += 1
                if count_when(step_vals[step_to], step_vals[step_from]):
                    entry[0] += 1


def write_refine_monitor_table(
    pbar,
    i_step: int,
    tracker: Dict[Tuple, Tuple[int, int]],
    log: Dict[str, float],
    title: str,
    label: str,
    log_prefix: str,
    pairs: Sequence[Tuple[int, int]],
    invert: bool = False,
) -> None:
    """Print one "% of instances that improved" table over refine-step transitions.

    `tracker` maps (name, step_from, step_to) -> (count, total). The percentage
    reported is `count / total`, or its complement when `invert` is set -- which
    the loss table needs because it counts instances whose loss *increased* but
    reports the fraction that decreased.

    Consumes the tracker: it is cleared once written. Percentages are also
    written into `log` under `log_prefix` for upload with the next metric batch.
    """
    if not tracker:
        return
    pbar.write(f'[Step {i_step}] {title}')
    names = sorted({name for name, _, _ in tracker})
    name_width = max(len(label), *(len(name) for name in names))
    header = '  '.join(f'{f"{a}->{b}":>7s}' for a, b in pairs)
    pbar.write(f'  {label:<{name_width}s}  {header}')
    for name in names:
        cells = []
        for step_from, step_to in pairs:
            entry = tracker.get((name, step_from, step_to))
            if entry is None:
                cells.append('      -')
                continue
            count, total = entry
            # NOTE: a zero-total cell prints as N/A but is still logged as 0.0,
            # so the metric's key set stays stable across steps.
            pct = 100.0 * ((total - count) if invert else count) / total if total > 0 else 0.0
            cells.append(f'{pct:6.1f}%' if total > 0 else '   N/A')
            log[f'{log_prefix}/{name}_{step_from}_to_{step_to}'] = pct
        pbar.write(f'  {name:<{name_width}s}  {"  ".join(cells)}')
    tracker.clear()
