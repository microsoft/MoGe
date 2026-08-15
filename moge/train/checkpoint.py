"""Checkpoint loading, saving and scheduling for the MoGe training entry points."""
import io
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import *

import torch

from .utils import cleanup_old_rolling_ckpts, record_rolling_ckpt, write_bytes_retry_loop

MA_BUFFER_MAXLEN = 1000


def load_checkpoint(
    ckpt_path: Optional[str],
    workspace: Path,
    accelerator,
    enable_ema: bool,
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint by explicit path, by "latest", or by step number.

    "latest" and step-number forms read `latest.pt` (which holds only a step
    pointer) and then hydrate the model / optimizer / EMA shards written
    alongside it. Returns None when no checkpoint is requested or found.
    """
    with accelerator.local_main_process_first():
        checkpoint = None
        if not ckpt_path or ckpt_path == 'none':
            # - No checkpoint requested
            pass
        elif ckpt_path.endswith('.pt'):
            # - Load specific checkpoint file
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f'Requested checkpoint file does not exist: {ckpt_path}')
            print(f'Load checkpoint: {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        elif ckpt_path == 'latest':
            # - Load latest
            latest_path = Path(workspace, 'checkpoint', 'latest.pt')
            if latest_path.exists():
                print(f'Load checkpoint: {latest_path}')
                checkpoint = torch.load(latest_path, map_location='cpu', weights_only=True)
                # `ckpt_name` is the shard basename. Pointers written before it existed carry the
                # name in `step` instead, where it is the string 'final' for a completed run.
                ckpt_name = checkpoint.get('ckpt_name', checkpoint.get('step'))
                checkpoint = _hydrate_shards(checkpoint, ckpt_name, workspace, accelerator, enable_ema)
        elif ckpt_path.isdigit() or ckpt_path == 'final':
            # - Load by step number, or by the 'final' name a completed run writes
            i_step = int(ckpt_path) if ckpt_path.isdigit() else 'final'
            ckpt_name = f'{i_step:08d}' if isinstance(i_step, int) else i_step
            if Path(workspace, 'checkpoint', f'{ckpt_name}.pt').exists():
                checkpoint = _hydrate_shards({'step': i_step}, i_step, workspace, accelerator, enable_ema)
            else:
                # Returning None rather than a shard-less dict is what lets the caller fall back
                # to `--initial_checkpoint`, as that option's help text promises. Hydrating here
                # would instead hand back `{'step': N}` and hard-fail in restore_training_state.
                print(f"Warning: no checkpoint named '{ckpt_name}.pt' under {Path(workspace, 'checkpoint')}")
        else:
            raise ValueError(
                f'Unrecognized checkpoint specifier {ckpt_path!r}; expected a path ending in .pt, '
                "'latest', 'final', 'none', or a step number"
            )
    return checkpoint


def _hydrate_shards(checkpoint, i_step, workspace, accelerator, enable_ema):
    ckpt_name = f'{i_step:08d}' if isinstance(i_step, int) else i_step
    if 'model' not in checkpoint and (path := Path(workspace, 'checkpoint', f'{ckpt_name}.pt')).exists():
        print(f'Load model checkpoint: {path}')
        checkpoint['model'] = torch.load(path, map_location='cpu', weights_only=True)['model']
    if 'optimizer' not in checkpoint and (path := Path(workspace, 'checkpoint', f'{ckpt_name}_optimizer.pt')).exists():
        print(f'Load optimizer checkpoint: {path}')
        checkpoint.update(torch.load(path, map_location='cpu', weights_only=True))
    if enable_ema and accelerator.is_main_process:
        if 'ema_model' not in checkpoint and (path := Path(workspace, 'checkpoint', f'{ckpt_name}_ema.pt')).exists():
            print(f'Load EMA model checkpoint: {path}')
            ema_checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            checkpoint['ema_model'] = ema_checkpoint['model']
            if 'ema_n_averaged' in ema_checkpoint:
                checkpoint['ema_n_averaged'] = ema_checkpoint['ema_n_averaged']
    return checkpoint


def restore_training_state(
    checkpoint: Optional[Dict[str, Any]],
    model,
    optimizer,
    lr_scheduler,
    ema_model,
    accelerator,
    enable_ema: bool,
) -> int:
    """Apply a checkpoint to the training state, or initialise from scratch.

    Weights are always initialised first and the checkpoint is then loaded on top,
    so any module the checkpoint does not carry keeps a proper initialisation.
    `strict=False` is load-bearing for that: a base checkpoint from a non-refiner
    run has no `refiner.*` keys, and the refiner relies on `init_weights()` zeroing
    its output projection so its residual starts as an exact identity. Merely
    leaving it at its constructor initialisation makes the first refine steps a
    random perturbation of the base prediction.
    Returns the step to resume from.
    """
    if checkpoint is None:
        print('Initialize model weights')
        with accelerator.local_main_process_first():
            model.init_weights()
        if enable_ema and accelerator.is_main_process:
            ema_model.module.load_state_dict(model.state_dict())
        return 0

    if 'model' not in checkpoint:
        raise FileNotFoundError(
            f"Checkpoint step {checkpoint.get('step', '?')} has no model state; "
            "the checkpoint may be incomplete or the requested step may not exist"
        )

    print('Initialize model weights before loading checkpoint')
    with accelerator.local_main_process_first():
        model.init_weights()
    model.load_state_dict(checkpoint['model'], strict=False)
    step = checkpoint.get('step')
    if isinstance(step, int):
        initial_step = step + 1
        print(f'Resume from step {initial_step}')
    elif step is None:
        initial_step = 0
        print('No step info found in checkpoint, start from step 0')
    else:
        # `latest.pt` pointers written before `ckpt_name` existed store the string 'final' here,
        # and the real step number only arrives with the optimizer shard. When that shard is
        # missing there is no step to resume from, so start over instead of crashing on
        # `'final' + 1`.
        initial_step = 0
        print(f'Warning: checkpoint step is {step!r}, not a step number; start from step 0')
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('Warning: No optimizer state found in checkpoint, optimizer is re-initialized')
    if enable_ema and accelerator.is_main_process:
        if 'ema_model' in checkpoint:
            ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
            # Restoring `n_averaged` is what keeps the average alive: at 0, torch's
            # `update_parameters()` overwrites every EMA parameter with the live model.
            # Checkpoints written before this key existed fall back to 1 rather than 0,
            # which preserves the restored weights (the exact count only matters to
            # `avg_fn`, and the configured one ignores it).
            n_averaged = checkpoint.get('ema_n_averaged')
            if n_averaged is None:
                n_averaged = 1
                print('Warning: EMA checkpoint has no n_averaged (written before this was tracked); assuming 1')
            ema_model.n_averaged.fill_(int(n_averaged))
        else:
            ema_model.module.load_state_dict(model.state_dict())
            ema_model.n_averaged.zero_()
            print('Warning: EMA enabled but no EMA state found; initialized EMA from the loaded model')
    if 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        print('Warning: No lr_scheduler state found in checkpoint, lr_scheduler is re-initialized')
    return initial_step


def restore_ma_buffer(workspace: Path, initial_step: int, accelerator) -> deque:
    """Restore the moving-average buffer so ma1000 curves stay continuous on resume."""
    ma_buffer = deque(maxlen=MA_BUFFER_MAXLEN)
    if initial_step > 0 and accelerator.is_main_process:
        path = Path(workspace, 'checkpoint', 'latest_ma_buffer.pt')
        if path.exists():
            state = torch.load(path, map_location='cpu', weights_only=False)
            ma_buffer = deque(state.get('ma_buffer', []), maxlen=MA_BUFFER_MAXLEN)
            print(f"Restored ma_buffer ({len(ma_buffer)} entries) from step {state.get('step', '?')}")
        else:
            print('Warning: No ma_buffer state found, ma1000 metrics will restart from the beginning')
    return ma_buffer


def restore_data_pipeline_states(
    workspace: Path,
    initial_step: int,
    accelerator,
    pipelines: Dict[str, Any],
) -> None:
    """Restore per-rank dataloader RNG states that match the resumed step."""
    if initial_step <= 0:
        return
    expected_step = initial_step - 1
    for name, pipeline in pipelines.items():
        path = Path(
            workspace, 'checkpoint', 'data_pipeline',
            f'latest_{name}_data_pipeline_rank_{accelerator.process_index}.pt',
        )
        if not path.exists():
            print(f'Warning: No {name} data pipeline state found; data ordering will restart')
            continue
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('step') != expected_step:
            print(
                f"Warning: {name} data pipeline state is from step {state.get('step', '?')}, "
                f'not requested step {expected_step}; data ordering will restart'
            )
            continue
        pipeline.load_state_dict(state)
        print(f'Restored {name} data pipeline state for rank {accelerator.process_index} from step {expected_step}')


def save_data_pipeline_states(
    workspace: Path,
    i_step: int,
    accelerator,
    pipelines: Dict[str, Any],
) -> None:
    """Save each dataloader RNG state separately for every distributed rank."""
    state_dir = Path(workspace, 'checkpoint', 'data_pipeline')
    state_dir.mkdir(parents=True, exist_ok=True)
    for name, pipeline in pipelines.items():
        path = state_dir / f'latest_{name}_data_pipeline_rank_{accelerator.process_index}.pt'
        torch.save({'step': i_step, **pipeline.state_dict()}, path)


class CheckpointSaver:
    """Writes the checkpoint shards for a run and decides when to write them.

    Writing goes through a single-worker executor so a slow (often network-mounted)
    filesystem does not stall training. Built once before the loop rather than as a
    per-step closure.
    """

    def __init__(
        self,
        workspace: Path,
        config: Dict[str, Any],
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        ema_model,
        enable_ema: bool,
        ma_buffer: deque,
        executor: ThreadPoolExecutor,
        pbar,
        num_iterations: int,
        checkpoint_every: int,
        rolling_checkpoint_every: int,
        initial_step: int,
    ):
        self.workspace = workspace
        self.config = config
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema_model = ema_model
        self.enable_ema = enable_ema
        self.ma_buffer = ma_buffer
        self.executor = executor
        self.pbar = pbar
        self.num_iterations = num_iterations
        self.checkpoint_every = checkpoint_every
        self.rolling_checkpoint_every = rolling_checkpoint_every
        self.initial_step = initial_step

    def _write(self, name: str, payload: Dict[str, Any], async_save: bool):
        with io.BytesIO() as f:
            torch.save(payload, f)
            data = f.getvalue()
        path = Path(self.workspace, 'checkpoint', name)
        if async_save:
            self.executor.submit(write_bytes_retry_loop, path, data)
        else:
            write_bytes_retry_loop(path, data)

    def save(self, i_step: int, async_save: bool = True):
        ckpt_name = 'final' if i_step == self.num_iterations - 1 else f'{i_step:08d}'
        self.pbar.write(f'Save checkpoint: {i_step:08d}')
        Path(self.workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)

        model_config = self.config['model']
        self._write(f'{ckpt_name}.pt', {
            'model_config': model_config,
            'model': self.accelerator.unwrap_model(self.model).state_dict(),
        }, async_save)
        self._write(f'{ckpt_name}_optimizer.pt', {
            'model_config': model_config,
            'step': i_step,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }, async_save)
        if self.enable_ema:
            # `n_averaged` lives on the AveragedModel wrapper, not on `.module`, so it must be
            # saved alongside the weights. Without it, a resumed run restores `n_averaged == 0`
            # and torch's first `update_parameters()` hard-copies the live model over the EMA,
            # silently discarding the average. Kept as a sibling key so `model` stays a bare
            # state_dict that moge.model.v1/v2 `from_pretrained` can load directly.
            self._write(f'{ckpt_name}_ema.pt', {
                'model_config': model_config,
                'model': self.ema_model.module.state_dict(),
                'ema_n_averaged': int(self.ema_model.n_averaged),
            }, async_save)
        # `step` stays an int so a resume can always derive the next step without depending on
        # the optimizer shard; `ckpt_name` carries the shard basename, which is 'final' for the
        # last iteration and `{step:08d}` otherwise.
        self._write('latest.pt', {'model_config': model_config, 'step': i_step, 'ckpt_name': ckpt_name}, async_save)
        self._write('latest_ma_buffer.pt', {'step': i_step, 'ma_buffer': list(self.ma_buffer)}, async_save)

    def is_due(self, i_step: int) -> bool:
        """Whether this step warrants a checkpoint (permanent, rolling or final)."""
        return self._classify(i_step)[0]

    def _classify(self, i_step: int) -> Tuple[bool, bool]:
        is_final = (i_step == self.num_iterations - 1)
        is_permanent = (
            self.checkpoint_every > 0
            and i_step % self.checkpoint_every == 0
            and i_step != self.initial_step
        )
        # A final step is saved under the name 'final', not '{step:08d}', so it must never be
        # tracked as rolling: the manifest entry would name files that were never written, the
        # 'final' shards would leak, and a later cleanup would delete an identically-numbered
        # checkpoint that some other run legitimately wrote into this workspace.
        is_rolling = (
            self.rolling_checkpoint_every > 0
            and i_step % self.rolling_checkpoint_every == 0
            and i_step != self.initial_step
            and not is_permanent
            and not is_final
        )
        return is_permanent or is_rolling or is_final, is_rolling

    def save_if_due(self, i_step: int) -> bool:
        """Write a permanent, rolling or final checkpoint if this step calls for one."""
        due, is_rolling = self._classify(i_step)
        if self.accelerator.is_main_process and due:
            self.save(i_step)
            # For rolling checkpoints, drop the previous rolling one. Record this step
            # first so a crash before cleanup leaves it tracked rather than orphaned.
            if is_rolling:
                record_rolling_ckpt(self.workspace, i_step)
                self.executor.submit(cleanup_old_rolling_ckpts, self.workspace, i_step)
        return due
