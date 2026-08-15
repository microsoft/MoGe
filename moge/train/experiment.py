"""Run scaffolding shared by the MoGe training entry points: the Accelerator,
the logging backends, and the periodic metric upload.
"""
import json
import traceback
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import *

import click
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

from ..utils.tools import key_average
from .utils import filter_outliers, materialize_log_records

# Diagnostics whose magnitude carries no signal when averaged over a window.
_MA_EXCLUDE_SUFFIXES = ('num_groups', 'points_per_group')


def setup_accelerator(
    gradient_accumulation_steps: int,
    find_unused_parameters: bool,
    batch_size_forward: int,
    workspace_path: str,
) -> Tuple[Accelerator, Any, int, Path]:
    """Build the Accelerator and derive the values every trainer needs from it."""
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters),
            InitProcessGroupKwargs(timeout=timedelta(hours=1))
        ]
    )
    batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes
    return accelerator, accelerator.device, batch_size_total, Path(workspace_path)


class RunLogger:
    """Fans metrics out to whichever of mlflow / tensorboard / wandb were requested.

    Holding the backends together avoids the failure mode of the original inline
    code, where `log_type_set` and the `mlflow` module were bound only inside an
    `if accelerator.is_main_process:` block but read from several places later --
    which worked only because every reader happened to be main-process-guarded too.
    Here a non-main rank simply gets a logger with no backends.
    """

    def __init__(self, accelerator: Accelerator, log_type: Iterable[str]):
        self.accelerator = accelerator
        self.log_type = set(log_type)
        self.tb_writer = None
        self.wandb_run = None
        self.mlflow = None

    def __contains__(self, backend: str) -> bool:
        return backend in self.log_type

    def setup(
        self,
        workspace: Path,
        config: Dict[str, Any],
        experiment_name: str,
        tb_log_root: Optional[str],
        wandb_project: str,
        batch_size_total: int,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Initialise the backends and persist the run's config. Main process only."""
        if not self.accelerator.is_main_process:
            return {}

        try:
            import git
            current_git_commit_id = git.Repo(search_parent_directories=True).head.object.hexsha
        except Exception:
            current_git_commit_id = 'N/A'
        experiment_params = {
            **click.get_current_context().params,
            **(extra_params or {}),
            'batch_size_total': batch_size_total,
            'git_commit_id': current_git_commit_id,
        }

        if 'mlflow' in self.log_type:
            try:
                import mlflow
                mlflow.log_params(experiment_params)
                self.mlflow = mlflow
            except Exception:
                print('Failed to log config to MLFlow')
                traceback.print_exc()
        if 'tensorboard' in self.log_type:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=Path(tb_log_root or './tensorboard/', experiment_name))
                self.tb_writer.add_text('params', json.dumps(experiment_params, indent=4))
                self.tb_writer.flush()
            except Exception:
                print('Failed to log config to TensorBoard')
                traceback.print_exc()
        if 'wandb' in self.log_type:
            try:
                import wandb
                self.wandb_run = wandb.init(name=experiment_name, config=experiment_params, project=wandb_project)
            except Exception:
                print('Failed to log config to Weights & Biases')
                traceback.print_exc()

        workspace.mkdir(parents=True, exist_ok=True)
        with workspace.joinpath('config.json').open('w') as f:
            json.dump(config, f, indent=4)
        with workspace.joinpath('experiment_params.json').open('w') as f:
            json.dump(experiment_params, f, indent=4)
        return experiment_params

    def log_images(self, images: Dict[str, Any], step: int):
        """Upload images to whichever backend supports them (currently mlflow only)."""
        if self.mlflow is None:
            return
        try:
            for key, image in images.items():
                self.mlflow.log_image(image, key=key, step=step)
        except Exception as e:
            print(f'Failed to log image to mlflow: {e}')

    def _upload(self, records: Dict[str, float], step: int):
        if self.mlflow is not None:
            try:
                self.mlflow.log_metrics(records, step=step)
            except Exception as e:
                print(f'Error while logging metrics to mlflow: {e}')
                traceback.print_exc()
        if self.tb_writer is not None:
            try:
                for k, v in records.items():
                    self.tb_writer.add_scalar(k, v, step)
                self.tb_writer.flush()
            except Exception:
                print('Error while logging metrics to TensorBoard')
                traceback.print_exc()
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(records, step=step)
            except Exception:
                print('Error while logging metrics to Weights & Biases')
                traceback.print_exc()

    def log_metrics(
        self,
        records: List[Dict[str, Any]],
        ma_buffer: deque,
        lr_scheduler,
        i_step: int,
        initial_step: int,
        extra_scalars: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate, gather across ranks, augment and upload. Returns a fresh record list.

        Must be called on EVERY rank: `gather_for_metrics` is collective.
        """
        records = [key_average(materialize_log_records(records))]
        self.accelerator.wait_for_everyone()
        records = self.accelerator.gather_for_metrics(records, use_gather_object=True)
        if self.accelerator.is_main_process:
            records = key_average(records)

            # Moving average over the last 1000 log points for loss and misc metrics.
            loss_misc_snapshot = {
                k: v for k, v in records.items()
                if k.startswith(('loss/', 'misc/')) and not k.endswith(_MA_EXCLUDE_SUFFIXES)
            }
            ma_buffer.append(loss_misc_snapshot)
            for k in loss_misc_snapshot:
                values = filter_outliers([d[k] for d in ma_buffer if k in d])
                if values:
                    prefix, rest = k.split('/', 1)
                    records[f'ma1000_{prefix}/{rest}'] = sum(values) / len(values)

            # Label each LR curve by its optimizer param-group name rather than by position, so a
            # config that reorders `optimizer.params` cannot silently mislabel the curves.
            last_lrs = lr_scheduler.get_last_lr()
            optimizer = getattr(lr_scheduler, 'optimizer', None)
            group_names = [g.get('name') for g in optimizer.param_groups] if optimizer is not None else []
            records['train/lr'] = last_lrs[0]
            for idx, lr in enumerate(last_lrs):
                name = group_names[idx] if idx < len(group_names) and group_names[idx] else f'group{idx}'
                records[f'train/lr_{name}'] = lr

            if extra_scalars:
                records.update(extra_scalars)

            self._upload(records, i_step)
        return []
