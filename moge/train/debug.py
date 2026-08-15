"""Debug dumping for training runs that hit NaNs or unusually large gradients."""
from pathlib import Path
from typing import *

import torch

from .utils import detach_to_cpu


class DebugDumper:
    """Collects dump triggers during a step and writes the offending state to disk.

    Call `add_reason(tag)` from anywhere in the step to flag it. Tags starting with
    `nan_` are fatal and count toward the abort threshold; anything else (e.g.
    `large_grad_norm_12.34`) is informational and capped so a pathological run
    cannot fill the disk. Call `begin_step()` once per step and `flush(...)` once per
    accumulation step.

    Attributing a non-finite gradient
    ---------------------------------
    `clip_grad_norm_` only runs on the sync micro-step, and by then `.grad` holds the
    sum over every micro-batch of the step *and* the DDP all-reduce across every rank.
    A non-finite norm there therefore says nothing about which sample caused it: the
    batch in scope is one of `num_processes * gradient_accumulation_steps` candidates,
    and because the all-reduce spreads the NaN, every rank reports it and dumps its own
    innocent micro-batch.

    `check_grads` closes that gap. It runs after every micro-step's backward and flags
    the transition from all-finite to non-finite, so the dump carries the micro-batch
    that actually introduced the corruption. Before the sync micro-step nothing has
    been all-reduced yet, so that attribution is rank-local and exact; on the sync
    micro-step the gradient is already pooled and the tag says so.
    """

    def __init__(
        self,
        workspace: Path,
        accelerator,
        model,
        dump_grad_norm_above: Optional[float] = None,
        max_nan_dumps_before_abort: int = 10,
        max_extra_dumps: int = 25,
        save_model_on_first_dump: bool = False,
    ):
        self.workspace = workspace
        self.accelerator = accelerator
        # Unwrapped, so parameter names match the checkpoint and the optimizer
        # assignment log rather than carrying DDP's `module.` prefix. DDP shares the
        # underlying parameter objects, so the gradients are the same tensors.
        self.model = accelerator.unwrap_model(model)
        self.dump_grad_norm_above = dump_grad_norm_above
        self.max_nan_dumps_before_abort = max_nan_dumps_before_abort
        self.max_extra_dumps = max_extra_dumps
        self.save_model_on_first_dump = save_model_on_first_dump
        self.reasons: List[str] = []
        self.nan_encountered_times = 0
        self.extra_dump_count = 0
        self.model_saved = False
        # Gradient attribution state, reset per step by `begin_step`.
        self.grads_flagged = False
        self.grad_nonfinite: List[Dict[str, Any]] = []

    def add_reason(self, tag: str):
        self.reasons.append(tag)

    def begin_step(self):
        """Reset the per-step gradient attribution state.

        Gradients accumulate across a step's micro-steps and are only zeroed on the
        sync one, so "have they already gone bad?" is a per-step question.
        """
        self.grads_flagged = False
        self.grad_nonfinite = []

    def note_grad_norm(self, grad_norm, grad_norm_is_finite: bool):
        """Flag an unusually large but finite gradient norm, if a threshold was configured.

        The threshold and quota are checked before reading the value, so the
        `.cpu()` sync only happens when the feature is actually armed.
        """
        if (
            self.dump_grad_norm_above is not None
            and grad_norm_is_finite
            and self.extra_dump_count < self.max_extra_dumps
        ):
            grad_norm_value = float(grad_norm.detach().cpu().item())
            if grad_norm_value > self.dump_grad_norm_above:
                self.add_reason(f'large_grad_norm_{grad_norm_value:.2f}')

    def check_grads(self, i_accumulate: int, synced: bool) -> bool:
        """Screen the accumulated gradients after one micro-step's backward.

        Returns whether they still look finite. Only the micro-step that first trips
        the screen is flagged; gradients accumulate, so every later micro-step in the
        same step inherits the corruption and would otherwise each claim it.
        """
        named_grads = [(name, p.grad) for name, p in self.model.named_parameters() if p.grad is not None]
        if not named_grads:
            return True

        # Fused screen: one multi-tensor kernel rather than one per parameter, which is
        # ~16x cheaper on a 370M-parameter model (1.6 ms vs 25 ms). This is the same
        # quantity `clip_grad_norm_` computes, so it errs in the useful direction: it
        # never misses a non-finite gradient, and it also trips on a finite-but-huge one
        # whose squares overflow -- exactly what would give a non-finite clipped norm.
        finite = bool(torch.stack(torch._foreach_norm([g for _, g in named_grads])).isfinite().all().item())
        if finite or self.grads_flagged:
            return finite

        self.grads_flagged = True
        # Rare path only: name the offending parameters and separate a NaN (an invalid
        # op such as 0/0) from an Inf (an overflow). Enough to localise the module
        # without dumping any weights.
        self.grad_nonfinite = [
            {
                'name': name,
                'num_nan': int(torch.isnan(grad).sum()),
                'num_inf': int(torch.isinf(grad).sum()),
                'numel': grad.numel(),
            }
            for name, grad in named_grads if not torch.isfinite(grad).all()
        ]
        where = 'after_sync' if synced else f'local_accum{i_accumulate}'
        if self.grad_nonfinite:
            self.add_reason(f'nan_grad_{where}')
        else:
            # Every individual gradient is finite yet their norm is not, so the sum of
            # squares overflowed rather than a NaN propagating: a different failure.
            self.add_reason(f'nan_gradnorm_overflow_{where}')
        return False

    def _save_model_once(self):
        """Snapshot the weights the first time we dump, so a step > 0 can be replayed.

        Step 0 needs no snapshot (the weights are still the initial checkpoint), and a
        per-event snapshot would be ~1.5 GB times every rank and every event, so this
        is one file for the whole run and only on the main process.
        """
        if self.model_saved or not self.accelerator.is_main_process:
            return
        self.model_saved = True
        path = Path(self.workspace, 'debug', 'model_at_first_dump.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': self.model.state_dict()}, path)

    def _dump(self, step: int, accumulate_step: int, batch: Any, output: Any, meta: Optional[Dict[str, Any]]) -> Path:
        """Dump the flagged micro-batch so the failing forward pass can be replayed offline."""
        dump_path = Path(
            self.workspace,
            'debug',
            f'step_{step:08d}_accum_{accumulate_step}_proc_{self.accelerator.process_index}'
            f'_reasons_{self.reasons[0]}.pkl',
        )
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open('wb') as f:
            torch.save({
                'batch': detach_to_cpu(batch),
                'output': detach_to_cpu(output),
                'reasons': self.reasons,
                'grad_nonfinite': self.grad_nonfinite,
                'meta': {
                    'step': step,
                    'accumulate_step': accumulate_step,
                    'process_index': self.accelerator.process_index,
                    'num_processes': self.accelerator.num_processes,
                    **(meta or {}),
                },
            }, f)
        return dump_path

    def flush(self, i_step: int, i_accumulate: int, batch: Any, output: Any, meta: Optional[Dict[str, Any]] = None):
        """Write a dump if anything flagged this step, and abort after too many NaNs."""
        if not self.reasons:
            return
        if any(r.startswith('nan_') for r in self.reasons):
            self.nan_encountered_times += 1
        else:
            self.extra_dump_count += 1
        # A gradient that only went bad once it was all-reduced is shared by every
        # rank, so every rank would write an identical dump of its own unrelated
        # micro-batch. Keep a single witness instead of `num_processes` of them.
        bystander = all(r.endswith('after_sync') for r in self.reasons)
        if not bystander or self.accelerator.is_main_process:
            if self.save_model_on_first_dump:
                self._save_model_once()
            self._dump(i_step, i_accumulate, batch, output, meta)
        self.reasons = []
        if self.nan_encountered_times >= self.max_nan_dumps_before_abort:
            raise RuntimeError('NaN encountered too many times, abort training.')
