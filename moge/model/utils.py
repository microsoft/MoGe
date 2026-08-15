from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint
    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__
        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
        
    module.__class__ = _CheckpointingWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    module.__class__ = module.__class__._restore_cls


def sync_ddp_hook(state, bucket: torch.distributed.GradBucket) -> torch.futures.Future[torch.Tensor]:
    group_to_use = torch.distributed.group.WORLD
    world_size = group_to_use.size()
    grad = bucket.buffer()
    grad.div_(world_size)
    torch.distributed.all_reduce(grad, group=group_to_use)
    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut


class AutocastHandle:
    """Handle returned by `wrap_module_with_autocast`. Call `remove` to undo the wrapping."""

    def __init__(self, pre_handle, post_handle):
        self._pre_handle = pre_handle
        self._post_handle = post_handle
        self._removed = False

    def remove(self) -> None:
        if self._removed:
            return
        self._pre_handle.remove()
        self._post_handle.remove()
        self._removed = True


def wrap_module_with_autocast(module: nn.Module, **autocast_kwargs) -> AutocastHandle:
    """Run `module`'s forward inside a `torch.autocast(**autocast_kwargs)` context, via forward hooks.

    The context is entered in a pre-hook and exited in a post-hook registered with
    `always_call=True`, so it is closed even if forward raises. The post-hook uses
    `prepend=True` so that stacked wrappers unwind in LIFO order.
    """
    cm_stack: List[torch.autocast] = []

    def _pre_hook(_module, _args, _kwargs):
        cm = torch.autocast(**autocast_kwargs)
        cm.__enter__()
        cm_stack.append(cm)

    def _post_hook(_module, _args, _kwargs, output):
        if cm_stack:
            cm_stack.pop().__exit__(None, None, None)
        return output

    pre_handle = module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    post_handle = module.register_forward_hook(_post_hook, with_kwargs=True, always_call=True, prepend=True)
    return AutocastHandle(pre_handle, post_handle)
