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


def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    assert torch.__version__ >= '2.0', "SDPA requires PyTorch 2.0 or later"
    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)      # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C) 

            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    module.__class__ = _AttentionWrapper
    return module


def sync_ddp_hook(state, bucket: torch.distributed.GradBucket) -> torch.futures.Future[torch.Tensor]:
    group_to_use = torch.distributed.group.WORLD
    world_size = group_to_use.size()
    grad = bucket.buffer()
    grad.div_(world_size)
    torch.distributed.all_reduce(grad, group=group_to_use)
    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut
