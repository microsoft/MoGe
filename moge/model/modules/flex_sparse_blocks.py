from typing import Literal, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flex_gemm.nn import (
    SubmanifoldConv3d,
    SparsePool3d,
    SparseUpsample3d,
)


def _pick_spconv_algorithm(in_channels: int, out_channels: int) -> str:
    """Pick the submanifold-conv index-GEMM variant by channel width.

    Wide GEMMs (max(in, out) >= 128) benefit from `masked_implicit_gemm`;
    narrow ones stay on the default `implicit_gemm`.
    """
    if max(in_channels, out_channels) >= 128:
        return "masked_implicit_gemm"
    return "implicit_gemm"


def make_conv3d(in_channels: int, out_channels: int, kernel_size: int = 3) -> SubmanifoldConv3d:
    return SubmanifoldConv3d(
        in_channels, out_channels, kernel_size,
        algorithm=_pick_spconv_algorithm(in_channels, out_channels),
    )


def _with_channels(shape: torch.Size, channels: int) -> torch.Size:
    return torch.Size(list(shape[:-1]) + [channels])


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _ceil_downsample_shape(shape: torch.Size, sparse_dim: int, factor: int) -> torch.Size:
    sparse_shape = list(shape[:sparse_dim])
    dense_shape = list(shape[sparse_dim:])
    for dim in range(sparse_dim - 3, sparse_dim):
        sparse_shape[dim] = _ceil_div(int(sparse_shape[dim]), factor)
    return torch.Size([*sparse_shape, *dense_shape])


class SparseResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = nn.LayerNorm(channels, elementwise_affine=True, eps=1e-6)
        self.activation_fn = F.silu

        self.conv1 = make_conv3d(channels, self.out_channels)
        self.conv2 = make_conv3d(self.out_channels, self.out_channels)
        self.skip_connection = (
            nn.Linear(channels, self.out_channels)
            if channels != self.out_channels else nn.Identity()
        )

    def init_weights(self):
        for parameter in self.conv2.parameters():
            nn.init.zeros_(parameter)

    def forward(self, feats, coords, shape, neighbor_cache=None):
        h = self.activation_fn(self.norm1(feats).type_as(feats))
        h, neighbor_cache = self.conv1(h, coords, shape, neighbor_cache=neighbor_cache)
        h = self.activation_fn(h)
        h, neighbor_cache = self.conv2(h, coords, shape, neighbor_cache=neighbor_cache)
        return h + self.skip_connection(feats), neighbor_cache


class PoolDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        self.factor = factor
        self.pool = SparsePool3d(kernel_size=factor, stride=factor, reduce="mean")
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, feats, coords, shape):
        output_shape = _ceil_downsample_shape(shape, coords.shape[1], self.factor)
        feats, coords, shape, down_cache = self.pool(feats, coords, shape, output_shape=output_shape)
        feats = self.linear(feats)
        return feats, coords, _with_channels(shape, feats.shape[-1]), down_cache


class NearestUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        self.upsample = SparseUpsample3d(scale_factor=factor, mode="nearest")
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, feats, coords, shape, target_coords, target_shape, up_cache=None):
        feats = self.linear(feats)
        shape = _with_channels(shape, feats.shape[-1])
        feats, coords, shape, _ = self.upsample(
            feats, coords, shape,
            output_coords=target_coords, output_shape=target_shape,
            neighbor_cache=up_cache,
        )
        return feats, coords, shape