from typing import *
import math
import warnings

import torch
import torch.nn as nn

from .flex_sparse_blocks import (
    SparseResBlock3d,
    PoolDown,
    NearestUp,
)
from ..utils import wrap_module_with_gradient_checkpointing, wrap_module_with_autocast
from flex_gemm.ops import NeighborCache


class Sparse3DUNet(nn.Module):
    """
    Generic sparse 3D UNet with separated resampling and residual refinement.

    Operates purely on sparse features -- it has no knowledge of point maps,
    log-depth, voxelization, or UV. The caller is responsible for building the
    sparse representation (feats/coords/shape) and interpreting the output.

    forward(feats, coords, shape, encoder_feature):
    - feats: (M, in_channels) raw input features.
    - coords: (M, 4) int32, columns (batch, i, j, z_bin).
    - shape: Size([B, H, W, Z, in_channels]).
    - encoder_feature: (B, C_enc, H/encoder_downsample, W/encoder_downsample),
      a dense conditioning map sampled at the bottleneck coords.
    Returns: (M, out_channels).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_channels: int,
        model_channels: List[int],
        encoder_blocks_per_level: Union[int, List[int]],
        decoder_blocks_per_level: Union[int, List[int]],
        bottleneck_blocks: int = 1,
        downsample_factors: Optional[List[int]] = None,
        encoder_downsample: int = 16,
        **deprecated_kwargs,
    ):
        super().__init__()
        if deprecated_kwargs:
            warnings.warn(f"Warning: Sparse3DUNet got unexpected kwargs: {deprecated_kwargs}")
        if len(model_channels) < 2:
            raise ValueError(f"model_channels must have at least 2 levels, got {model_channels}")
        if downsample_factors is None:
            downsample_factors = [2] * (len(model_channels) - 1)
        if len(downsample_factors) != len(model_channels) - 1:
            raise ValueError(
                f"downsample_factors must have length {len(model_channels) - 1}, got {len(downsample_factors)}"
            )
        if any(f <= 1 for f in downsample_factors):
            raise ValueError(f"All downsample_factors must be > 1, got {downsample_factors}")
        if bottleneck_blocks < 0:
            raise ValueError(f"bottleneck_blocks must be non-negative, got {bottleneck_blocks}")

        self.encoder_downsample = encoder_downsample
        self.downsample_factors = downsample_factors
        assert self.encoder_downsample == math.prod(self.downsample_factors), \
            f"encoder_downsample ({self.encoder_downsample}) must equal the product of downsample_factors ({math.prod(self.downsample_factors)})"

        encoder_block_counts = self._resolve_blocks_per_level(
            encoder_blocks_per_level, len(model_channels), 'encoder_blocks_per_level'
        )
        decoder_block_counts = self._resolve_blocks_per_level(
            decoder_blocks_per_level, len(model_channels) - 1, 'decoder_blocks_per_level'
        )

        self.input_proj = nn.Linear(in_channels, model_channels[0])
        self.encoder_fuse = nn.Linear(encoder_channels, model_channels[-1])
        bottleneck_channels = model_channels[-1]
        self.fuse_proj = nn.Sequential(
            nn.Linear(bottleneck_channels * 2, bottleneck_channels),
            nn.SiLU(),
            nn.Linear(bottleneck_channels, bottleneck_channels),
        )

        self.down_stages = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        for i, ch in enumerate(model_channels):
            self.down_stages.append(
                self._make_stage(ch, encoder_block_counts[i])
            )
            if i < len(model_channels) - 1:
                self.downsample_blocks.append(
                    PoolDown(model_channels[i], 
                             model_channels[i + 1], 
                             self.downsample_factors[i])
                )

        self.bottleneck_stage = self._make_stage(
            model_channels[-1], bottleneck_blocks
        )

        self.upsample_blocks = nn.ModuleList()
        self.up_stages = nn.ModuleList()
        for i in range(len(model_channels) - 1):
            source_level = len(model_channels) - 1 - i
            target_level = source_level - 1
            self.upsample_blocks.append(
                NearestUp(model_channels[source_level], 
                          model_channels[target_level],
                          self.downsample_factors[target_level])
            )
            self.up_stages.append(
                self._make_stage(model_channels[target_level], decoder_block_counts[i])
            )

        self.out_proj = nn.Linear(model_channels[0], out_channels)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, SparseResBlock3d):
                module.init_weights()
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _resolve_blocks_per_level(
        blocks_per_level: Union[int, List[int]],
        num_levels: int,
        name: str,
    ) -> List[int]:
        if isinstance(blocks_per_level, int):
            blocks_per_level = [blocks_per_level] * num_levels
        if len(blocks_per_level) != num_levels:
            raise ValueError(f"{name} must have length {num_levels}, got {blocks_per_level}")
        if any(blocks < 0 for blocks in blocks_per_level):
            raise ValueError(f"{name} must be non-negative, got {blocks_per_level}")
        return list(blocks_per_level)

    def _make_stage(self, channels: int, num_blocks: int) -> nn.ModuleList:
        return nn.ModuleList([
            SparseResBlock3d(channels)
            for _ in range(num_blocks)
        ])

    def enable_gradient_checkpointing(self):
        for stage in [*self.down_stages, self.bottleneck_stage, *self.up_stages]:
            for block in stage:
                wrap_module_with_gradient_checkpointing(block)

    def enable_mixed_precision(self, dtype: torch.dtype = torch.bfloat16):
        if getattr(self, '_autocast_handle', None) is not None:
            self._autocast_handle.remove()
        self._autocast_handle = wrap_module_with_autocast(self, device_type='cuda', dtype=dtype)

    def _sample_encoder_feature(
        self,
        encoder_feature: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        if encoder_feature.ndim != 4:
            raise ValueError(f"encoder_feature must be [B, C, H/stride, W/stride], got {encoder_feature.shape}")

        coords_long = coords.long()
        batch = coords_long[:, 0]
        y = coords_long[:, 1]
        x = coords_long[:, 2]

        sampled = encoder_feature[batch, :, y, x]
        return sampled

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        encoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        point_cloud_h, point_cloud_w = shape[1], shape[2]
        encoder_h, encoder_w = encoder_feature.shape[2], encoder_feature.shape[3]
        assert point_cloud_h == encoder_h * self.encoder_downsample and point_cloud_w == encoder_w * self.encoder_downsample

        feats = self.input_proj(feats)
        shape = torch.Size([*shape[:4], feats.shape[-1]])

        # Cache layout: each kind of neighbor cache lives in its own per-index list.
        #
        #   level_conv_caches[k]   -- submanifold-conv neighborhood at level k.
        #                             Coords at level k are identical between
        #                             the down pass and the symmetric up pass
        #                             (the upsample restores coords to the
        #                             matching skip), so down_stages[k],
        #                             up_stages[num_levels-2-k] and (for
        #                             k == num_levels-1) bottleneck_stage all
        #                             share this cache. Length = num_levels.
        #   level_down_caches[k]   -- pool neighbor cache for the k -> k+1
        #                             downsample edge, reused as `.T` by the
        #                             symmetric upsample. Length = num_levels-1.
        #   skip_features[k]       -- (feats, coords, shape) at level k
        #                             captured immediately before the k -> k+1
        #                             downsample.
        num_levels = len(self.down_stages)
        num_transitions = len(self.downsample_blocks)
        level_conv_caches: List[Optional[NeighborCache]] = [None] * num_levels
        level_down_caches: List[Optional[NeighborCache]] = [None] * num_transitions
        skip_features: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Size]]] = [None] * num_transitions

        for i, stage in enumerate(self.down_stages):
            conv_cache = level_conv_caches[i]
            for block in stage:
                feats, conv_cache = block(feats, coords, shape, neighbor_cache=conv_cache)
            level_conv_caches[i] = conv_cache
            if i < num_transitions:
                skip_features[i] = (feats, coords, shape)
                feats, coords, shape, down_cache = self.downsample_blocks[i](feats, coords, shape)
                level_down_caches[i] = down_cache

        enc_feat = self._sample_encoder_feature(encoder_feature, coords)
        enc_feat = self.encoder_fuse(enc_feat)
        feats = self.fuse_proj(torch.cat([feats, enc_feat], dim=-1))

        # Bottleneck runs at the deepest level (num_levels-1) on the same
        # coords as down_stages[-1], so it reuses that level's conv cache.
        conv_cache = level_conv_caches[num_levels - 1]
        for block in self.bottleneck_stage:
            feats, conv_cache = block(feats, coords, shape, neighbor_cache=conv_cache)
        level_conv_caches[num_levels - 1] = conv_cache

        for i, (upsample_block, stage) in enumerate(zip(self.upsample_blocks, self.up_stages)):
            target_level = num_levels - 2 - i
            skip_feats, skip_coords, skip_shape = skip_features[target_level]
            feats, coords, shape = upsample_block(
                feats, coords, shape, skip_coords, skip_shape,
                up_cache=level_down_caches[target_level].T,
            )
            feats = feats + skip_feats
            # After the upsample, coords match the down side at target_level
            # -- reuse the conv cache built there.
            conv_cache = level_conv_caches[target_level]
            for block in stage:
                feats, conv_cache = block(feats, coords, shape, neighbor_cache=conv_cache)
            level_conv_caches[target_level] = conv_cache

        return self.out_proj(feats)
