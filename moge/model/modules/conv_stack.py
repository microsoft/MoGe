from typing import *
import itertools
import contextlib
import functools

import torch
import torch.nn as nn

from ..utils import wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing


class ResidualConvBlock(nn.Module):  
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = None, 
        hidden_channels: int = None, 
        kernel_size: int = 3, 
        padding_mode: str = 'replicate', 
        activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', 
        in_norm: Literal['group_norm', 'layer_norm', 'instance_norm', 'none'] = 'layer_norm',
        hidden_norm: Literal['group_norm', 'layer_norm', 'instance_norm'] = 'group_norm',
    ):  
        super(ResidualConvBlock, self).__init__()  
        if out_channels is None:  
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation =='relu':
            activation_cls = nn.ReLU
        elif activation == 'leaky_relu':
            activation_cls = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        elif activation =='silu':
            activation_cls = nn.SiLU
        elif activation == 'elu':
            activation_cls = nn.ELU
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(in_channels // 32, in_channels) if in_norm == 'group_norm' else \
                nn.GroupNorm(1, in_channels) if in_norm == 'layer_norm' else \
                nn.InstanceNorm2d(in_channels) if in_norm == 'instance_norm' else \
                nn.Identity(),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32, hidden_channels) if hidden_norm == 'group_norm' else \
                nn.GroupNorm(1, hidden_channels) if hidden_norm == 'layer_norm' else \
                nn.InstanceNorm2d(hidden_channels) if hidden_norm == 'instance_norm' else\
                nn.Identity(),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode)
        )
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()  
  
    def forward(self, x):  
        skip = self.skip_connection(x)  
        x = self.layers(x)
        x = x + skip
        return x  

class Resampler(nn.Sequential):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        type_: Literal['pixel_shuffle', 'nearest', 'bilinear', 'conv_transpose', 'pixel_unshuffle', 'avg_pool', 'max_pool'],
        scale_factor: int = 2, 
    ):
        if type_ == 'pixel_shuffle':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
            for i in range(1, scale_factor ** 2):
                self[0].weight.data[i::scale_factor ** 2] = self[0].weight.data[0::scale_factor ** 2]
                self[0].bias.data[i::scale_factor ** 2] = self[0].bias.data[0::scale_factor ** 2]
        elif type_ in ['nearest', 'bilinear']:
            nn.Sequential.__init__(self,
                nn.Upsample(scale_factor=scale_factor, mode=type_, align_corners=False if type_ == 'bilinear' else None),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
        elif type_ == 'conv_transpose':
            nn.Sequential.__init__(self,
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
            self[0].weight.data[:] = self[0].weight.data[:, :, :1, :1]
        elif type_ == 'pixel_unshuffle':
            nn.Sequential.__init__(self,
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(in_channels * (scale_factor ** 2), out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
        elif type_ == 'avg_pool': 
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        elif type_ == 'max_pool':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        else:
            raise ValueError(f'Unsupported resampler type: {type_}')


class ConvStack(nn.Module):
    def __init__(self, 
        dim_in: List[Optional[int]],
        dim_res_blocks: List[int],
        dim_out: List[Optional[int]],
        resamplers: Union[Literal['pixel_shuffle', 'nearest', 'bilinear', 'conv_transpose', 'pixel_unshuffle', 'avg_pool', 'max_pool'], List],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_in_norm: Literal['layer_norm', 'group_norm' , 'instance_norm', 'none'] = 'layer_norm',
        res_block_hidden_norm: Literal['layer_norm', 'group_norm' , 'instance_norm', 'none'] = 'group_norm',
        activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu',
        fp32_output_projection: bool = False
    ):
        super().__init__()
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(dim_in_, dim_res_block_, kernel_size=1, stride=1, padding=0) if dim_in_ is not None else nn.Identity() 
                for dim_in_, dim_res_block_ in zip(dim_in if isinstance(dim_in, Sequence) else itertools.repeat(dim_in), dim_res_blocks)
        ])
        self.resamplers = nn.ModuleList([
            Resampler(dim_prev, dim_succ, scale_factor=2, type_=resampler) 
                for i, (dim_prev, dim_succ, resampler) in enumerate(zip(
                    dim_res_blocks[:-1], 
                    dim_res_blocks[1:], 
                    resamplers if isinstance(resamplers, Sequence) else itertools.repeat(resamplers)
                ))
        ])
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                *(
                    ResidualConvBlock(
                        dim_res_block_, dim_res_block_, dim_times_res_block_hidden * dim_res_block_, 
                        activation=activation, in_norm=res_block_in_norm, hidden_norm=res_block_hidden_norm
                    ) for _ in range(num_res_blocks[i] if isinstance(num_res_blocks, list) else num_res_blocks)
                )
            ) for i, dim_res_block_ in enumerate(dim_res_blocks)
        ])
        self.output_blocks = nn.ModuleList([
            nn.Conv2d(dim_res_block_, dim_out_, kernel_size=1, stride=1, padding=0) if dim_out_ is not None else nn.Identity() 
                for dim_out_, dim_res_block_ in zip(dim_out if isinstance(dim_out, Sequence) else itertools.repeat(dim_out), dim_res_blocks)
        ])
        self.fp32_output_projection = fp32_output_projection

    def enable_gradient_checkpointing(self):
        for i in range(len(self.resamplers)):
            self.resamplers[i] = wrap_module_with_gradient_checkpointing(self.resamplers[i])
        for i in range(len(self.res_blocks)):
            for j in range(len(self.res_blocks[i])):
                self.res_blocks[i][j] = wrap_module_with_gradient_checkpointing(self.res_blocks[i][j])

    def forward(self, in_features: List[torch.Tensor]):
        out_features = []
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = self.res_blocks[i](x)
            # Optionally force the 1x1 output projection to run in fp32
            with (
                torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=True)
                if self.fp32_output_projection
                else contextlib.nullcontext()
            ):
                out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features