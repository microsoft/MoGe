from typing import *
import importlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2.models.vision_transformer import DinoVisionTransformer
from ..utils import wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing


class DINOv2Encoder(nn.Module):
    "Wrapped DINOv2 encoder supporting gradient checkpointing. Input is RGB image in range [0, 1]."
    backbone: DinoVisionTransformer
    image_mean: torch.Tensor
    image_std: torch.Tensor
    dim_features: int

    def __init__(self, backbone: str, intermediate_layers: Union[int, List[int]], dim_out: int, **deprecated_kwargs):
        super(DINOv2Encoder, self).__init__()

        self.intermediate_layers = intermediate_layers

        # Load the backbone
        self.hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), backbone)
        self.backbone_name = backbone
        self.backbone = self.hub_loader(pretrained=False)

        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers)

        self.output_projections = nn.ModuleList([
            nn.Conv2d(in_channels=self.dim_features, out_channels=dim_out, kernel_size=1, stride=1, padding=0,) 
                for _ in range(self.num_features)
        ])

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.backbone.onnx_compatible_mode = value

    def init_weights(self):
        pretrained_backbone_state_dict = self.hub_loader(pretrained=True).state_dict()
        self.backbone.load_state_dict(pretrained_backbone_state_dict)

    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def forward(self, image: torch.Tensor, token_rows: Union[int, torch.LongTensor], token_cols: Union[int, torch.LongTensor], return_class_token: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        image_14 = F.interpolate(image, (token_rows * 14, token_cols * 14), mode="bilinear", align_corners=False, antialias=not self.onnx_compatible_mode)
        image_14 = (image_14 - self.image_mean) / self.image_std

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers(image_14, n=self.intermediate_layers, return_class_token=True)
    
        # Project features to the desired dimensionality
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
                for proj, (feat, clstoken) in zip(self.output_projections, features)
        ], dim=1).sum(dim=1)                    

        if return_class_token:
            return x, features[-1][1]
        else:
            return x