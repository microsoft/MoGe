# Reference https://github.com/DepthAnything/Depth-Anything-V2/metric_depth
import os
import sys
from typing import *
from pathlib import Path

import click
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):

    def __init__(self, repo_path: str, backbone: str, domain: str, num_tokens: int, device: str):
        device = torch.device(device)
        repo_path = os.path.abspath(repo_path)
        if not Path(repo_path).exists():
            raise FileNotFoundError(f'Cannot find the Depth-Anything repository at {repo_path}. Please clone the repository and provide the path to it using the --repo option.')
        sys.path.append(os.path.join(repo_path, 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        if domain == 'indoor':
            dataset = 'hypersim'
            max_depth = 20
        elif domain == 'outdoor':
            dataset = 'vkitti'
            max_depth = 80
        else:
            raise ValueError(f"Invalid domain: {domain}")

        model = DepthAnythingV2(**model_configs[backbone], max_depth=max_depth)
        checkpoint_path = os.path.join(repo_path, f'checkpoints/depth_anything_v2_metric_{dataset}_{backbone}.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Cannot find the checkpoint file at {checkpoint_path}. Please download the checkpoint file and place it in the checkpoints directory.')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        model.eval().to(device)
                
        self.model = model
        self.num_tokens = num_tokens
        self.device = device

    @click.command()
    @click.option('--repo', 'repo_path', type=click.Path(), default='../Depth-Anything-V2', help='Path to the Depth-Anything repository.')
    @click.option('--backbone', type=click.Choice(['vits', 'vitb', 'vitl']), default='vitl', help='Backbone architecture.')
    @click.option('--domain', type=click.Choice(['indoor', 'outdoor']), help='Domain of the dataset.')
    @click.option('--num_tokens', type=int, default=None, help='Number of tokens for the ViT model')
    @click.option('--device', type=str, default='cuda', help='Device to use for inference.')
    @staticmethod
    def load(repo_path: str, backbone: str, domain: str, num_tokens: int, device: str):  
        return Baseline(repo_path, backbone, domain, num_tokens, device)

    @torch.inference_mode()
    def infer(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        original_height, original_width = image.shape[-2:]

        assert intrinsics is None, "Depth-Anything-V2 does not support camera intrinsics input"

        if image.ndim == 3:
            image = image.unsqueeze(0)
            omit_batch_dim = True
        else:
            omit_batch_dim = False

        if self.num_tokens is None:
            resize_factor = 518 / min(original_height, original_width)
            expected_width = round(original_width * resize_factor / 14) * 14
            expected_height = round(original_height * resize_factor / 14) * 14
        else:
            aspect_ratio = original_width / original_height
            tokens_rows = round((self.num_tokens * aspect_ratio) ** 0.5)
            tokens_cols = round((self.num_tokens / aspect_ratio) ** 0.5)
            expected_width = tokens_cols * 14
            expected_height = tokens_rows * 14
        image = TF.resize(image, (expected_height, expected_width), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        depth = self.model(image)

        depth = F.interpolate(depth[:, None], size=(original_height, original_width), mode='bilinear', align_corners=False, antialias=False)[:, 0]

        if omit_batch_dim:
            depth = depth.squeeze(0)

        return {
            'depth_metric': depth
        }

