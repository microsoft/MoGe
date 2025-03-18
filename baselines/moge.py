import os
import sys
from typing import *
import importlib

import click
import torch
import utils3d

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):

    def __init__(self, num_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0', version: str = 'v1'):
        super().__init__()
        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(version)
        self.version = version

        self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        
        self.device = torch.device(device)
        self.num_tokens = num_tokens
        self.resolution_level = resolution_level
        self.use_fp16 = use_fp16
    
    @click.command()
    @click.option('--num_tokens', type=int, default=None)
    @click.option('--resolution_level', type=int, default=9)
    @click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl')
    @click.option('--fp16', 'use_fp16', is_flag=True)
    @click.option('--device', type=str, default='cuda:0')
    @click.option('--version', type=str, default='v1')
    @staticmethod
    def load(num_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0', version: str = 'v1'):
        return Baseline(num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version)

    # Implementation for inference
    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.torch.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self.model.infer(image, fov_x=fov_x, apply_mask=True, num_tokens=self.num_tokens)
        
        if self.version == 'v1':
            return {
                'points_scale_invariant': output['points'],
                'depth_scale_invariant': output['depth'],
                'intrinsics': output['intrinsics'],
            }
        else:
            return {
                'points_metric': output['points'],
                'depth_metric': output['depth'],
                'intrinsics': output['intrinsics'],
            }

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.torch.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self.model.infer(image, fov_x=fov_x, apply_mask=False, num_tokens=self.num_tokens, use_fp16=self.use_fp16)
        
        if self.version == 'v1':
            return {
                'points_scale_invariant': output['points'],
                'depth_scale_invariant': output['depth'],
                'intrinsics': output['intrinsics'],
            }
        else:
            return {
                'points_metric': output['points'],
                'depth_metric': output['depth'],
                'intrinsics': output['intrinsics'],
            }
        
