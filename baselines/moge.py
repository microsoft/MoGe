import os
import sys
from typing import *
import importlib

import click
import torch
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):

    def __init__(self, num_tokens: int, resolution_level: int, refine_steps: int, pretrained_model_name_or_path: Optional[str], use_fp16: bool, device: str = 'cuda:0', version: str = 'v1'):
        super().__init__()
        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(version)
        self.version = version

        if pretrained_model_name_or_path is None:
            default_pretrained_models = {
                'v1': 'Ruicheng/moge-vitl',
                'v2': 'Ruicheng/moge-2-vitl-normal',
            }
            if version == 'v3':
                raise click.UsageError('--pretrained is required when --version is v3.')
            pretrained_model_name_or_path = default_pretrained_models[version]
        self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        if version == 'v3' and refine_steps > 0 and not hasattr(self.model, 'refiner'):
            raise click.UsageError('The loaded v3 checkpoint has no refiner; set --refine_steps 0.')
        
        self.device = torch.device(device)
        self.num_tokens = num_tokens
        self.resolution_level = resolution_level
        self.refine_steps = refine_steps
        self.use_fp16 = use_fp16
    
    @click.command()
    @click.option('--num_tokens', type=int, default=None)
    @click.option('--resolution_level', type=int, default=9)
    @click.option('--refine_steps', type=click.IntRange(min=0), default=3, help='Number of sparse refinement steps for v3. Defaults to 3.')
    @click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default=None, help='Pretrained model name or path. Optional for v1/v2 and required for v3.')
    @click.option('--fp16', 'use_fp16', is_flag=True)
    @click.option('--device', type=str, default='cuda:0')
    @click.option('--version', type=click.Choice(['v1', 'v2', 'v3']), default='v1')
    @staticmethod
    def load(num_tokens: int, resolution_level: int, refine_steps: int, pretrained_model_name_or_path: Optional[str], use_fp16: bool, device: str = 'cuda:0', version: str = 'v1'):
        return Baseline(num_tokens, resolution_level, refine_steps, pretrained_model_name_or_path, use_fp16, device, version)

    def _infer(self, image: torch.FloatTensor, fov_x: Optional[torch.Tensor], apply_mask: bool) -> Dict[str, torch.Tensor]:
        infer_kwargs = {
            'fov_x': fov_x,
            'apply_mask': apply_mask,
            'num_tokens': self.num_tokens,
            'resolution_level': self.resolution_level,
            'use_fp16': self.use_fp16,
        }
        if self.version == 'v3':
            infer_kwargs['refine_steps'] = self.refine_steps
        return self.model.infer(image, **infer_kwargs)

    # Implementation for inference
    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.pt.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self._infer(image, fov_x, apply_mask=True)
        
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
            fov_x, _ = utils3d.pt.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self._infer(image, fov_x, apply_mask=False)
        
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
        
