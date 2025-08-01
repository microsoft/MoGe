from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.amp
import torch.version
import utils3d
from huggingface_hub import hf_hub_download

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, angle_diff_vec3
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from .modules import DINOv2Encoder, MLP, ConvStack

    
class MoGeModel(nn.Module):
    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP
    onnx_compatible_mode: bool

    def __init__(self, 
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: Literal['linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        num_tokens_range: List[int] = [1200, 3600],
        **deprecated_kwargs
    ):
        super(MoGeModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range
        
        self.encoder = DINOv2Encoder(**encoder) 
        self.neck = ConvStack(**neck)
        if points_head is not None:
            self.points_head = ConvStack(**points_head) 
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head)
        if scale_head is not None:
            self.scale_head = MLP(**scale_head)

        self._initialize_device_functions()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    @property
    def onnx_compatible_mode(self) -> bool:
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'MoGeModel':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `compiled`
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model'], strict=False)
        
        return model
    
    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        for head in ['points_head', 'normal_head', 'mask_head']:
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == 'linear':
            pass
        elif self.remap_output =='sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output =='sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points
    
    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = int((num_tokens / aspect_ratio) ** 0.5), int((num_tokens * aspect_ratio) ** 0.5)
        num_tokens = base_h * base_w

        # Convert image to numpy if on CPU
        if device.type != 'cuda':
            image_np = image.numpy()
        
        # Backbones encoding
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input - use device specific function
        for level in range(5):
            if device.type == 'cuda':
                uv = self.geometry['normalized_view_plane_uv'](
                    width=base_w * 2 ** level,
                    height=base_h * 2 ** level,
                    aspect_ratio=aspect_ratio,
                    dtype=dtype,
                    device=device
                )
            else:
                uv = torch.from_numpy(
                    self.geometry['normalized_view_plane_uv'](
                        width=base_w * 2 ** level,
                        height=base_h * 2 ** level,
                        aspect_ratio=aspect_ratio
                    )
                ).to(device)

            if features[level] is not None:
                features[level] = torch.cat([features[level], uv[None].expand(batch_size, -1, -1, -1)], dim=1)
        
        # Process features through heads
        features = self.neck(features)
        points = self.points_head(features)[0] if self.points_head is not None else None
        normal = self.normal_head(features)[0] if self.normal_head is not None else None
        mask = self.mask_head(features)[0] if self.mask_head is not None else None
        metric_scale = self.scale_head(cls_token)[0] if self.scale_head is not None else None

        # Resize to original resolution
        points, normal, mask = (F.interpolate(v, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False) if v is not None else None for v in [points, normal, mask])
        
        # Handle output based on device
        if points is not None:
            points = points.permute(0, 2, 3, 1)
            points = self._remap_points(points)
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            'points': points,
            'normal': normal,
            'mask': mask,
            'metric_scale': metric_scale
        }
        
        return return_dict

    @torch.inference_mode()
    def infer(
        self, 
        image: torch.Tensor, 
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: Literal[False, True, 'blend'] = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `num_tokens`: the number of base ViT tokens to use for inference, `'least'` or `'most'` or an integer. Suggested range: 1200 ~ 2500. 
            More tokens will result in significantly higher accuracy and finer details, but slower inference time. Default: `'most'`. 
        - `force_projection`: if True, the output point map will be computed using the actual depth map. Default: True
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: True
            
        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        if image.ndim == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        
        # Convert to appropriate device and dtype
        image = image.to(device=self.device, dtype=self.dtype)
        use_fp16 = use_fp16 and self.device.type == 'cuda'  # Only use fp16 on CUDA devices

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        
        # Forward pass with appropriate precision
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16):
            output = self.forward(image, num_tokens=num_tokens)
        
        points, normal, mask, metric_scale = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale'])

        # Process output in fp32 precision
        if points is not None:
            points = points.float()
        if normal is not None:
            normal = normal.float()
        if mask is not None:
            mask = mask.float()
            mask_binary = mask > 0.5
        else:
            mask_binary = None
        if metric_scale is not None:
            metric_scale = metric_scale.float()
        if isinstance(fov_x, torch.Tensor):
            fov_x = fov_x.float()

        # Process points and compute camera parameters
        if points is not None:
            # Convert to numpy for CPU operations if needed
            if self.device.type != 'cuda':
                points_np = points.cpu().numpy()
                mask_binary_np = mask_binary.cpu().numpy() if mask_binary is not None else None
            
            # Handle focal length and FOV
            if fov_x is None:
                if self.device.type == 'cuda':
                    focal = (1 + aspect_ratio ** 2) ** -0.5 / (points[..., 0].std(-1).std(-1) + 1e-5)
                else:
                    focal = (1 + aspect_ratio ** 2) ** -0.5 / (np.std(points_np[..., 0]) + 1e-5)
            else:
                focal = 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))

            # Convert scalar focal to tensor if needed
            if not isinstance(focal, torch.Tensor):
                focal = torch.tensor(focal, device=points.device, dtype=points.dtype)
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])

            # Build camera intrinsics
            fx = focal * aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
            fy = focal / (1 + aspect_ratio ** 2) ** 0.5
            intrinsics = torch.zeros((*points.shape[:-3], 3, 3), device=points.device, dtype=points.dtype)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = 0.5
            intrinsics[..., 2, 2] = 1

            # Process depth
            if force_projection:
                if self.device.type == 'cuda':
                    depth = self.geometry['points_to_depth'](points)
                    points = self.geometry['depth_to_points'](depth, intrinsics=intrinsics)
                else:
                    depth = torch.from_numpy(self.geometry['points_to_depth'](points_np)).to(self.device)
                    points = torch.from_numpy(
                        self.geometry['depth_to_points'](depth.cpu().numpy(), intrinsics=intrinsics.cpu().numpy())
                    ).to(self.device)
            else:
                depth = points[..., 2]

        # Assemble output dictionary
        return_dict = {}
        for k, v in [('points', points), ('depth', depth), ('normal', normal), 
                     ('mask', mask if apply_mask else None), ('intrinsics', intrinsics)]:
            if v is not None:
                if omit_batch_dim:
                    v = v.squeeze(0)
                return_dict[k] = v
        
        return return_dict

    def _initialize_device_functions(self):
        """Initialize device-specific geometry functions."""
        if self.device.type == 'cuda':
            from ..utils.geometry_torch import (
                normalized_view_plane_uv,
                depth_to_points,
                points_to_depth,
                points_to_normals,
                gaussian_blur_2d
            )
        else:
            from ..utils.geometry_cpu import (
                normalized_view_plane_uv_cpu as normalized_view_plane_uv,
                depth_to_points_cpu as depth_to_points,
                points_to_depth_cpu as points_to_depth,
                points_to_normals_cpu as points_to_normals,
                gaussian_blur_2d_cpu as gaussian_blur_2d
            )
        
        self.geometry = {
            'normalized_view_plane_uv': normalized_view_plane_uv,
            'depth_to_points': depth_to_points,
            'points_to_depth': points_to_depth,
            'points_to_normals': points_to_normals,
            'gaussian_blur_2d': gaussian_blur_2d
        }
