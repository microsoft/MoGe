from typing import *
from numbers import Number
import warnings

import torch
import torch.nn.functional as F
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift
from .v2 import MoGeModel as MoGeModelV2
from .modules.sparse_unet import Sparse3DUNet


class MoGeModel(MoGeModelV2):
    def __init__(
        self,
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        num_tokens_range: List[int] = [1200, 3600],
        refiner: Optional[Dict[str, Any]] = None,
        refiner_depth_resolution: float = 256,
        **deprecated_kwargs,
    ):
        super().__init__(
            encoder=encoder,
            neck=neck,
            points_head=points_head,
            mask_head=mask_head,
            normal_head=normal_head,
            scale_head=scale_head,
            remap_output='exp',
            num_tokens_range=num_tokens_range,
            **deprecated_kwargs,
        )

        if refiner is not None:
            refiner_cfg = dict(refiner)
            self.refiner_depth_resolution = refiner_depth_resolution
            self.refiner = Sparse3DUNet(**refiner_cfg)
        else:
            warnings.warn("Warning: refiner is not enabled.")

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'refiner'):
            self.refiner.init_weights()

    def enable_gradient_checkpointing(self):
        super().enable_gradient_checkpointing()
        if hasattr(self, 'refiner'):
            self.refiner.enable_gradient_checkpointing()

    def _voxelize(
        self,
        point_coord: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.Tensor]:
        """
        Convert dense point coordinates to a sparse representation.

        - point_coord: [B, H, W, 3] at (x/z, y/z, logz).

        Returns (feats, coords, shape, logz):
        - feats:  (M, 3) fp32 input features [uv, logz].
        - coords: (M, 4) int32, columns (batch, i, j, z_bin).
        - shape:  Size([B, H, W, z_extent, in_channels]).
        - logz:   [B, H, W] dense fp32 log-depth (for the residual update).

        Everything here is forced to fp32: the z binning is quantized at
        1/refiner_depth_resolution, which is finer than fp16 resolution over the
        usual logz range, so binning in fp16 would collapse/jitter voxels.
        """
        if point_coord.ndim != 4 or point_coord.shape[-1] != 3:
            raise ValueError(f"point_coord must be [B, H, W, 3], got {point_coord.shape}")

        point_coord = point_coord.float()
        bsz, height, width, _ = point_coord.shape
        device = point_coord.device

        logz = point_coord[..., 2]
        zq = torch.round(logz * self.refiner_depth_resolution).long()
        z_offset = zq.amin(dim=(1, 2), keepdim=True)
        z_idx = zq - z_offset
        z_extent = z_idx.amax().item() + 1

        i = torch.arange(height, device=device, dtype=torch.long).view(1, height, 1).expand(bsz, height, width)
        j = torch.arange(width, device=device, dtype=torch.long).view(1, 1, width).expand(bsz, height, width)
        batch = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1, 1).expand(bsz, height, width)
        coords = torch.stack([batch, i, j, z_idx], dim=-1).reshape(-1, 4).to(torch.int32)

        feats = point_coord.reshape(-1, 3)
        shape = torch.Size([bsz, height, width, z_extent, feats.shape[-1]])
        return feats, coords, shape, logz

    def _refine_logz(
        self,
        point_coord: torch.Tensor,
        encoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        bsz, height, width, _ = point_coord.shape
        feats, coords, shape, logz = self._voxelize(point_coord)
        out: torch.Tensor = self.refiner(feats, coords, shape, encoder_feature)
        out_logz = out.float().squeeze(-1).reshape(bsz, height, width)
        refined_logz = logz + out_logz
        return refined_logz

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor],
        refine_steps: int = 3,
        refiner_detach_backbone: bool = True,
        return_per_step: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if refine_steps > 0 and not hasattr(self, 'refiner'):
            raise ValueError("Refiner is not enabled but refine_steps > 0.")

        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # Backbones encoding
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        neck_features = self.neck(features)

        # Heads decoding
        raw_coord = self.points_head(neck_features)[-1] if hasattr(self, 'points_head') else None
        normal, mask = (
            getattr(self, head)(neck_features)[-1] if hasattr(self, head) else None
            for head in ['normal_head', 'mask_head']
        )
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None

        # Refine point map in factorized coordinate space
        coord_per_step: List[torch.Tensor] = []
        coords: Optional[torch.Tensor] = None
        points: Optional[torch.Tensor] = None
        points_per_step: Optional[List[torch.Tensor]] = None
        if raw_coord is not None: # raw_coord is B3HW at (x/z, y/z, logz)
            current_coord = raw_coord.permute(0, 2, 3, 1).float() # BHW3 at (x/z, y/z, logz)
            if return_per_step:
                coord_per_step.append(current_coord)

            if refine_steps > 0:
                refiner_feature: torch.Tensor = features[0]

                for _ in range(refine_steps):
                    feature_for_refiner = refiner_feature.detach() if refiner_detach_backbone else refiner_feature
                    refined_logz = self._refine_logz(current_coord.detach(), feature_for_refiner)
                    current_coord = torch.cat([current_coord[..., :2], refined_logz.unsqueeze(-1)], dim=-1)
                    if return_per_step:
                        coord_per_step.append(current_coord)

            coords = torch.stack(coord_per_step, dim=1) if return_per_step else current_coord.unsqueeze(1)

        # Resize and remap outputs
        resize = lambda x, channel_last=False: F.interpolate(
            x.movedim(-1, -3) if channel_last else x,
            (img_h, img_w),
            mode='bilinear',
            align_corners=False,
            antialias=False,
        ).movedim(-3, -1 if channel_last else -3)

        if coords is not None:
            num_point_steps = coords.shape[1]
            coords = resize(coords.flatten(0, 1), channel_last=True)
            coords = coords.unflatten(0, (batch_size, num_point_steps))
            points_all = self._remap_points(coords)
            points = points_all[:, -1]
            if return_per_step:
                points_per_step = list(points_all.unbind(dim=1))

        if normal is not None:
            normal = resize(normal)
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = resize(mask)
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            'points': points,
            'points_per_step': points_per_step,
            'normal': normal,
            'mask': mask,
            'metric_scale': metric_scale,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        refine_steps: int = 3,
        return_per_step: bool = False,
        use_fp16: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W).
        - `num_tokens`: the number of base ViT tokens to use for inference. If None, it is determined by `resolution_level`.
        - `resolution_level`: inference resolution level from 0 to 9. Higher values use more tokens and preserve finer details. Default: 9.
        - `force_projection`: if True, recompute each point map from its depth map and intrinsics. Default: True.
        - `apply_mask`: if True, mask invalid points and depths using the predicted mask. Default: True.
        - `fov_x`: horizontal camera field of view in degrees. If None, it is inferred from each returned point map. Default: None.
        - `refine_steps`: number of sparse 3D refinement updates. Default: 3.
        - `return_per_step`: if True, return predictions for the initial estimate and every refinement step. Default: False.
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: False.

        ### Returns
        A dictionary containing the following keys when the corresponding outputs are available:
        - `points`: final camera-space point map of shape (B, H, W, 3) or (H, W, 3).
        - `points_per_step`: point maps for the initial prediction and every refinement step, when `return_per_step=True`.
        - `intrinsics`: camera intrinsics associated with the final point map, of shape (B, 3, 3) or (3, 3).
        - `intrinsics_per_step`: camera intrinsics for the initial prediction and every refinement step, when `return_per_step=True`.
        - `depth`: final depth map of shape (B, H, W) or (H, W).
        - `depth_per_step`: depth maps for the initial prediction and every refinement step, when `return_per_step=True`.
        - `mask`: predicted valid-pixel mask of shape (B, H, W) or (H, W).
        - `normal`: predicted normal map of shape (B, H, W, 3) or (H, W, 3).
        """
        if refine_steps > 0 and not hasattr(self, 'refiner'):
            raise ValueError("Refiner is not enabled but refine_steps > 0.")

        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(image, num_tokens=num_tokens, refine_steps=refine_steps, return_per_step=return_per_step)
        affine_points, normal, mask, metric_scale = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale'])
        affine_points_per_step = output.get('points_per_step', None)

        # Always process the output in fp32 precision
        if affine_points_per_step is None:
            affine_points_per_step = [affine_points] if affine_points is not None else None
        affine_points_per_step = [p.float() for p in affine_points_per_step] if affine_points_per_step is not None else None
        normal, mask, metric_scale, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [normal, mask, metric_scale, fov_x])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None

            if affine_points_per_step is not None:
                # Per-step (focal, shift) recovery: refinement modifies logz which changes
                # the 3D shape, so focal recovered from each step's point map differs slightly
                # Jointly solving (focal, shift) on the same point map gives the optimal
                # affine->camera alignment for that step (matches v2_4's single-step logic).
                if fov_x is not None:
                    focal_fixed = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=self.device, dtype=torch.float32) / 2))
                    if focal_fixed.ndim == 0:
                        focal_fixed = focal_fixed[None].expand(affine_points_per_step[-1].shape[0])
                else:
                    focal_fixed = None

                points_per_step, depth_per_step, intrinsics_per_step = [], [], []
                for affine_points in affine_points_per_step:
                    if focal_fixed is None:
                        focal_i, shift_i = recover_focal_shift(affine_points, mask_binary)
                    else:
                        focal_i = focal_fixed
                        _, shift_i = recover_focal_shift(affine_points, mask_binary, focal=focal_i)
                    fx_i, fy_i = focal_i / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal_i / 2 * (1 + aspect_ratio ** 2) ** 0.5
                    intrinsics_i = utils3d.pt.intrinsics_from_focal_center(fx_i, fy_i, 0.5, 0.5)

                    points = affine_points.clone()
                    points[..., 2] += shift_i[..., None, None]
                    depth = points[..., 2].clone()

                    if force_projection:
                        points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics_i)

                    if metric_scale is not None:
                        points *= metric_scale[:, None, None, None]
                        depth *= metric_scale[:, None, None]

                    points_per_step.append(points)
                    depth_per_step.append(depth)
                    intrinsics_per_step.append(intrinsics_i)

                # Final intrinsics correspond to the final-step point map (the one used as
                # the canonical output `points` / `depth`).
                intrinsics = intrinsics_per_step[-1]

                # Build per-step masks so each step is self-consistently masked
                # against its own depth>0 (mirrors v2_4's `mask_binary &= points[..., 2] > 0`).
                if mask_binary is not None:
                    mask_per_step = [mask_binary & (d > 0) for d in depth_per_step]
                    mask_binary = mask_per_step[-1]
                else:
                    mask_per_step = None

                points = points_per_step[-1]
                depth = depth_per_step[-1]
            else:
                points_per_step = None
                depth_per_step = None
                intrinsics_per_step = None
                mask_per_step = None
                points, depth, intrinsics = None, None, None

            if apply_mask:
                if mask_per_step is not None:
                    points_per_step = [torch.where(m[..., None], p, torch.inf) for p, m in zip(points_per_step, mask_per_step)]
                    depth_per_step  = [torch.where(m, d, torch.inf) for d, m in zip(depth_per_step, mask_per_step)]
                    points, depth = points_per_step[-1], depth_per_step[-1]
                if mask_binary is not None and normal is not None:
                    normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

            if not return_per_step:
                points_per_step = None
                depth_per_step = None
                intrinsics_per_step = None

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'normal': normal,
            'points_per_step': points_per_step,
            'intrinsics_per_step': intrinsics_per_step,
            'depth_per_step': depth_per_step,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {
                k: [item.squeeze(0) for item in v] if isinstance(v, list) else v.squeeze(0)
                for k, v in return_dict.items()
            }

        return return_dict
