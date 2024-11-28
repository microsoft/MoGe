from typing import *
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
import utils3d

from .tools import timeit
from .geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def harmonic_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps).add(eps).reciprocal()


def geometric_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).log().mean(dim=dim).exp()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).log(), w, dim=dim, keepdim=keepdim, eps=eps).exp()


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def gaussian_blur_2d(input: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = torch.exp(-(torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=input.dtype, device=input.device) ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).reshape(1, 1, kernel_size, kernel_size)
    input = F.pad(input, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='replicate')
    input = F.conv2d(input, kernel, groups=input.shape[1])
    return input


def focal_to_fov(focal: torch.Tensor):
    return 2 * torch.atan(0.5 / focal)


def fov_to_focal(fov: torch.Tensor):
    return 0.5 / torch.tan(fov / 2)


def intrinsics_to_fov(intrinsics: torch.Tensor):
    """
    Returns field of view in radians from normalized intrinsics matrix.
    ### Parameters:
    - intrinsics: torch.Tensor of shape (..., 3, 3)

    ### Returns:
    - fov_x: torch.Tensor of shape (...)
    - fov_y: torch.Tensor of shape (...)
    """
    focal_x = intrinsics[..., 0, 0]
    focal_y = intrinsics[..., 1, 1]
    return 2 * torch.atan(0.5 / focal_x), 2 * torch.atan(0.5 / focal_y)


def point_map_to_depth_legacy(points: torch.Tensor):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    # Solve least squares problem
    b = (uv * points[..., 2:]).flatten(-3, -1)                        # (..., H * W * 2)
    A = torch.stack([points[..., :2], -uv.expand_as(points[..., :2])], dim=-1).flatten(-4, -2)   # (..., H * W * 2, 2)

    M = A.transpose(-2, -1) @ A 
    solution = (torch.inverse(M + 1e-6 * torch.eye(2).to(A)) @ (A.transpose(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution.unbind(-1)

    depth = points[..., 2] + shift[..., None, None]
    fov_x = torch.atan(width / diagonal / focal) * 2
    fov_y = torch.atan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def view_plane_uv_to_focal(uv: torch.Tensor):
    normed_uv = normalized_view_plane_uv(width=uv.shape[-2], height=uv.shape[-3], device=uv.device, dtype=uv.dtype)
    focal = (uv * normed_uv).sum() / uv.square().sum().add(1e-12)
    return focal


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `mask: torch.Tensor` of shape (..., H, W). Optional.
    - `focal: torch.Tensor` of shape (...). Optional.
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def mask_aware_nearest_resize(mask: torch.BoolTensor, target_width: int, target_height: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    """
    Resize 2D map by nearest interpolation. Return the nearest neighbor index and mask of the resized map.

    ### Parameters
    - `mask`: Input 2D mask of shape (..., H, W)
    - `target_width`: target width of the resized map
    - `target_height`: target height of the resized map

    ### Returns
    - `nearest_idx`: Nearest neighbor index of the resized map of shape (..., target_height, target_width) for each dimension
    - `target_mask`: Mask of the resized map of shape (..., target_height, target_width)
    """
    height, width = mask.shape[-2:]
    device = mask.device
    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = round(filter_h_f / 2), round(filter_w_f / 2)

    # Window the original mask and uv
    uv = utils3d.torch.image_pixel_center(width=width, height=height, dtype=torch.float32, device=device)
    indices = torch.arange(height * width, dtype=torch.long, device=device).reshape(height, width)
    padded_uv = torch.full((height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=torch.float32, device=device)
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv
    padded_mask = torch.full((*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w), False, dtype=torch.bool, device=device)
    padded_mask[..., padding_h:padding_h + height, padding_w:padding_w + width] = mask
    padded_indices = torch.full((height + 2 * padding_h, width + 2 * padding_w), 0, dtype=torch.long, device=device)
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices
    windowed_uv = utils3d.torch.sliding_window_2d(padded_uv, (filter_h_i, filter_w_i), 1, dim=(0, 1))
    windowed_mask = utils3d.torch.sliding_window_2d(padded_mask, (filter_h_i, filter_w_i), 1, dim=(-2, -1))
    windowed_indices = utils3d.torch.sliding_window_2d(padded_indices, (filter_h_i, filter_w_i), 1, dim=(0, 1))

    # Gather the target pixels's local window
    target_uv = utils3d.torch.image_uv(width=target_width, height=target_height, dtype=torch.float32, device=device) * torch.tensor([width, height], dtype=torch.float32, device=device)
    target_corner = target_uv - torch.tensor((filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device)
    target_corner = torch.round(target_corner - 0.5).long() + torch.tensor((padding_w, padding_h), dtype=torch.long, device=device)

    target_window_uv = windowed_uv[target_corner[..., 1], target_corner[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                          # (target_height, tgt_width, 2, filter_size)
    target_window_mask = windowed_mask[..., target_corner[..., 1], target_corner[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = windowed_indices[target_corner[..., 1], target_corner[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)
    target_window_indices = target_window_indices.expand_as(target_window_mask)

    # Compute nearest neighbor in the local window for each pixel 
    dist = torch.where(target_window_mask, torch.norm(target_window_uv - target_uv[..., None], dim=-2), torch.inf)  # (..., target_height, tgt_width, filter_size)
    nearest = torch.argmin(dist, dim=-1, keepdim=True)                                                              # (..., target_height, tgt_width, 1)
    nearest_idx = torch.gather(target_window_indices, index=nearest, dim=-1).squeeze(-1)                            # (..., target_height, tgt_width)
    target_mask = torch.any(target_window_mask, dim=-1)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    batch_indices = [torch.arange(n, device=device).reshape([1] * i + [n] + [1] * (mask.dim() - i - 1)) for i, n in enumerate(mask.shape[:-2])]

    return (*batch_indices, nearest_i, nearest_j), target_mask
