from typing import *
from functools import partial
import math

import numpy as np
import utils3d

from .tools import timeit

def weighted_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int,...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        return (x * w).mean(axis=axis) / np.clip(w.mean(axis=axis), eps, None)


def harmonic_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int,...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return 1 / (1 / np.clip(x, eps, None)).mean(axis=axis)
    else:
        w = w.astype(x.dtype)
        return 1 / (weighted_mean_numpy(1 / (x + eps), w, axis=axis, keepdims=keepdims, eps=eps) + eps)


def normalized_view_plane_uv_numpy(width: int, height: int, aspect_ratio: float = None, dtype: np.dtype = np.float32) -> np.ndarray:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    uv = np.stack([u, v], axis=-1)
    return uv


def focal_to_fov_numpy(focal: np.ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal_numpy(fov: np.ndarray):
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov_numpy(intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fov_x = focal_to_fov_numpy(intrinsics[..., 0, 0])
    fov_y = focal_to_fov_numpy(intrinsics[..., 1, 1])
    return fov_x, fov_y


def point_map_to_depth_legacy_numpy(points: np.ndarray):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv_numpy(width, height, dtype=points.dtype)  # (H, W, 2)
    _, uv = np.broadcast_arrays(points[..., :2], uv)

    # Solve least squares problem
    b = (uv * points[..., 2:]).reshape(*points.shape[:-3], -1)                                  # (..., H * W * 2)
    A = np.stack([points[..., :2], -uv], axis=-1).reshape(*points.shape[:-3], -1, 2)   # (..., H * W * 2, 2)

    M = A.swapaxes(-2, -1) @ A 
    solution = (np.linalg.inv(M + 1e-6 * np.eye(2)) @ (A.swapaxes(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution

    depth = points[..., 2] + shift[..., None, None]
    fov_x = np.arctan(width / diagonal / focal) * 2
    fov_y = np.arctan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[: , None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy/ (z + shift)[: , None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def recover_focal_shift_numpy(points: np.ndarray, mask: np.ndarray = None, focal: float = None, downsample_size: Tuple[int, int] = (64, 64)):
    import cv2
    assert points.shape[-1] == 3, "Points should (H, W, 3)"

    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    uv = normalized_view_plane_uv_numpy(width=width, height=height)
    
    if mask is None:
        points_lr = cv2.resize(points, downsample_size, interpolation=cv2.INTER_LINEAR).reshape(-1, 3)
        uv_lr = cv2.resize(uv, downsample_size, interpolation=cv2.INTER_LINEAR).reshape(-1, 2)
    else:
        index, mask_lr = mask_aware_nearest_resize_numpy(mask, *downsample_size)
        points_lr, uv_lr = points[index][mask_lr], uv[index][mask_lr]
    
    if points_lr.size == 0:
        return np.zeros((height, width)), 0, 0, 0
    
    if focal is None:
        focal, shift = solve_optimal_focal_shift(uv_lr, points_lr)
    else:
        shift = solve_optimal_shift(uv_lr, points_lr, focal)

    return focal, shift


def mask_aware_nearest_resize_numpy(mask: np.ndarray, target_width: int, target_height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize 2D map by nearest interpolation. Return the nearest neighbor index and mask of the resized map.

    ### Parameters
    - `mask`: Input 2D mask of shape (..., H, W)
    - `target_width`: target width of the resized map
    - `target_height`: target height of the resized map

    ### Returns
    - `nearest_idx`: Nearest neighbor index of the resized map of shape (..., target_height, target_width). Indices are like j + i * W, where j is the row index and i is the column index.
    - `target_mask`: Mask of the resized map of shape (..., target_height, target_width)
    """
    height, width = mask.shape[-2:]
    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = round(filter_h_f / 2), round(filter_w_f / 2)
    
    # Window the original mask and uv
    uv = utils3d.numpy.image_pixel_center(width=width, height=height, dtype=np.float32)
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    padded_uv = np.full((height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=np.float32)
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv
    padded_mask = np.full((*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w), False, dtype=bool)
    padded_mask[..., padding_h:padding_h + height, padding_w:padding_w + width] = mask
    padded_indices = np.full((height + 2 * padding_h, width + 2 * padding_w), 0, dtype=np.int32)
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices
    windowed_uv = utils3d.numpy.sliding_window_2d(padded_uv, (filter_h_i, filter_w_i), 1, axis=(0, 1))
    windowed_mask = utils3d.numpy.sliding_window_2d(padded_mask, (filter_h_i, filter_w_i), 1, axis=(-2, -1))
    windowed_indices = utils3d.numpy.sliding_window_2d(padded_indices, (filter_h_i, filter_w_i), 1, axis=(0, 1))

    # Gather the target pixels's local window
    target_uv = utils3d.numpy.image_uv(width=target_width, height=target_height, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    target_corner = target_uv - np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_corner = np.round(target_corner - 0.5).astype(np.int32) + np.array((padding_w, padding_h), dtype=np.int32)

    target_window_uv = windowed_uv[target_corner[..., 1], target_corner[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                          # (target_height, tgt_width, 2, filter_size)
    target_window_mask = windowed_mask[..., target_corner[..., 1], target_corner[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = windowed_indices[target_corner[..., 1], target_corner[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)

    # Compute nearest neighbor in the local window for each pixel 
    dist = np.square(target_window_uv - target_uv[..., None])
    dist = dist[..., 0, :] + dist[..., 1, :]
    dist = np.where(target_window_mask, dist, np.inf)                                                   # (..., target_height, tgt_width, filter_size)
    nearest_in_window = np.argmin(dist, axis=-1, keepdims=True)                                         # (..., target_height, tgt_width, 1)
    nearest_idx = np.take_along_axis(target_window_indices, nearest_in_window, axis=-1).squeeze(-1)     # (..., target_height, tgt_width)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = np.any(target_window_mask, axis=-1)
    batch_indices = [np.arange(n).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(mask.shape[:-2])]

    return (*batch_indices, nearest_i, nearest_j), target_mask