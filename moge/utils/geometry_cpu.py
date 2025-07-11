from typing import *
import math
import numpy as np

def normalized_view_plane_uv_cpu(width: int, height: int, aspect_ratio: float = None) -> np.ndarray:
    """CPU version of normalized view plane UV coordinates."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    x = np.linspace(-0.5, 0.5, width, dtype=np.float32)
    y = np.linspace(-0.5, 0.5, height, dtype=np.float32)
    u, v = np.meshgrid(x * aspect_ratio, y)
    return np.stack([u, v], axis=-1)

def depth_to_points_cpu(depth: np.ndarray, intrinsics: np.ndarray = None) -> np.ndarray:
    """Convert depth map to 3D points in camera space (CPU version)."""
    height, width = depth.shape[-2:]
    if intrinsics is None:
        # Default normalized intrinsics
        focal_x = focal_y = 1.0
        center_x = center_y = 0.5
    else:
        focal_x, focal_y = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
        center_x, center_y = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    
    y, x = np.meshgrid(
        np.linspace(0, height-1, height),
        np.linspace(0, width-1, width),
        indexing='ij'
    )
    
    x = (x - center_x * width) / focal_x
    y = (y - center_y * height) / focal_y
    
    points = np.stack([
        x * depth,
        y * depth,
        depth
    ], axis=-1)
    
    return points

def points_to_depth_cpu(points: np.ndarray) -> np.ndarray:
    """Extract depth from points map (CPU version)."""
    return points[..., 2]

def points_to_normals_cpu(points: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute surface normals from points map (CPU version)."""
    # Compute tangent vectors
    height, width = points.shape[-3:-1]
    dx = points[..., 1:, :, :] - points[..., :-1, :, :]
    dy = points[..., :, 1:, :] - points[..., :, :-1, :]
    
    # Compute normals from cross product
    normal = np.cross(dx[..., :-1, :, :], dy[..., 1:, :, :], axis=-1)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
    
    # Pad normals to match input size
    normal_pad = np.pad(
        normal,
        tuple((0, 0) for _ in range(normal.ndim - 3)) + ((0, 1), (0, 1), (0, 0)),
        mode='edge'
    )
    
    if mask is not None:
        mask_valid = mask[..., 1:, 1:] & mask[..., :-1, 1:] & mask[..., 1:, :-1] & mask[..., :-1, :-1]
        mask_pad = np.pad(
            mask_valid,
            tuple((0, 0) for _ in range(mask.ndim - 2)) + ((0, 1), (0, 1)),
            mode='constant'
        )
    else:
        mask_pad = np.ones_like(mask, dtype=bool)
        
    return normal_pad, mask_pad

def gaussian_blur_2d_cpu(x: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply 2D Gaussian blur (CPU version)."""
    kernel = np.exp(-(np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel_2d = kernel[:, None] * kernel[None, :]
    kernel_2d = kernel_2d[None, None, :, :]
    
    padding = kernel_size // 2
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='reflect')
    
    output = np.zeros_like(x)
    for i in range(x.shape[1]):
        output[:, i] = conv2d_cpu(x_pad[:, i:i+1], kernel_2d)
    return output

def conv2d_cpu(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution implementation for CPU."""
    from scipy.ndimage import convolve
    out_shape = x.shape[:-2] + (
        x.shape[-2] - kernel.shape[-2] + 1,
        x.shape[-1] - kernel.shape[-1] + 1
    )
    out = np.zeros(out_shape, dtype=x.dtype)
    
    for b in range(x.shape[0]):
        out[b] = convolve(x[b, 0], kernel[0, 0], mode='valid')
    return out
