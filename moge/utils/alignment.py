from typing import *
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
import utils3d


def scatter_min(size: int, dim: int, index: torch.LongTensor, src: torch.Tensor) -> torch.return_types.min:
    "Scatter the minimum value along the given dimension of `input` into `src` at the indices specified in `index`."
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device).scatter_reduce(dim=dim, index=index, src=src, reduce='amin', include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1:])] = minimum_where[dim]
    return torch.return_types.min((minimum, indices))
    

def split_batch_fwd(fn: Callable, chunk_size: int, *args, **kwargs):
    batch_size = next(x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)).shape[0]
    n_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
    splited_args = tuple(arg.split(chunk_size, dim=0) if isinstance(arg, torch.Tensor) else [arg] * n_chunks for arg in args)
    splited_kwargs = {k: [v.split(chunk_size, dim=0) if isinstance(v, torch.Tensor) else [v] * n_chunks] for k, v in kwargs.items()}
    results = []
    for i in range(n_chunks):
        chunk_args = tuple(arg[i] for arg in splited_args)
        chunk_kwargs = {k: v[i] for k, v in splited_kwargs.items()}
        results.append(fn(*chunk_args, **chunk_kwargs))

    if isinstance(results[0], tuple):
        return tuple(torch.cat(r, dim=0) for r in zip(*results))
    else:
        return torch.cat(results, dim=0)


def _pad_inf(x_: torch.Tensor):
    return torch.cat([torch.full_like(x_[..., :1], -torch.inf), x_, torch.full_like(x_[..., :1], torch.inf)], dim=-1)


def _pad_cumsum(cumsum: torch.Tensor):
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)


def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    return a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)


def align(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, trunc: Optional[Union[float, torch.Tensor]] = None, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    If trunc is None, solve `min sum_i w_i * |a * x_i - y_i|`, otherwise solve `min sum_i min(trunc, w_i * |a * x_i - y_i|)`.
    
    w_i must be >= 0.

    ### Parameters:
    - `x`: tensor of shape (..., n)
    - `y`: tensor of shape (..., n)
    - `w`: tensor of shape (..., n)
    - `trunc`: optional, float or tensor of shape (..., n) or None

    ### Returns:
    - `a`: tensor of shape (...), differentiable
    - `loss`: tensor of shape (...), value of loss function at `a`, detached
    - `index`: tensor of shape (...), where a = y[idx] / x[idx]
    """
    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(derivatives, torch.zeros_like(derivatives[..., :1]), side='left').clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
        
    else:
        # Reshape to (batch_size, n) for simplicity
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)    # Stacked for convenient gathering

        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
            # Caculate prefix sum by orders of A, B, C    
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)    # Pad [-inf, A1, ..., An, inf] and [0, Q1, ..., Qn, Qn] to handle edge cases.

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)
            
            # Caculate left and right derivative of A
            j_A = torch.searchsorted(A, y_div_x, side='left').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='left').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='left').sub_(1)
            left_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)
            j_A = torch.searchsorted(A, y_div_x, side='right').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='right').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='right').sub_(1)
            right_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)

            # Find extrema
            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)                       # In case all derivatives are zero, take the first one as extrema.
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)          

            # Calculate objective value at extrema
            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]               # (num_extrema,)
            MAX_ELEMENTS = 4096 ** 2      # Split into small batches to avoid OOM in case there are too many extrema.(~1G)
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat([
                _compute_residual(extrema_a_split[:, None], xyw[extrema_i_split, :, :], trunc)
                for extrema_a_split, extrema_i_split in zip(extrema_a.split(SPLIT_SIZE), where_extrema_batch.split(SPLIT_SIZE))
            ])          # (num_extrema,)
            
            # Find minima among corresponding extrema
            minima, indices = scatter_min(size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value)        # (batch_size,)
            index = where_extrema_index[indices]

        a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(x, dim=-1, index=index[..., None]).clamp_min(eps)
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index


def align_depth_scale(depth_src: torch.Tensor, depth_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None):
    """
    Align `depth_src` to `depth_tgt` with given constant weights. 

    ### Parameters:
    - `depth_src: torch.Tensor` of shape (..., N)
    - `depth_tgt: torch.Tensor` of shape (..., N)

    """
    scale, _, _ = align(depth_src, depth_tgt, weight, trunc)

    return scale


def align_depth_affine(depth_src: torch.Tensor, depth_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None):
    """
    Align `depth_src` to `depth_tgt` with given constant weights.

    ### Parameters:
    - `depth_src: torch.Tensor` of shape (..., N)
    - `depth_tgt: torch.Tensor` of shape (..., N)
    - `weight: torch.Tensor` of shape (..., N)
    - `trunc: float` or tensor of shape (..., N) or None

    ### Returns:
    - `scale: torch.Tensor` of shape (...).
    - `shift: torch.Tensor` of shape (...).
    """
    dtype, device = depth_src.dtype, depth_src.device
 
    # Flatten batch dimensions for simplicity
    batch_shape, n = depth_src.shape[:-1], depth_src.shape[-1]
    batch_size = math.prod(batch_shape)
    depth_src, depth_tgt, weight = depth_src.reshape(batch_size, n), depth_tgt.reshape(batch_size, n), weight.reshape(batch_size, n)

    # Here, we take anchors only for non-zero weights.
    # Although the results will be still correct even anchor points have zero weight,
    # it is wasting computation and may cause instability in some cases, e.g. too many extrema.
    anchors_where_batch, anchors_where_n = torch.where(weight > 0)

    # Stop gradient when solving optimal anchors
    with torch.no_grad():
        depth_src_anchor = depth_src[anchors_where_batch, anchors_where_n]                              # (anchors)
        depth_tgt_anchor = depth_tgt[anchors_where_batch, anchors_where_n]                              # (anchors)

        depth_src_anchored = depth_src[anchors_where_batch, :] - depth_src_anchor[..., None]            # (anchors, n)
        depth_tgt_anchored = depth_tgt[anchors_where_batch, :] - depth_tgt_anchor[..., None]            # (anchors, n)
        weight_anchored = weight[anchors_where_batch, :]                                                # (anchors, n)

        scale, loss, index = align(depth_src_anchored, depth_tgt_anchored, weight_anchored, trunc)      # (anchors)

        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchors_where_batch, src=loss)   # (batch_size,)

    # Reproduce by indexing for shorter compute graph
    index_1 = anchors_where_n[index_anchor]      # (batch_size,)
    index_2 = index[index_anchor]                # (batch_size,)

    tgt_1, src_1 = torch.gather(depth_tgt, dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(depth_src, dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_2, src_2 = torch.gather(depth_tgt, dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(depth_src, dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1e-7)
    shift = tgt_1 - scale * src_1

    scale, shift = scale.reshape(batch_shape), shift.reshape(batch_shape)

    return scale, shift

def align_depth_affine_irls(depth_src: torch.Tensor, depth_tgt: torch.Tensor, weight: Optional[torch.Tensor], max_iter: int = 100, eps: float = 1e-12):
    """
    Align `depth_src` to `depth_tgt` with given constant weights using IRLS.
    """
    dtype, device = depth_src.dtype, depth_src.device
    
    w = weight
    x = torch.stack([depth_src, torch.ones_like(depth_src)], dim=-1)
    y = depth_tgt

    for i in range(max_iter):
        beta = (x.transpose(-1, -2) @ (w * y)) @ (x.transpose(-1, -2) @ (w[..., None] * x)).inverse().transpose(-2, -1)
        w = 1 / (y - (x @ beta[..., None])[..., 0]).abs().clamp_min(eps)

    return beta[..., 0], beta[..., 1]


def align_points_scale(points_src: torch.Tensor, points_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None):
    """
    ### Parameters:
    - `points_src: torch.Tensor` of shape (..., N, 3)
    - `points_tgt: torch.Tensor` of shape (..., N, 3)
    - `weight: torch.Tensor` of shape (..., N)

    ### Returns:
    - `a: torch.Tensor` of shape (...). Only positive solutions are garunteed. You should filter out negative scales before using it.
    - `b: torch.Tensor` of shape (...)
    """
    dtype, device = points_src.dtype, points_src.device
    
    scale, _, _ = align(points_src.flatten(-2), points_tgt.flatten(-2), weight[..., None].expand_as(points_src).flatten(-2), trunc)

    return scale


def align_points_scale_z_shift(points_src: torch.Tensor, points_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None):
    """
    Align `points_src` to `points_tgt` with respect to a shared xyz scale and z shift. 
    It is similar to `align_affine` but scale and shift are applied to different dimensions.

    ### Parameters:
    - `points_src: torch.Tensor` of shape (..., N, 3)
    - `points_tgt: torch.Tensor` of shape (..., N, 3)
    - `weights: torch.Tensor` of shape (..., N)

    ### Returns:
    - `scale: torch.Tensor` of shape (...).
    - `shift: torch.Tensor` of shape (..., 3). x and y shifts are zeros.
    """
    dtype, device = points_src.dtype, points_src.device

    # Flatten batch dimensions for simplicity
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = points_src.reshape(batch_size, n, 3), points_tgt.reshape(batch_size, n, 3), weight.reshape(batch_size, n)

    # Take anchors
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)
    with torch.no_grad():
        zeros = torch.zeros(anchor_where_batch.shape[0], device=device, dtype=dtype)
        points_src_anchor = torch.stack([zeros, zeros, points_src[anchor_where_batch, anchor_where_n, 2]], dim=-1)      # (anchors, 3)
        points_tgt_anchor = torch.stack([zeros, zeros, points_tgt[anchor_where_batch, anchor_where_n, 2]], dim=-1)      # (anchors, 3)

        points_src_anchored = points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]    # (anchors, n, 3)
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]    # (anchors, n, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)                         # (anchors, n, 3)

        # Solve optimal scale and shift for each anchor
        MAX_ELEMENTS = 2 ** 20
        scale, loss, index = split_batch_fwd(align, MAX_ELEMENTS // n, points_src_anchored.flatten(-2), points_tgt_anchored.flatten(-2), weight_anchored.flatten(-2), trunc)   # (anchors,)

        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)    # (batch_size,)

    # Reproduce by indexing for shorter compute graph
    index_2 = index[index_anchor]                               # (batch_size,) [0, 3n)
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3    # (batch_size,) [0, 3n)

    zeros = torch.zeros((batch_size, n), device=device, dtype=dtype)
    points_tgt_00z, points_src_00z = torch.stack([zeros, zeros, points_tgt[..., 2]], dim=-1), torch.stack([zeros, zeros, points_src[..., 2]], dim=-1)
    tgt_1, src_1 = torch.gather(points_tgt_00z.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(points_src_00z.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_2, src_2 = torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(points_tgt_00z, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2) - scale[..., None] * torch.gather(points_src_00z, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2)
    scale, shift = scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)

    return scale, shift


def align_points_scale_xyz_shift(points_src: torch.Tensor, points_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None, max_iters: int = 30, eps: float = 1e-6):
    """
    Align `points_src` to `points_tgt` with respect to a shared xyz scale and z shift. 
    It is similar to `align_affine` but scale and shift are applied to different dimensions.

    ### Parameters:
    - `points_src: torch.Tensor` of shape (..., N, 3)
    - `points_tgt: torch.Tensor` of shape (..., N, 3)
    - `weights: torch.Tensor` of shape (..., N)

    ### Returns:
    - `scale: torch.Tensor` of shape (...).
    - `shift: torch.Tensor` of shape (..., 3)
    """
    dtype, device = points_src.dtype, points_src.device

    # Flatten batch dimensions for simplicity
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = points_src.reshape(batch_size, n, 3), points_tgt.reshape(batch_size, n, 3), weight.reshape(batch_size, n)

    # Take anchors
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)

    with torch.no_grad():
        points_src_anchor = points_src[anchor_where_batch, anchor_where_n]          # (anchors, 3)
        points_tgt_anchor = points_tgt[anchor_where_batch, anchor_where_n]          # (anchors, 3)

        points_src_anchored = points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]    # (anchors, n, 3)
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]    # (anchors, n, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)                         # (anchors, n, 3)

        # Solve optimal scale and shift for each anchor
        MAX_ELEMENTS = 2 ** 20
        scale, loss, index = split_batch_fwd(align, MAX_ELEMENTS // 2, points_src_anchored.flatten(-2), points_tgt_anchored.flatten(-2), weight_anchored.flatten(-2), trunc)   # (anchors,)

        # Get optimal scale and shift for each batch element
        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)    # (batch_size,)

    index_2 = index[index_anchor]                               # (batch_size,) [0, 3n)
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3    # (batch_size,) [0, 3n)

    src_1, tgt_1 = torch.gather(points_src.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(points_tgt.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    src_2, tgt_2 = torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(points_tgt, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2) - scale[..., None] * torch.gather(points_src, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2)

    scale, shift = scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)

    return scale, shift


def align_points_z_shift(points_src: torch.Tensor, points_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None, max_iters: int = 30, eps: float = 1e-6):
    """
    Align `points_src` to `points_tgt` with respect to a Z-axis shift. 

    ### Parameters:
    - `points_src: torch.Tensor` of shape (..., N, 3)
    - `points_tgt: torch.Tensor` of shape (..., N, 3)
    - `weights: torch.Tensor` of shape (..., N)

    ### Returns:
    - `scale: torch.Tensor` of shape (...).
    - `shift: torch.Tensor` of shape (..., 3)
    """
    dtype, device = points_src.dtype, points_src.device

    shift, _, _ = align(torch.ones_like(points_src[..., 2]), points_tgt[..., 2] - points_src[..., 2], weight, trunc)
    shift = torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)

    return shift


def align_points_xyz_shift(points_src: torch.Tensor, points_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None, max_iters: int = 30, eps: float = 1e-6):
    """
    Align `points_src` to `points_tgt` with respect to a Z-axis shift. 

    ### Parameters:
    - `points_src: torch.Tensor` of shape (..., N, 3)
    - `points_tgt: torch.Tensor` of shape (..., N, 3)
    - `weights: torch.Tensor` of shape (..., N)

    ### Returns:
    - `scale: torch.Tensor` of shape (...).
    - `shift: torch.Tensor` of shape (..., 3)
    """
    dtype, device = points_src.dtype, points_src.device

    shift, _, _ = align(torch.ones_like(points_src).swapaxes(-2, -1), (points_tgt - points_src).swapaxes(-2, -1), weight[..., None, :], trunc)

    return shift


def align_affine_lstsq(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve `min sum_i w_i * (a * x_i + b - y_i ) ^ 2`, where `a` and `b` are scalars, with respect to `a` and `b` using least squares.

    ### Parameters:
    - `x: torch.Tensor` of shape (..., N)
    - `y: torch.Tensor` of shape (..., N)
    - `w: torch.Tensor` of shape (..., N)

    ### Returns:
    - `a: torch.Tensor` of shape (...,)
    - `b: torch.Tensor` of shape (...,)
    """
    w_sqrt = torch.ones_like(x) if w is None else w.sqrt()
    A = torch.stack([w_sqrt * x, torch.ones_like(x)], dim=-1)
    B = (w_sqrt * y)[..., None]
    a, b = torch.linalg.lstsq(A, B)[0].squeeze(-1).unbind(-1)
    return a, b