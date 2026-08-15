from typing import *
import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from ..utils.geometry_torch import (
    weighted_mean, 
    harmonic_mean, 
    geometric_mean,
    mask_aware_nearest_resize,
)
from ..utils.alignment import (
    align_points_scale_z_shift,
    align_points_scale,
    align_points_scale_xyz_shift,
    align_points_z_shift,
    align_points_xyz_shift,
    segment_align_shift,
)


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def silog_mse_loss(pred_points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor, lambda_: float = 1, eps: float = 1e-6, trim_ratio: float = 0.0):
    batch_size = pred_points.shape[0]
    device, dtype = pred_points.device, pred_points.dtype

    pred_uv = pred_points[..., :2] / (pred_points[..., 2:3] + eps)
    gt_uv = gt_points[..., :2] / (gt_points[..., 2:3] + eps)

    pred_z = pred_points[..., 2]
    gt_z = gt_points[..., 2]

    uv_mse = F.mse_loss(pred_uv, gt_uv, reduction='none').mean(dim=-1)
    diff_log_z = torch.log(pred_z.clamp_min(eps)) - torch.log(gt_z.clamp_min(eps))
    si_log = diff_log_z.square() - lambda_ * weighted_mean(diff_log_z, mask, dim=(-2, -1), keepdim=True).square()

    loss = (si_log + uv_mse) * mask
    
    loss = loss.mean()

    return loss, {}


def scale_invariant_global_loss(points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor, align: Literal['harmonic', 'optimal'], beta: float = 0.0, trim_ratio: float = 0.0):
    device, dtype = points.device, points.dtype

    # Normalize gt_points
    gt_points = gt_points / harmonic_mean(gt_points[..., 2], mask, dim=(-2, 1)).add(1e-3)[..., None, None, None]

    # Align points
    if align == 'harmonic':
        scale = harmonic_mean(gt_points[..., 2], mask, dim=(1, 2)).add(1e-3) / harmonic_mean(points[..., 2], mask, dim=(1, 2)).add(1e-3)
        points = scale[:, None, None, None] * points
        valid = torch.tensor(True, device=device, dtype=torch.bool)
    elif align == 'optimal':
        scale = align_points_scale(points.flatten(-3, -2), gt_points.flatten(-3, -2), mask.flatten(-2, -1), trunc=1.0)
        valid = scale > 0
        scale = torch.where(valid, scale, 0)
        points = scale[..., None, None, None] * points
    
    weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(0.2)

    loss = _smooth((points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=-1)

    # Trim loss
    if trim_ratio > 0:
        trim_thres = torch.nanquantile(torch.where(mask, loss, torch.nan).flatten(), 1.0 - trim_ratio, interpolation='lower') 
        mask = mask & (loss < trim_thres)
        loss = torch.where(mask, loss, 0)

    loss = loss.mean()

    return loss, scale.detach()


def affine_invariant_global_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    align_resolution: int = 64, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False,
    improve_weighting: bool = False,
    fixed_scale: Optional[torch.Tensor] = None,
):
    device = pred_points.device

    # Align. Normally we solve a fresh global (scale, z-shift) that best fits this point map to the
    # GT, which makes the loss invariant to the point map's absolute scale. When `fixed_scale` is
    # given (e.g. the step-0 alignment reused across refine steps) we PIN the scale and only solve
    # the z-shift at that scale, making the loss scale-sensitive: a global size change of
    # `pred_points` (e.g. the refiner emitting a net-positive delta-z that inflates the whole scene)
    # is then penalized instead of being silently absorbed by a per-step re-alignment.
    pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(pred_points, gt_points, mask=mask, size=(align_resolution, align_resolution))
    align_weight = lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2)
    if fixed_scale is None:
        scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), align_weight, trunc=trunc)
        valid = scale > 0
        scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)
    else:
        scale = fixed_scale
        valid = scale > 0
        shift = align_points_z_shift((scale[..., None, None, None] * pred_points_lr).flatten(-3, -2), gt_points_lr.flatten(-3, -2), align_weight, trunc=trunc)
        shift = torch.where(valid[..., None], shift, 0)

    pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

    # Compute loss
    if improve_weighting:
        weight = (valid[..., None, None] & mask).float() / torch.maximum(gt_points[..., 2], pred_points[..., 2].detach())
    else:
        weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
        weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))
    
    diff = (pred_points - gt_points).abs()
    loss = _smooth(diff * weight[..., None], beta=beta).mean(dim=-1).mean(dim=(-2, -1))

    # Sparsity aware reweighting
    if sparsity_aware:
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)

    err = (pred_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2]
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).detach(),
        'delta': weighted_mean((err < 1).float(), mask).detach(),
        'delta_0p1': weighted_mean((err < 0.1).float(), mask).detach(),
        'delta_0p01': weighted_mean((err < 0.01).float(), mask).detach(),
    }

    return loss, misc, scale, shift.detach()


def monitoring(points: torch.Tensor):
    return {
        'std': points.std().detach(),
    }

def monitor_delta(points_before: torch.Tensor, points_after: torch.Tensor):
    delta_z = points_after[..., 2].detach() - points_before[..., 2].detach()
    return {
        "mean": delta_z.mean().detach(),
        "std": delta_z.std().detach(),
    }

def compute_anchor_sampling_weight(points: torch.Tensor, mask: torch.Tensor, radius_2d: torch.Tensor, radius_3d: torch.Tensor, num_test: int = 64) -> torch.Tensor:
    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device), 
        torch.arange(width, device=points.device),
        indexing='ij'
    )
    
    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
    return weight


def affine_invariant_local_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    focal: torch.Tensor, 
    global_scale: torch.Tensor, 
    level: Literal[4, 16, 64], 
    align_resolution: int = 32, 
    num_patches: int = 16, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False, 
    improve_weighting: bool = False
):
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)

    pred_points, gt_points, mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None

    # Sample patch anchor points indices [num_total_patches]
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
    radius_3d = 0.5 / level / focal[:, None, None] * gt_points[..., 2]
    anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, mask, radius_2d, radius_3d, num_test=64)
    where_mask = torch.where(mask)
    random_selection = torch.multinomial(anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True)
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]     # [num_total_patches]

    # Get patch indices [num_total_patches, patch_h, patch_w]
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device), 
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij'
    )
    patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)
    
    # Get patch mask and gt patch points
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)    
    patch_mask &= mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    # Pick only non-empty patches
    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # Finalize all patch variables
    patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
    patch_mask = patch_mask[nonempty]                                   # [num_nonempty_patches, patch_h, patch_w]
    gt_patch_points = gt_patch_points[nonempty]                         # [num_nonempty_patches, patch_h, patch_w, 3]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]                   # [num_nonempty_patches]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]           # [num_nonempty_patches, 3]
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    
    # Align patch points
    pred_patch_points_lr, gt_patch_points_lr, patch_lr_mask = utils3d.pt.masked_nearest_resize(pred_patch_points, gt_patch_points, mask=patch_mask, size=(align_resolution, align_resolution))
    local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
    if global_scale is not None:
        global_scale_per_patch = global_scale.index_select(0, patch_batch_idx)
        scale_differ = local_scale / global_scale_per_patch
        patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale_per_patch > 0)
    else:
        patch_valid = local_scale > 0
    local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
    patch_mask &= patch_valid[:, None, None]

    pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]                                    # [num_patches_nonempty, patch_h, patch_w, 3]
    
    # Compute loss
    if improve_weighting:
        patch_weight = patch_mask.float() / torch.maximum(gt_patch_points[..., 2], pred_patch_points[..., 2].detach())
    else:
        gt_mean = harmonic_mean(gt_points[..., 2], mask, dim=(-2, -1))
        patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])          # [num_patches_nonempty, patch_h, patch_w]
    loss = _smooth((pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=(-3, -2, -1))                     # [num_patches_nonempty]

    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    loss = torch.scatter_reduce(torch.zeros(batch_size, dtype=dtype, device=device), dim=0, index=patch_batch_idx, src=loss, reduce='sum') / num_patches
    loss = loss.reshape(batch_shape)
    
    err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None]

    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).detach(),
        'delta': weighted_mean((err < 1).float(), patch_mask).detach(),
    }

    return loss, misc

def normal_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
    gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
    gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = mask_upxleft * _smooth(utils3d.pt.angle_between(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_leftxdown * _smooth(utils3d.pt.angle_between(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_downxright * _smooth(utils3d.pt.angle_between(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_rightxup * _smooth(utils3d.pt.angle_between(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

    loss = loss.mean() / (4 * max(points.shape[-3:-1]))

    return loss, {}


def edge_loss(pred_points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device, dtype = pred_points.device, pred_points.dtype
    height, width = pred_points.shape[-3:-1]

    pred_dx = torch.diff(pred_points, dim=-3)
    pred_dy = torch.diff(pred_points, dim=-2)
    
    gt_dx = torch.diff(gt_points, dim=-3)
    gt_dy = torch.diff(gt_points, dim=-2)

    mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
    mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(0.1), math.radians(90), math.radians(3)

    loss_dx = mask_dx * _smooth(utils3d.pt.angle_between(pred_dx, gt_dx).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss_dy = mask_dy * _smooth(utils3d.pt.angle_between(pred_dy, gt_dy).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss = (loss_dx.mean(dim=(-2, -1)) + loss_dy.mean(dim=(-2, -1))) / min(height, width)

    return loss, {}


def normal_map_loss(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(gt_normal).all(dim=-1)
    gt_normal = torch.where(mask[..., None], gt_normal, 1)

    loss = (mask * utils3d.pt.angle_between(pred_normal, gt_normal).square()).mean(dim=(-2, -1))
    return loss, {}


def mask_l2_loss(pred_mask: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    # NOTE: Only the labelled pixels contribute. Pixels that are in neither mask (e.g. regions
    # with no depth annotation) are ignored rather than being supervised towards "valid".
    loss = gt_mask_neg.float() * pred_mask.square() + gt_mask_pos.float() * (1 - pred_mask).square()
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


def metric_scale_loss(scale_pred: torch.Tensor, scale_gt: torch.Tensor):
    valid = scale_gt > 0
    return torch.where(valid, F.mse_loss(scale_pred.log(), torch.where(valid, scale_gt.log(), 0), reduction='none'), 0), {}


def metric_scale_l1_loss(scale_pred: torch.Tensor, scale_gt: torch.Tensor):
    valid = scale_gt > 0
    return torch.where(valid, F.l1_loss(scale_pred.log(), torch.where(valid, scale_gt.log(), 0), reduction='none'), 0), {}


def shift_invariant_global_loss(points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor, align_resolution: int = 64, beta: float = 0.0, trunc: float = 1.0, sparsity_aware: bool = True):
    device = points.device

    (points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize((points, gt_points), mask=mask, size=(align_resolution, align_resolution))
    shift = align_points_z_shift(points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)

    points = points + shift[..., None, None, :]

    weight = mask.float() / gt_points[..., 2].clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))

    loss = _smooth((points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=-1)

    if sparsity_aware:
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)

    loss = loss.mean()

    return loss, {}


def decoupled_shift_invariant_global_loss(pred_metric_scale: torch.Tensor, pred_points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor, align_resolution: int = 64, beta: float = 0.0, trunc: float = 1.0, sparsity_aware: bool = True, detach: bool = False):
    device = pred_points.device
    
    if detach:
        pred_points = pred_points.detach()
    pred_points = pred_points * pred_metric_scale[..., None]

    (pred_points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize((pred_points, gt_points), mask, (align_resolution, align_resolution))
    shift = align_points_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)

    pred_points = pred_points + shift[..., None, None, :]
    
    weight = mask.float() / gt_points[..., 2].clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))

    loss = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=-1)

    if sparsity_aware:
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)

    loss = loss.mean()

    return loss, {}


def l1_global_loss(pred_points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor, beta: float = 0.0):
    gt_mean = harmonic_mean(gt_points[..., 2], mask, dim=(-2, -1)).add(1e-3)

    weight = mask.float() / gt_points[..., 2].clamp_min(0.1 * gt_mean[..., None, None])
    loss = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))

    return loss, {}


def normal_map_tv_loss(pred_normal: torch.Tensor):
    dx = utils3d.pt.angle_between(pred_normal[..., :, :-1, :], pred_normal[..., :, 1:, :])
    dy = utils3d.pt.angle_between(pred_normal[..., :-1, :, :], pred_normal[..., 1:, :, :])
    loss = (dx.mean(axis=(-2, -1)) + dy.mean(axis=(-2, -1))) / 2
    return loss, {}


def radial_partition(pts: torch.Tensor, alpha: float = 4.0, eps: float = 1e-7) -> torch.Tensor:
    """
    径向分区函数：将3D点按照方向和对数距离进行量化分区

    将点云按照以下方式分区：
    1. 方向：将点归一化到单位立方体表面（使用无穷范数），得到方向标签
    2. 距离：使用欧氏距离的对数，得到距离标签

    这种分区方式使得空间上相近的点被分到同一组，适合局部对齐。

    Parameters
    ----------
    pts : torch.Tensor
        输入3D点，形状 (N, 3)
    alpha : float
        分区粒度参数，越大分区越细（默认4.0）
    eps : float
        数值稳定性参数（默认1e-7）

    Returns
    -------
    partition : torch.Tensor
        分区标签，形状 (N, 4)，包含 [dir_x, dir_y, dir_z, log_dist]

    Examples
    --------
    >>> pts = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    >>> labels = radial_partition(pts, alpha=4.0)
    >>> labels.shape
    torch.Size([2, 4])
    """
    # 方向：归一化到单位立方体表面（使用无穷范数）
    # 无穷范数 = max(|x|, |y|, |z|)，归一化后至少有一个坐标为 ±1
    direction = pts / (pts.norm(dim=-1, p=float('inf'), keepdim=True) + eps)

    # 对数距离：使用欧氏距离的对数，使远近分布更均匀
    log_dist = torch.log2(pts.norm(dim=-1, p=2, keepdim=True) + eps) / 4

    # 拼接方向和距离，然后量化为整数标签
    partition = (alpha * torch.cat([direction, log_dist], dim=-1)).round().short()

    return partition


@dataclass
class RadialPartitionLocalLossCache:
    batch_shape: Tuple[int, ...]
    height: int
    width: int
    device: torch.device
    dtype: torch.dtype
    flat_indices: torch.Tensor
    gt_points_flat: torch.Tensor
    pixel_weights_flat: torch.Tensor
    batch_flat: torch.Tensor
    gt_dist_flat: torch.Tensor
    geometry_weights_flat: torch.Tensor
    weights_flat: torch.Tensor
    group_cache: Dict[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = field(default_factory=dict)
    pred_cache: Dict[int, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_inputs(cls, gt_points: torch.Tensor, mask: torch.Tensor) -> "RadialPartitionLocalLossCache":
        *batch_shape, height, width, _ = gt_points.shape
        batch_size = math.prod(batch_shape)

        gt_points = gt_points.reshape(batch_size, height, width, 3)
        mask = mask.reshape(batch_size, height, width)
        gt_mask = mask & torch.isfinite(gt_points).all(dim=-1)
        flat_indices = gt_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)

        pixels_per_batch = height * width
        batch_flat = flat_indices.div(pixels_per_batch, rounding_mode='floor')
        gt_points_safe = gt_points.masked_fill(~gt_mask[..., None], 1.0)
        gt_points_flat = gt_points_safe.reshape(batch_size * pixels_per_batch, 3).index_select(0, flat_indices)

        valid_counts = gt_mask.reshape(batch_size, pixels_per_batch).sum(dim=1).clamp_min(1).to(dtype=torch.float32)
        pixel_weights_flat = valid_counts.reciprocal().index_select(0, batch_flat)

        gt_dist_flat = gt_points_flat.norm(dim=-1)
        geometry_weights_flat = 1.0 / gt_dist_flat.clamp_min(1e-5)
        if geometry_weights_flat.numel() > 0:
            geometry_weights_flat = geometry_weights_flat.clamp_max(10.0 * geometry_weights_flat.mean())
        weights_flat = pixel_weights_flat * geometry_weights_flat

        return cls(
            batch_shape=tuple(batch_shape),
            height=height,
            width=width,
            device=gt_points.device,
            dtype=gt_points.dtype,
            flat_indices=flat_indices,
            gt_points_flat=gt_points_flat,
            pixel_weights_flat=pixel_weights_flat,
            batch_flat=batch_flat,
            gt_dist_flat=gt_dist_flat,
            geometry_weights_flat=geometry_weights_flat,
            weights_flat=weights_flat,
        )

    @property
    def batch_size(self) -> int:
        return math.prod(self.batch_shape)

    def pred_points_flat(self, pred_points: torch.Tensor) -> torch.Tensor:
        cache_key = id(pred_points)
        cached = self.pred_cache.get(cache_key)
        if cached is None:
            pred_points = pred_points.reshape(self.batch_size, self.height, self.width, 3)
            cached = pred_points.reshape(self.batch_size * self.height * self.width, 3).index_select(0, self.flat_indices)
            self.pred_cache[cache_key] = cached
        return cached

    def grouped_indices(self, alpha: float, pred_points_flat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        alpha_key = float(alpha)
        if pred_points_flat is not None:
            # 如果提供了 pred_points_flat，则使用它来计算分区标签（包含随机旋转），而不是 gt_points_flat
            rand_scale = 2 ** torch.rand((), device=self.device, dtype=torch.float32)
            rand_rotation = utils3d.pt.quaternion_to_matrix(torch.randn(4, device=self.device, dtype=torch.float32))
            partition_labels_flat = radial_partition(
                rand_scale * pred_points_flat.float() @ rand_rotation.mT,
                alpha=alpha_key,
            )
            partition_labels_flat = torch.cat([
                partition_labels_flat,
                self.batch_flat[:, None].short(),
            ], dim=1)
            partition_groups, iflat_grouped, offsets_grouped, igroup_flat = utils3d.pt.group_as_segments(partition_labels_flat, return_group_ids=True)
            cached = (iflat_grouped, offsets_grouped, igroup_flat, len(partition_groups))
            return cached
        cached = self.group_cache.get(alpha_key)
        if cached is None:
            rand_scale = 2 ** torch.rand((), device=self.device, dtype=torch.float32)
            rand_rotation = utils3d.pt.quaternion_to_matrix(torch.randn(4, device=self.device, dtype=torch.float32))
            partition_labels_flat = radial_partition(
                rand_scale * self.gt_points_flat.float() @ rand_rotation.mT,
                alpha=alpha_key,
            )
            partition_labels_flat = torch.cat([
                partition_labels_flat,
                self.batch_flat[:, None].short(),
            ], dim=1)
            partition_groups, iflat_grouped, offsets_grouped, igroup_flat = utils3d.pt.group_as_segments(partition_labels_flat, return_group_ids=True)
            cached = (iflat_grouped, offsets_grouped, igroup_flat, len(partition_groups))
            self.group_cache[alpha_key] = cached
        return cached


def radial_partition_local_loss_rand_partition(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    global_scale: torch.Tensor,
    alpha: float = 4.0,
    beta: float = 0.0,
    trunc: float | None = None,
    pred_partition_ratio: float = 0.5,
    packed: RadialPartitionLocalLossCache | None = None,
):
    """
    基于径向分区的局部几何损失函数（单目相机版本）

    该函数借鉴多视图几何的局部对齐思想，应用于单目深度估计：
    1. 将3D空间按照径向分区（方向+距离）划分为多个局部区域
    2. 在每个局部区域内独立计算最优平移对齐
    3. 对齐后计算几何误差损失

    相比于基于重要性采样的方法，该方法：
    - 覆盖所有有效点（而非采样）
    - 只优化平移参数（直接使用 global_scale）
    - 自动处理单目深度的尺度模糊性

    Parameters
    ----------
    pred_points : torch.Tensor
        预测的3D点，形状 (B, H, W, 3)
    gt_points : torch.Tensor
        真实的3D点，形状 (B, H, W, 3)
    mask : torch.Tensor
        有效点掩码，形状 (B, H, W)
    global_scale : torch.Tensor
        全局尺度因子，形状 (B,)
    alpha : float
        分区大小控制参数，越大分区越细（默认4.0）
    beta : float
        Smooth L1 loss 的参数（默认0.0，即标准L1）
    trunc : float | None
        截断阈值（用于鲁棒对齐，默认None）
    pred_partition_ratio : float
        使用 pred 点生成 partition 的概率；0.0 表示全用 gt，1.0 表示全用 pred（默认0.5）

    Returns
    -------
    loss : torch.Tensor
        损失值，标量
    misc : dict
        额外的监控指标：
        - 'truncated_error': 截断后的加权平均误差
        - 'delta': 相对误差 < 1.0 的点的比例
        - 'num_groups': 分区数量
        - 'points_per_group': 每个分区的平均点数

    Notes
    -----
    - 每次迭代使用随机旋转进行数据增强，避免固定分区边界伪影
    - 近处的点获得更高的权重（1/距离）
    - 不同batch的点不会被分到同一组

    Examples
    --------
    >>> pred = torch.randn(2, 64, 64, 3)
    >>> gt = torch.randn(2, 64, 64, 3)
    >>> mask = torch.ones(2, 64, 64, dtype=torch.bool)
    >>> global_scale = torch.ones(2)
    >>> loss, misc = radial_partition_local_loss(pred, gt, mask, global_scale)
    """
    device, dtype = pred_points.device, pred_points.dtype
    if not 0.0 <= pred_partition_ratio <= 1.0:
        raise ValueError(f"pred_partition_ratio must be in [0, 1], got {pred_partition_ratio}")
    if packed is None:
        packed = RadialPartitionLocalLossCache.from_inputs(gt_points, mask)

    batch_size = packed.batch_size
    batch_shape = packed.batch_shape
    global_scale = global_scale.reshape(batch_size)

    pred_points_flat = packed.pred_points_flat(pred_points)
    gt_points_flat = packed.gt_points_flat
    pixel_weights_flat = packed.pixel_weights_flat
    batch_flat = packed.batch_flat
    gt_dist_flat = packed.gt_dist_flat
    geometry_weights_flat = packed.geometry_weights_flat
    weights_flat = packed.weights_flat

    if pixel_weights_flat.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), {}

    # Align shifts grouped by partition (vectorized, avoids per-group loops)
    # NOTE: Avoid fancy indexing `global_scale[batch_flat, None]` — its backward
    # triggers `indexing_backward_kernel_stride_1` which scatter-adds N gradients
    # back to batch_size elements via slow atomicAdd. Use index_select instead,
    # whose backward is a simple gather (much faster).
    global_scale_flat = global_scale.index_select(0, batch_flat).unsqueeze(-1)
    pred_points_flat = global_scale_flat * pred_points_flat
    if pred_partition_ratio > 0.0 and random.random() < pred_partition_ratio:
        iflat_grouped, offsets_grouped, igroup_flat, num_groups = packed.grouped_indices(alpha, pred_points_flat.detach())
    else:
        iflat_grouped, offsets_grouped, igroup_flat, num_groups = packed.grouped_indices(alpha)
    shifts, _ = segment_align_shift(
        pred_points_flat.index_select(0, iflat_grouped).swapaxes(0, 1),
        gt_points_flat.index_select(0, iflat_grouped).swapaxes(0, 1),
        weights_flat.index_select(0, iflat_grouped),
        offsets=offsets_grouped,
    )  # (3, num_groups)
    shifts_flat = shifts.swapaxes(0, 1).index_select(0, igroup_flat)
    pred_points_flat = pred_points_flat + shifts_flat

    # 计算几何误差
    error_flat = (pred_points_flat - gt_points_flat).abs().mean(dim=-1) * geometry_weights_flat
    error_flat = _smooth(error_flat, beta=beta)

    # 加权损失
    loss_flat = pixel_weights_flat * error_flat

    # 按batch聚合损失
    loss_per_batch = torch.zeros(batch_size, dtype=dtype, device=device)
    loss_per_batch.scatter_add_(0, batch_flat, loss_flat)
    loss = loss_per_batch.reshape(batch_shape).mean()

    # 计算监控指标
    err = (pred_points_flat.detach() - gt_points_flat).norm(dim=-1) / gt_dist_flat
    points_per_group = len(iflat_grouped) / num_groups if num_groups > 0 else 0

    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), pixel_weights_flat).detach(),
        'delta': weighted_mean((err < 1.0).float(), pixel_weights_flat).detach(),
        'delta_0p1': weighted_mean((err < 0.1).float(), pixel_weights_flat).detach(),
        'delta_0p01': weighted_mean((err < 0.01).float(), pixel_weights_flat).detach(),
        'num_groups': num_groups,
        'points_per_group': points_per_group,
    }

    return loss, misc


def radial_partition_local_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    global_scale: torch.Tensor,
    alpha: float = 4.0,
    beta: float = 0.0,
    trunc: float | None = None,
    packed: RadialPartitionLocalLossCache | None = None,
):
    """
    基于径向分区的局部几何损失函数（单目相机版本）

    该函数借鉴多视图几何的局部对齐思想，应用于单目深度估计：
    1. 将3D空间按照径向分区（方向+距离）划分为多个局部区域
    2. 在每个局部区域内独立计算最优平移对齐
    3. 对齐后计算几何误差损失

    相比于基于重要性采样的方法，该方法：
    - 覆盖所有有效点（而非采样）
    - 只优化平移参数（直接使用 global_scale）
    - 自动处理单目深度的尺度模糊性

    Parameters
    ----------
    pred_points : torch.Tensor
        预测的3D点，形状 (B, H, W, 3)
    gt_points : torch.Tensor
        真实的3D点，形状 (B, H, W, 3)
    mask : torch.Tensor
        有效点掩码，形状 (B, H, W)
    global_scale : torch.Tensor
        全局尺度因子，形状 (B,)
    alpha : float
        分区大小控制参数，越大分区越细（默认4.0）
    beta : float
        Smooth L1 loss 的参数（默认0.0，即标准L1）
    trunc : float | None
        截断阈值（用于鲁棒对齐，默认None）

    Returns
    -------
    loss : torch.Tensor
        损失值，标量
    misc : dict
        额外的监控指标：
        - 'truncated_error': 截断后的加权平均误差
        - 'delta': 相对误差 < 1.0 的点的比例
        - 'num_groups': 分区数量
        - 'points_per_group': 每个分区的平均点数

    Notes
    -----
    - 每次迭代使用随机旋转进行数据增强，避免固定分区边界伪影
    - 近处的点获得更高的权重（1/距离）
    - 不同batch的点不会被分到同一组

    Examples
    --------
    >>> pred = torch.randn(2, 64, 64, 3)
    >>> gt = torch.randn(2, 64, 64, 3)
    >>> mask = torch.ones(2, 64, 64, dtype=torch.bool)
    >>> global_scale = torch.ones(2)
    >>> loss, misc = radial_partition_local_loss(pred, gt, mask, global_scale)
    """
    device, dtype = pred_points.device, pred_points.dtype
    if packed is None:
        packed = RadialPartitionLocalLossCache.from_inputs(gt_points, mask)

    batch_size = packed.batch_size
    batch_shape = packed.batch_shape
    global_scale = global_scale.reshape(batch_size)

    pred_points_flat = packed.pred_points_flat(pred_points)
    gt_points_flat = packed.gt_points_flat
    pixel_weights_flat = packed.pixel_weights_flat
    batch_flat = packed.batch_flat
    gt_dist_flat = packed.gt_dist_flat
    geometry_weights_flat = packed.geometry_weights_flat
    weights_flat = packed.weights_flat

    if pixel_weights_flat.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), {}

    iflat_grouped, offsets_grouped, igroup_flat, num_groups = packed.grouped_indices(alpha)

    # Align shifts grouped by partition (vectorized, avoids per-group loops)
    # NOTE: Avoid fancy indexing `global_scale[batch_flat, None]` — its backward
    # triggers `indexing_backward_kernel_stride_1` which scatter-adds N gradients
    # back to batch_size elements via slow atomicAdd. Use index_select instead,
    # whose backward is a simple gather (much faster).
    global_scale_flat = global_scale.index_select(0, batch_flat).unsqueeze(-1)
    pred_points_flat = global_scale_flat * pred_points_flat
    shifts, _ = segment_align_shift(
        pred_points_flat.index_select(0, iflat_grouped).swapaxes(0, 1),
        gt_points_flat.index_select(0, iflat_grouped).swapaxes(0, 1),
        weights_flat.index_select(0, iflat_grouped),
        offsets=offsets_grouped,
    )  # (3, num_groups)
    shifts_flat = shifts.swapaxes(0, 1).index_select(0, igroup_flat)
    pred_points_flat = pred_points_flat + shifts_flat

    # 计算几何误差
    error_flat = (pred_points_flat - gt_points_flat).abs().mean(dim=-1) * geometry_weights_flat
    error_flat = _smooth(error_flat, beta=beta)

    # 加权损失
    loss_flat = pixel_weights_flat * error_flat

    # 按batch聚合损失
    loss_per_batch = torch.zeros(batch_size, dtype=dtype, device=device)
    loss_per_batch.scatter_add_(0, batch_flat, loss_flat)
    loss = loss_per_batch.reshape(batch_shape).mean()

    # 计算监控指标
    err = (pred_points_flat.detach() - gt_points_flat).norm(dim=-1) / gt_dist_flat
    points_per_group = len(iflat_grouped) / num_groups if num_groups > 0 else 0

    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), pixel_weights_flat).detach(),
        'delta': weighted_mean((err < 1.0).float(), pixel_weights_flat).detach(),
        'delta_0p1': weighted_mean((err < 0.1).float(), pixel_weights_flat).detach(),
        'delta_0p01': weighted_mean((err < 0.01).float(), pixel_weights_flat).detach(),
        'num_groups': num_groups,
        'points_per_group': points_per_group,
    }

    return loss, misc