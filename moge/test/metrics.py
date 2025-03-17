from typing import *
from numbers import Number

import torch
import torch.nn.functional as F
import numpy as np
import utils3d

from ..utils.geometry_torch import (
    weighted_mean, 
    mask_aware_nearest_resize,
    intrinsics_to_fov
)
from ..utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale_xyz_shift, 
    align_points_xyz_shift,
    align_affine_lstsq, 
    align_depth_scale, 
    align_depth_affine, 
    align_points_scale,
)
from ..utils.tools import key_average, timeit


def rel_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    rel = (torch.abs(pred - gt) / (gt + eps)).mean()
    return rel.item()


def delta1_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    delta1 = (torch.maximum(gt / pred, pred / gt) < 1.25).float().mean()
    return delta1.item()


def rel_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / (dist_gt + eps)).mean()
    return rel.item()


def delta1_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_pred = torch.norm(pred, dim=-1)
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)

    delta1 = (dist_err < 0.25 * torch.minimum(dist_gt, dist_pred)).float().mean()
    return delta1.item()


def rel_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / diameter).mean()
    return rel.item()


def delta1_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    delta1 = (dist_err < 0.25 * diameter).float().mean()
    return delta1.item()


def boundary_f1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, radius: int = 1):
    neighbor_x, neight_y = torch.meshgrid(
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        indexing='xy'
    )
    neighbor_mask = (neighbor_x ** 2 + neight_y ** 2) <= radius ** 2 + 1e-5

    pred_window = utils3d.torch.sliding_window_2d(pred, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                 # [H, W, 2*R+1, 2*R+1]
    gt_window = utils3d.torch.sliding_window_2d(gt, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                     # [H, W, 2*R+1, 2*R+1]
    mask_window = neighbor_mask & utils3d.torch.sliding_window_2d(mask, window_size=2 * radius + 1, stride=1, dim=(-2, -1)) # [H, W, 2*R+1, 2*R+1]

    pred_rel = pred_window / pred[radius:-radius, radius:-radius, None, None]
    gt_rel = gt_window / gt[radius:-radius, radius:-radius, None, None]
    valid = mask[radius:-radius, radius:-radius, None, None] & mask_window
    
    f1_list = []
    w_list = t_list = torch.linspace(0.05, 0.25, 10).tolist()

    for t in t_list:
        pred_label = pred_rel > 1 + t
        gt_label = gt_rel > 1 + t
        TP = (pred_label & gt_label & valid).float().sum()
        precision = TP / (gt_label & valid).float().sum().clamp_min(1e-12)
        recall = TP / (pred_label & valid).float().sum().clamp_min(1e-12)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        f1_list.append(f1.item())
    
    f1_avg = sum(w * f1 for w, f1 in zip(w_list, f1_list)) / sum(w_list)
    return f1_avg


def compute_metrics(
    pred: Dict[str, torch.Tensor], 
    gt: Dict[str, torch.Tensor], 
    vis: bool = False
) -> Tuple[Dict[str, Dict[str, Number]], Dict[str, torch.Tensor]]:
    """
    A unified function to compute metrics for different types of predictions and ground truths.
    
    #### Supported keys in pred:
        - `disparity_affine_invariant`: disparity map predicted by a depth estimator with scale and shift invariant. 
        - `depth_scale_invariant`: depth map predicted by a depth estimator with scale invariant. 
        - `depth_affine_invariant`: depth map predicted by a depth estimator with scale and shift invariant. 
        - `depth_metric`: depth map predicted by a depth estimator with no scale or shift. 
        - `points_scale_invariant`: point map predicted by a point estimator with scale invariant. 
        - `points_affine_invariant`: point map predicted by a point estimator with scale and xyz shift invariant. 
        - `points_metric`: point map predicted by a point estimator with no scale or shift. 
        - `intrinsics`: normalized camera intrinsics matrix.

    #### Required keys in gt:
        - `depth`: depth map ground truth (in metric units if `depth_metric` is used)
        - `points`: point map ground truth in camera coordinates.
        - `mask`: mask indicating valid pixels in the ground truth.
        - `intrinsics`: normalized ground-truth camera intrinsics matrix.
        - `is_metric`: whether the depth is in metric units.
    """
    metrics = {}
    misc = {}
    
    mask = gt['depth_mask']
    gt_depth = gt['depth']
    gt_points = gt['points']

    height, width = mask.shape[-2:]
    _, lr_mask, lr_index = mask_aware_nearest_resize(None, mask, (64, 64), return_index=True)
    
    only_depth = not any('point' in k for k in pred)
    pred_depth_aligned, pred_points_aligned = None, None

    # Metric depth
    if 'depth_metric' in pred and gt['is_metric']:
        pred_depth, gt_depth = pred['depth_metric'], gt['depth']
        metrics['depth_metric'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Scale-invariant depth
    if 'depth_scale_invariant' in pred:
        pred_depth_scale_invariant = pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_depth_scale_invariant = pred['depth_metric']
    else:
        pred_depth_scale_invariant = None

    if pred_depth_scale_invariant is not None:
        pred_depth = pred_depth_scale_invariant

        pred_depth_lr_masked, gt_depth_lr_masked = pred_depth[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
        scale = align_depth_scale(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
        pred_depth = pred_depth * scale
    
        metrics['depth_scale_invariant'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Affine-invariant depth
    if 'depth_affine_invariant' in pred:
        pred_depth_affine_invariant = pred['depth_affine_invariant']
    elif 'depth_scale_invariant' in pred:
        pred_depth_affine_invariant = pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_depth_affine_invariant = pred['depth_metric']
    else:
        pred_depth_affine_invariant = None

    if pred_depth_affine_invariant is not None:
        pred_depth = pred_depth_affine_invariant

        pred_depth_lr_masked, gt_depth_lr_masked = pred_depth[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
        scale, shift = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
        pred_depth = pred_depth * scale + shift

        metrics['depth_affine_invariant'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Affine-invariant disparity
    if 'disparity_affine_invariant' in pred:
        pred_disparity_affine_invariant = pred['disparity_affine_invariant']
    elif 'depth_scale_invariant' in pred:
        pred_disparity_affine_invariant = 1 / pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_disparity_affine_invariant = 1 / pred['depth_metric']
    else:
        pred_disparity_affine_invariant = None
        
    if pred_disparity_affine_invariant is not None:
        pred_disp = pred_disparity_affine_invariant
        
        scale, shift = align_affine_lstsq(pred_disp[mask], 1 / gt_depth[mask])
        pred_disp = pred_disp * scale + shift

        # NOTE: The alignment is done on the disparity map could introduce extreme outliers at disparities close to 0.
        #       Therefore we clamp the disparities by minimum ground truth disparity.
        pred_depth = 1 / pred_disp.clamp_min(1 / gt_depth[mask].max().item())

        metrics['disparity_affine_invariant'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = 1 / pred_disp.clamp_min(1e-6)

    # Metric points
    if 'points_metric' in pred and gt['is_metric']:
        pred_points = pred['points_metric']

        pred_points_lr_masked, gt_points_lr_masked = pred_points[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        shift = align_points_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points + shift

        metrics['points_metric'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask])
        }

        if pred_points_aligned is None:
            pred_points_aligned = pred['points_metric']

    # Scale-invariant points (in camera space)
    if 'points_scale_invariant' in pred:
        pred_points_scale_invariant = pred['points_scale_invariant']
    elif 'points_metric' in pred:
        pred_points_scale_invariant = pred['points_metric']
    else:
        pred_points_scale_invariant = None
        
    if pred_points_scale_invariant is not None:
        pred_points = pred_points_scale_invariant

        pred_points_lr_masked, gt_points_lr_masked = pred_points_scale_invariant[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        scale = align_points_scale(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points * scale

        metrics['points_scale_invariant'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask])
        }

        if vis and pred_points_aligned is None:
            pred_points_aligned = pred['points_scale_invariant'] * scale
    
    # Affine-invariant points
    if 'points_affine_invariant' in pred:
        pred_points_affine_invariant = pred['points_affine_invariant']
    elif 'points_scale_invariant' in pred:
        pred_points_affine_invariant = pred['points_scale_invariant']
    elif 'points_metric' in pred:
        pred_points_affine_invariant = pred['points_metric']
    else:
        pred_points_affine_invariant = None

    if pred_points_affine_invariant is not None:
        pred_points = pred_points_affine_invariant

        pred_points_lr_masked, gt_points_lr_masked = pred_points[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        scale, shift = align_points_scale_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points * scale + shift

        metrics['points_affine_invariant'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask])
        }

        if vis and pred_points_aligned is None:
            pred_points_aligned = pred['points_affine_invariant'] * scale + shift

    # Local points
    if 'segmentation_mask' in gt and 'points' in gt and any('points' in k for k in pred.keys()):
        pred_points = next(pred[k] for k in pred.keys() if 'points' in k)
        gt_points = gt['points']
        segmentation_mask = gt['segmentation_mask']
        segmentation_labels = gt['segmentation_labels']
        segmentation_mask_lr =  segmentation_mask[lr_index]
        local_points_metrics = []
        for _, seg_id in segmentation_labels.items():
            valid_mask = (segmentation_mask == seg_id) & mask
            
            pred_points_masked = pred_points[valid_mask]
            gt_points_masked = gt_points[valid_mask]

            valid_mask_lr = (segmentation_mask_lr == seg_id) & lr_mask
            if valid_mask_lr.sum().item() < 10:
                continue
            pred_points_masked_lr = pred_points[lr_index][valid_mask_lr]
            gt_points_masked_lr = gt_points[lr_index][valid_mask_lr]
            diameter = (gt_points_masked.max(dim=0).values - gt_points_masked.min(dim=0).values).max()
            scale, shift = align_points_scale_xyz_shift(pred_points_masked_lr, gt_points_masked_lr, 1 / diameter.expand(gt_points_masked_lr.shape[0]))
            pred_points_masked = pred_points_masked * scale + shift

            local_points_metrics.append({
                'rel': rel_point_local(pred_points_masked, gt_points_masked, diameter),
                'delta1': delta1_point_local(pred_points_masked, gt_points_masked, diameter),
            })
        
        metrics['local_points'] = key_average(local_points_metrics)

    # FOV. NOTE: If there is no random augmentation applied to the input images, all GT FOV are generallly the same. 
    #            Fair evaluation of FOV requires random augmentation.
    if 'intrinsics' in pred and 'intrinsics' in gt:
        pred_intrinsics = pred['intrinsics']
        gt_intrinsics = gt['intrinsics']
        pred_fov_x, pred_fov_y = intrinsics_to_fov(pred_intrinsics)
        gt_fov_x, gt_fov_y = intrinsics_to_fov(gt_intrinsics)
        metrics['fov_x'] = {
            'mae': torch.rad2deg(pred_fov_x - gt_fov_x).abs().mean().item(),
            'deviation': torch.rad2deg(pred_fov_x - gt_fov_x).item(),
        }

    # Boundary F1
    if pred_depth_aligned is not None and gt['has_sharp_boundary']:
        metrics['boundary'] = {
            'radius1_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=1),
            'radius2_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=2),
            'radius3_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=3),
        }

    if vis:
        if pred_points_aligned is not None:
            misc['pred_points'] = pred_points_aligned
        if only_depth:
            misc['pred_points'] = utils3d.torch.depth_to_points(pred_depth_aligned, intrinsics=gt['intrinsics'])
        if pred_depth_aligned is not None:
            misc['pred_depth'] = pred_depth_aligned

    return metrics, misc