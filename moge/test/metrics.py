from typing import *
from numbers import Number

import torch
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from ..utils.geometry_torch import (
    mask_aware_nearest_resize,
    intrinsics_to_fov,
    angle_diff_vec3
)
from ..utils.alignment import (
    align_points_scale_xyz_shift,
    align_points_xyz_shift,
    align_points_xyz_shift_with_scale,
    align_affine_lstsq,
    align_depth_scale,
    align_depth_affine,
    align_depth_shift_with_scale,
    align_points_scale,
)
from ..utils.tools import key_average


def rel_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    rel = (torch.abs(pred - gt) / (gt + eps)).mean()
    return rel.item()


def delta_depth(pred: torch.Tensor, gt: torch.Tensor, threshold: float, eps: float = 1e-6):
    ratio = torch.maximum(gt / (pred + eps), pred / (gt + eps))
    return (ratio < threshold).float().mean().item()


def depth_metrics_dict(pred: torch.Tensor, gt: torch.Tensor, local: bool = False) -> dict:
    metrics = {
        'rel': rel_depth(pred, gt),
    }
    if local:
        metrics['delta0.01'] = delta_depth(pred, gt, 1.01)
    else:
        metrics['delta1'] = delta_depth(pred, gt, 1.25)
    return metrics


def rel_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / (dist_gt + eps)).mean()
    return rel.item()

def delta_point(pred: torch.Tensor, gt: torch.Tensor, threshold: float):
    dist_pred = torch.norm(pred, dim=-1)
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)

    delta = (dist_err < threshold * torch.minimum(dist_gt, dist_pred)).float().mean()
    return delta.item()


def point_metrics_dict(pred: torch.Tensor, gt: torch.Tensor, local: bool = False) -> dict:
    metrics = {
        'rel': rel_point(pred, gt),
    }
    if local:
        metrics['delta0.01'] = delta_point(pred, gt, 0.01)
    else:
        metrics['delta1'] = delta_point(pred, gt, 0.25)
    return metrics


# Used only by the `points_local_moge2` group, where the error is normalized by each
# segment's ground-truth bounding-box diameter rather than by point distance. Named
# `_moge2` to keep them distinct from the `points_local` group's metrics.
def rel_point_moge2(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / diameter).mean()
    return rel.item()


def delta1_point_moge2(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
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

    pred_window = utils3d.pt.sliding_window(pred, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                 # [H, W, 2*R+1, 2*R+1]
    gt_window = utils3d.pt.sliding_window(gt, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                     # [H, W, 2*R+1, 2*R+1]
    mask_window = neighbor_mask & utils3d.pt.sliding_window(mask, window_size=2 * radius + 1, stride=1, dim=(-2, -1)) # [H, W, 2*R+1, 2*R+1]

    pred_rel = pred_window / pred[radius:-radius, radius:-radius, None, None]
    gt_rel = gt_window / gt[radius:-radius, radius:-radius, None, None]
    valid = mask[radius:-radius, radius:-radius, None, None] & mask_window

    f1_list = []
    w_list = t_list = torch.linspace(0.05, 0.25, 10).tolist()

    for t in t_list:
        pred_label = pred_rel > 1 + t
        gt_label = gt_rel > 1 + t
        TP = (pred_label & gt_label & valid).float().sum()
        precision = TP / (pred_label & valid).float().sum().clamp_min(1e-12)
        recall = TP / (gt_label & valid).float().sum().clamp_min(1e-12)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        f1_list.append(f1.item())

    f1_avg = sum(w * f1 for w, f1 in zip(w_list, f1_list)) / sum(w_list)
    return f1_avg


_MG_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    'global': (
        'depth_affine_invariant',
        'depth_scale_invariant',
        'disparity_affine_invariant',
        'fov_x',
        'points_affine_invariant',
        'points_scale_invariant',
    ),
    'metric': (
        'depth_metric',
        'points_metric',
    ),
    'local': (
        'depth_local',
        'points_local',
    ),
}
# The two named evaluation suites.
#
# `moge2` reproduces the metric set the pre-MoGe-3 evaluator reported: the global
# and metric groups, per-segment `points_local_moge2`, and boundary F1 at radii 1/2/3.
# `moge3` is the current suite, which swaps `points_local_moge2` for the high-frequency
# per-segment groups and reports boundary F1 at radius 1 only.
_MG_CATEGORIES['moge2'] = (
    *_MG_CATEGORIES['global'],
    *_MG_CATEGORIES['metric'],
    'points_local_moge2',
    'boundary_f1_r123',
)
_MG_CATEGORIES['moge3'] = (
    *_MG_CATEGORIES['global'],
    *_MG_CATEGORIES['metric'],
    *_MG_CATEGORIES['local'],
    'boundary_f1_r1',
)

_MG_ALIASES = {
    'fov': 'fov_x',
}

_MG_GROUPS: FrozenSet[str] = frozenset({
    'disparity_affine_invariant',
    'depth_metric',
    'depth_affine_invariant',
    'depth_scale_invariant',
    'points_metric',
    'points_scale_invariant',
    'points_affine_invariant',
    'points_local_moge2',
    'normal',
    'fov_x',
    'boundary_f1_r1',
    'boundary_f1_r123',
    'depth_local',
    'points_local',
})

_MG_DEFAULT = 'moge3'


# Genuine data dependencies only: the `*_local` groups consume `global_*_scale`, which
# is computed inside the corresponding `*_scale_invariant` block. Groups absent from
# this table have no dependencies. Dependency groups are computed and reported as
# normal metrics.
_MG_DEPENDENCIES: Dict[str, List[str]] = {
    'depth_local': ['depth_scale_invariant'],
    'points_local': ['points_scale_invariant'],
}


def resolve_metric_groups(mg: Optional[Union[str, Sequence[str]]] = None) -> List[str]:
    """Parse, expand, de-duplicate, and dependency-resolve metric groups.

    `mg` may be a comma-separated string or a sequence whose items may
    themselves contain commas. A missing or empty value defaults to `moge3`.
    Category names are removed during expansion; the return value contains only
    concrete metric groups in stable first-seen order. Raises `ValueError` on an unrecognized name.
    """
    if mg is None:
        raw_items: Sequence[str] = (_MG_DEFAULT,)
    elif isinstance(mg, str):
        raw_items = (mg,)
    else:
        raw_items = mg

    tokens = [
        token.strip().lower()
        for item in raw_items
        for token in item.split(',')
        if token.strip()
    ]
    if not tokens:
        tokens = [_MG_DEFAULT]

    unknown = [
        t for t in tokens
        if _MG_ALIASES.get(t, t) not in _MG_CATEGORIES and _MG_ALIASES.get(t, t) not in _MG_GROUPS
    ]
    if unknown:
        known = sorted(set(_MG_CATEGORIES) | set(_MG_ALIASES) | _MG_GROUPS)
        raise ValueError(
            f"Unknown metric group(s): {', '.join(sorted(set(unknown)))}. "
            f"Known categories and groups: {', '.join(known)}"
        )

    resolved: List[str] = []
    seen = set()

    def append_group(group: str) -> None:
        group = _MG_ALIASES.get(group, group)
        if group in _MG_CATEGORIES:
            for child in _MG_CATEGORIES[group]:
                append_group(child)
        elif group not in seen:
            seen.add(group)
            resolved.append(group)

    for token in tokens:
        append_group(token)

    # Iterate over the growing list so transitive dependencies are resolved in
    # deterministic order. `append_group` also handles category-valued deps
    # should one be added in the future.
    index = 0
    while index < len(resolved):
        for dependency in _MG_DEPENDENCIES.get(resolved[index], ()):
            append_group(dependency)
        index += 1
    return resolved


@torch.no_grad()
def compute_metrics(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    vis: bool = False,
    mg: Optional[Union[str, Sequence[str]]] = None,
    is_ppd: bool = False,
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
        - `depth_mask`: mask indicating valid pixels in the ground-truth depth.
        - `points`: point map ground truth in camera coordinates.
        - `intrinsics`: normalized ground-truth camera intrinsics matrix.
        - `is_metric`: whether the depth is in metric units.
        - `has_sharp_boundary`: whether the ground truth resolves depth discontinuities
          sharply enough for the boundary groups to be meaningful.

    #### Optional keys in gt (the groups that need them are skipped if absent):
        - `normal`, `normal_mask`: normal map ground truth and its valid mask, for `normal`.
        - `segmentation_mask`, `segmentation_labels`: per-object segment id map and its
          `{name: id}` table, for `points_local_moge2`.
        - `local_mask`, `local_segmentation`: high-frequency region mask and the segment id
          map within it, for `depth_local` / `points_local`.

    #### Special flags:
        - `is_ppd`: if True, treat `pred['depth_affine_invariant']` as Pixel-Perfect Depth's
          raw output, which is affine-invariant in log-depth space
          (`pred ≈ a * log(depth + 1) + b`). The affine alignment is then performed
          in log-depth space to match PPD's official evaluator.

    """
    mg = frozenset(resolve_metric_groups(mg))

    metrics = {}
    misc = {}

    # Cross-block state. Every name below is written inside an `if '<group>' in mg:`
    # block and read outside it, so pre-binding to None makes a restricted `mg`
    # degrade gracefully instead of raising UnboundLocalError, and keeps
    # `_MG_DEPENDENCIES` purely about completeness rather than crash-avoidance.
    pred_disparity_affine_invariant = None
    pred_depth_affine_invariant = None
    pred_depth_scale_invariant = None
    pred_points_affine_invariant = None
    pred_points_scale_invariant = None
    global_depth_scale = None
    global_points_scale = None
    # Aligned-map slots, consumed by the boundary groups and by `vis` below.
    metric_depth_aligned = None
    scale_depth_aligned = None
    affine_depth_aligned = None
    disparity_depth_aligned = None
    metric_points_aligned = None
    scale_points_aligned = None
    affine_points_aligned = None

    mask = gt['depth_mask']
    gt_depth = gt['depth']
    gt_points = gt['points']

    _, lr_mask, lr_index = mask_aware_nearest_resize(None, mask, (64, 64), return_index=True)

    only_depth = not any('point' in k for k in pred)

    if 'disparity_affine_invariant' in mg:
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

            metrics['disparity_affine_invariant'] = depth_metrics_dict(pred_depth[mask], gt_depth[mask])

            if vis:
                disparity_depth_aligned = 1 / pred_disp.clamp_min(1e-6)

    if 'depth_metric' in mg:
        # Metric depth
        if gt['is_metric'] and 'depth_metric' in pred:
            pred_depth = pred['depth_metric']
            metrics['depth_metric'] = depth_metrics_dict(pred_depth[mask], gt_depth[mask])

            metric_depth_aligned = pred_depth

    if 'depth_affine_invariant' in mg:
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
            if is_ppd:
                # PPD's raw output is affine-invariant in log-depth space:
                #   pred ≈ a * log(gt + 1) + b
                # Mirror PPD's official `recover_metric_depth_ransac`.
                log_gt_lr_masked = torch.log(gt_depth_lr_masked + 1.0)
                scale, shift = align_depth_affine(
                    pred_depth_lr_masked, log_gt_lr_masked, 1 / gt_depth_lr_masked
                )
                pred_depth = torch.exp(pred_depth * scale + shift) - 1.0
                pred_depth = pred_depth.clamp(min=1e-3, max=gt_depth[mask].max().item())
            else:
                scale, shift = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
                pred_depth = pred_depth * scale + shift

            metrics['depth_affine_invariant'] = depth_metrics_dict(pred_depth[mask], gt_depth[mask])
            affine_depth_aligned = pred_depth

    if 'depth_scale_invariant' in mg:
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
            global_depth_scale = scale

            metrics['depth_scale_invariant'] = depth_metrics_dict(pred_depth[mask], gt_depth[mask])
            scale_depth_aligned = pred_depth

    if 'points_metric' in mg:
        # Metric points
        if 'points_metric' in pred and gt['is_metric']:
            pred_points = pred['points_metric']

            pred_points_lr_masked, gt_points_lr_masked = pred_points[lr_index][lr_mask], gt_points[lr_index][lr_mask]
            shift = align_points_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
            pred_points = pred_points + shift

            metrics['points_metric'] = point_metrics_dict(pred_points[mask], gt_points[mask])

            # NOTE: the legacy visualization dumped the *unshifted* metric points.
            metric_points_aligned = pred['points_metric']

    if 'points_scale_invariant' in mg:
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
            global_points_scale = scale

            metrics['points_scale_invariant'] = point_metrics_dict(pred_points[mask], gt_points[mask])
            scale_points_aligned = pred_points

    if 'points_affine_invariant' in mg:
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

            metrics['points_affine_invariant'] = point_metrics_dict(pred_points[mask], gt_points[mask])

            affine_points_aligned = pred_points

    if 'points_local_moge2' in mg:
        # MoGe-2's local metric: per-segment affine-aligned point error, normalized
        # by each segment's ground-truth bounding-box diameter.
        if (
            gt.get('segmentation_mask') is not None
            and gt.get('segmentation_labels')
            and 'points' in gt
            and any('points' in k for k in pred)
        ):
            local_pred_points = next(pred[k] for k in pred if 'points' in k)
            segmentation_mask = gt['segmentation_mask']
            segmentation_labels = gt['segmentation_labels']
            segmentation_mask_lr = segmentation_mask[lr_index]
            per_segment_metrics = []
            for _, seg_id in segmentation_labels.items():
                valid_mask_lr = (segmentation_mask_lr == seg_id) & lr_mask
                if valid_mask_lr.sum().item() < 10:
                    continue
                valid_mask = (segmentation_mask == seg_id) & mask
                if not valid_mask.any():
                    continue

                pred_points_masked = local_pred_points[valid_mask]
                gt_points_masked = gt_points[valid_mask]
                pred_points_masked_lr = local_pred_points[lr_index][valid_mask_lr]
                gt_points_masked_lr = gt_points[lr_index][valid_mask_lr]

                diameter = (gt_points_masked.max(dim=0).values - gt_points_masked.min(dim=0).values).max()
                scale, shift = align_points_scale_xyz_shift(
                    pred_points_masked_lr, gt_points_masked_lr,
                    1 / diameter.expand(gt_points_masked_lr.shape[0]),
                )
                pred_points_masked = pred_points_masked * scale + shift

                per_segment_metrics.append({
                    'rel': rel_point_moge2(pred_points_masked, gt_points_masked, diameter),
                    'delta1': delta1_point_moge2(pred_points_masked, gt_points_masked, diameter),
                })

            if per_segment_metrics:
                metrics['points_local_moge2'] = key_average(per_segment_metrics)

    if 'normal' in mg:
        # Normal
        if 'normal' in pred and 'normal' in gt:
            pred_normal = pred['normal']
            gt_normal = gt['normal']
            gt_normal_mask = gt['normal_mask']
            metrics['normal'] = {
                'median': angle_diff_vec3(pred_normal[gt_normal_mask], gt_normal[gt_normal_mask]).median().rad2deg().item(),
                'mean': angle_diff_vec3(pred_normal[gt_normal_mask], gt_normal[gt_normal_mask]).mean().rad2deg().item(),
            }

    if 'fov_x' in mg:
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

    if 'depth_metric' in pred:
        boundary_depth = pred['depth_metric']
    elif 'depth_scale_invariant' in pred:
        boundary_depth = pred['depth_scale_invariant']
    elif 'disparity_affine_invariant' in pred:
        boundary_depth = 1 / pred['disparity_affine_invariant'].clamp_min(1e-6)
    elif 'depth_affine_invariant' in pred:
        # Only shift-free option left, though such a prediction is defined only up to
        # an affine transform, so its raw ratios carry an arbitrary offset.
        boundary_depth = pred['depth_affine_invariant']
    else:
        boundary_depth = None

    # `boundary_f1_r1` reports radius 1 (MoGe-3); `boundary_f1_r123` reports radii
    # 1/2/3 (MoGe-2). Collecting the radii into a set first means radius 1 is computed
    # once even when both groups are requested, e.g. `--mg moge2,moge3`.
    boundary_radii = set()
    if 'boundary_f1_r1' in mg:
        boundary_radii.add(1)
    if 'boundary_f1_r123' in mg:
        boundary_radii.update((1, 2, 3))

    if boundary_radii and boundary_depth is not None and gt['has_sharp_boundary']:
        boundary = metrics.setdefault('boundary', {})
        for radius in sorted(boundary_radii):
            boundary[f'radius{radius}_f1'] = boundary_f1(boundary_depth, gt_depth, mask, radius=radius)

    # MoGe-3's local metrics: each segment is aligned to the GT independently by a
    # per-segment shift that shares the globally-fitted scale, then the per-segment
    # metrics are averaged unweighted. Isolating local shape error this way makes the
    # result insensitive to the global scale/shift error the other groups measure.
    #
    # Segments come from `local_segmentation` (full SAM segment id map, >0 per id).
    if ('depth_local' in mg or 'points_local' in mg) and gt.get('local_mask', None) is not None:
        hf_mask = gt['local_mask'] & mask
        if gt.get('local_segmentation', None) is not None:
            hf_seg = gt['local_segmentation'].to(torch.long)
            # Cleaning may zero-out some pixels; restrict hf_mask accordingly.
            hf_mask = hf_mask & (hf_seg > 0)
            hf_labels = torch.where(hf_mask, hf_seg, torch.zeros_like(hf_seg))
        else:
            hf_labels = hf_mask.to(torch.long)

        min_seg_pixels = 10
        # Cap the number of points used for per-segment alignment to avoid OOM in
        # `align_depth_shift_with_scale` / `align_points_xyz_shift_with_scale` (which
        # build O(N) or O(N^2) intermediates).
        max_align_pts = 4096

        # Group the high-frequency pixels by segment id once and share the result
        # between the depth and points groups. Sorting the flat pixel indices by label
        # makes every segment a contiguous slice, so scoring one costs O(segment)
        # instead of an O(H*W) mask comparison, and the sizes reach the host in a
        # single sync instead of one per segment per group. The sort must be stable:
        # `_subsample_idx` picks pixels by position, so a run-to-run reshuffle within
        # a segment would make the subsample non-deterministic.
        pixel_index = hf_mask.reshape(-1).nonzero(as_tuple=True)[0]
        pixel_label = hf_labels.reshape(-1)[pixel_index]
        order = torch.argsort(pixel_label, stable=True)
        pixel_index, pixel_label = pixel_index[order], pixel_label[order]
        seg_id, seg_count = torch.unique_consecutive(pixel_label, return_counts=True)
        seg_start = torch.cumsum(seg_count, dim=0) - seg_count
        segments = [
            (sid, start, count)
            for sid, start, count in zip(seg_id.tolist(), seg_start.tolist(), seg_count.tolist())
            if sid > 0 and count >= min_seg_pixels
        ]

        def _subsample_idx(n: int, sid: int) -> Optional[torch.Tensor]:
            if n <= max_align_pts:
                return None
            # Seed per segment id rather than sharing one generator across the loop:
            # a shared stream would make a segment's subsample depend on how many
            # segments — and which metric groups — were processed before it, so
            # `--mg points_local` and `--mg moge3` would disagree on the same image.
            rng = torch.Generator(device=pixel_index.device).manual_seed(sid)
            return torch.randperm(n, generator=rng, device=pixel_index.device)[:max_align_pts]

        def _average_over_segments(score_segment: Callable[..., Dict[str, Number]]) -> Optional[Dict[str, Number]]:
            """Apply `score_segment(seg_index, fit_idx)` to every high-frequency segment
            and average the per-segment metric dicts unweighted.

            `seg_index` indexes the flattened image; `fit_idx` subsamples it for the
            alignment fit only, as the metrics themselves are always computed over the
            full segment. Returns None if no segment was large enough to score.
            """
            per_seg = [
                score_segment(pixel_index[start:start + count], _subsample_idx(count, sid))
                for sid, start, count in segments
            ]
            return key_average(per_seg) if per_seg else None

        # ---- depth ----
        if (
            'depth_local' in mg
            and pred_depth_scale_invariant is not None
            and global_depth_scale is not None
        ):
            pred_depth_flat = pred_depth_scale_invariant.reshape(-1).detach()
            gt_depth_flat = gt_depth.reshape(-1).detach()

            def _score_depth(seg_index: torch.Tensor, sub: Optional[torch.Tensor]) -> Dict[str, Number]:
                # Per-segment scalar shift only; share the globally-fitted scale.
                p = pred_depth_flat[seg_index]
                g = gt_depth_flat[seg_index]
                p_fit = p if sub is None else p[sub]
                g_fit = g if sub is None else g[sub]
                t = align_depth_shift_with_scale(p_fit, g_fit, 1 / g_fit, global_depth_scale)
                # Clamp the shift so `scale * p + t` stays positive across the full
                # (non-subsampled) segment, in case the subsample missed the minimum.
                full_min = (p * global_depth_scale).min().detach()
                t = torch.maximum(t, -full_min + 1e-6)
                return depth_metrics_dict(p * global_depth_scale + t, g, local=True)

            result = _average_over_segments(_score_depth)
            if result is not None:
                metrics['depth_local'] = result

        # ---- points ----
        if (
            'points_local' in mg
            and pred_points_scale_invariant is not None
            and global_points_scale is not None
        ):
            pred_points_flat = pred_points_scale_invariant.reshape(-1, 3).detach()
            gt_points_flat = gt_points.reshape(-1, 3).detach()

            def _score_points(seg_index: torch.Tensor, sub: Optional[torch.Tensor]) -> Dict[str, Number]:
                # Per-segment xyz shift only; share the globally-fitted scale.
                p = pred_points_flat[seg_index]
                g = gt_points_flat[seg_index]
                w = 1 / g.norm(dim=-1).clamp_min(1e-6)
                p_fit = p if sub is None else p[sub]
                g_fit = g if sub is None else g[sub]
                w_fit = w if sub is None else w[sub]
                shift = align_points_xyz_shift_with_scale(p_fit, g_fit, w_fit, global_points_scale)
                return point_metrics_dict(p * global_points_scale + shift, g, local=True)

            result = _average_over_segments(_score_points)
            if result is not None:
                metrics['points_local'] = result

    if vis:
        # Reproduce the legacy "first non-None wins" priority. The order below is
        # the legacy *source* order, which no longer matches this function's block
        # order, so it is spelled out explicitly.
        pred_depth_aligned = next(
            (x for x in (
                metric_depth_aligned,
                scale_depth_aligned,
                affine_depth_aligned,
                disparity_depth_aligned,
            ) if x is not None),
            None,
        )
        pred_points_aligned = next(
            (x for x in (
                metric_points_aligned,
                scale_points_aligned,
                affine_points_aligned,
            ) if x is not None),
            None,
        )
        # `only_depth` implies no point prediction exists, so the branches below
        # are mutually exclusive. Both are guarded against a restricted `mg` that
        # produced no aligned map at all.
        if only_depth:
            if pred_depth_aligned is not None:
                misc['pred_points'] = utils3d.pt.depth_map_to_point_map(pred_depth_aligned, intrinsics=gt['intrinsics'])
        elif pred_points_aligned is not None:
            misc['pred_points'] = pred_points_aligned
        if pred_depth_aligned is not None:
            misc['pred_depth'] = pred_depth_aligned

    return metrics, misc
