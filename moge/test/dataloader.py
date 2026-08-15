import os
from typing import *
from pathlib import Path
import math

import numpy as np
import torch
from PIL import Image
import cv2
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d
import pipeline

from ..utils.geometry_numpy import focal_to_fov_numpy, norm3d
from ..utils.io import *
from ..utils.tools import timeit


class EvalDataLoaderPipeline:
    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        split: str = '.index.txt',
        depth: str = 'depth.png',
        drop_max_depth: float = 1000.,
        segmentation: str = None,
        normal: str = None,
        local_mask: str = None,
        local_segmentation: str = None,
        max_segments: int = 100,
        min_seg_area: int = 1000,
        depth_unit: str = None,
        has_sharp_boundary = False,
        subset: int = None,
    ):
        num_load_workers = int(os.environ.get('MOGE_NUM_LOAD_WORKERS', 4))
        num_process_workers = int(os.environ.get('MOGE_NUM_PROCESS_WORKERS', 8))
    
        filenames = Path(path).joinpath(split).read_text(encoding='utf-8').splitlines()
        filenames = filenames[::subset]
        self.width = width
        self.height = height
        self.drop_max_depth = drop_max_depth
        self.path = Path(path)
        self.filenames = filenames
        self.depth_file = depth
        self.segmentation_file = segmentation
        self.normal_file = normal
        self.local_mask_file = local_mask
        self.local_segmentation_file = local_segmentation
        self.max_segments = max_segments
        self.min_seg_area = min_seg_area
        self.depth_unit = depth_unit
        self.has_sharp_boundary = has_sharp_boundary

        self.rng = np.random.default_rng(seed=0)
        
        self.pipeline = pipeline.Sequential([
            self._generator,
            pipeline.Parallel([self._load_instance] * num_load_workers),
            pipeline.Parallel([self._process_instance] * num_process_workers),
            pipeline.Buffer(4)
        ])

    def __len__(self):
        return math.ceil(len(self.filenames)) 

    def _generator(self):
        for idx in range(len(self)):
            yield idx
    
    def _load_instance(self, idx):
        if idx >= len(self.filenames):
            return None
        
        path = self.path.joinpath(self.filenames[idx])

        instance = {
            'filename': self.filenames[idx],
            'width': self.width,
            'height': self.height,
        }
        instance['image'] = read_image(Path(path, 'image.jpg'))

        depth = read_depth(Path(path, self.depth_file))  # ignore depth unit from depth file, use config instead
        instance.update({
            'depth': np.nan_to_num(depth, nan=1, posinf=1, neginf=1),
            'depth_mask': np.isfinite(depth),
            'depth_mask_inf': np.isinf(depth),
        })

        if self.segmentation_file is not None:
            segmentation_mask, segmentation_labels = read_segmentation(Path(path, self.segmentation_file))
            if segmentation_labels is None:
                # SAM-style id maps (sam_segmentation*.png) carry no PNG `labels`
                # metadata. Synthesize one entry per non-zero id so the downstream
                # label filtering in _process_instance still works.
                segmentation_labels = {f'{int(i):05d}': int(i) for i in np.unique(segmentation_mask) if i > 0}
            instance.update({
                'segmentation_mask': segmentation_mask,
                'segmentation_labels': segmentation_labels,
            })

        meta = read_json(Path(path, 'meta.json'))
        instance['intrinsics'] = np.array(meta['intrinsics'], dtype=np.float32)

        if self.normal_file is not None:
            normal = read_normal(Path(path, self.normal_file))
            # `read_normal` marks invalid pixels with NaN, so derive the mask
            # before scrubbing them.
            normal_mask = np.isfinite(normal).all(axis=-1)
            normal = np.nan_to_num(normal, nan=0, posinf=0, neginf=0)
            instance.update({
                'normal': normal,
                'normal_mask': normal_mask,
            })

        if self.local_mask_file is not None:
            local_mask_path = Path(path, self.local_mask_file)
            if local_mask_path.exists():
                local_mask = cv2.imread(str(local_mask_path), cv2.IMREAD_GRAYSCALE)
                instance['local_mask'] = local_mask > 0

                if self.local_segmentation_file is not None:
                    hf_seg_path = Path(path, self.local_segmentation_file)
                    if hf_seg_path.exists():
                        hf_seg = cv2.imread(str(hf_seg_path), cv2.IMREAD_UNCHANGED)
                        if hf_seg.shape[:2] != instance['local_mask'].shape:
                            hf_seg = cv2.resize(
                                hf_seg,
                                (instance['local_mask'].shape[1], instance['local_mask'].shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        instance['local_segmentation'] = hf_seg.astype(np.int32)

        return instance

    def _process_instance(self, instance: dict):
        if instance is None:
            return None
        
        image, depth, depth_mask, intrinsics = instance['image'], instance['depth'], instance['depth_mask'], instance['intrinsics']
        segmentation_mask, segmentation_labels = instance.get('segmentation_mask', None), instance.get('segmentation_labels', None)
        normal, normal_mask = instance.get('normal', None), instance.get('normal_mask', None)

        raw_height, raw_width = image.shape[:2]
        raw_horizontal, raw_vertical = abs(1.0 / intrinsics[0, 0]), abs(1.0 / intrinsics[1, 1])
        raw_pixel_w, raw_pixel_h = raw_horizontal / raw_width, raw_vertical / raw_height
        tgt_width, tgt_height = instance['width'], instance['height']
        tgt_aspect = tgt_width / tgt_height

        # set expected target view field
        tgt_horizontal = min(raw_horizontal, raw_vertical * tgt_aspect)
        tgt_vertical = tgt_horizontal / tgt_aspect

        # set target view direction
        cu, cv = 0.5, 0.5
        direction = utils3d.np.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=intrinsics)[0]
        R = utils3d.np.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

        # restrict target view field within the raw view
        corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
        corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(intrinsics).T @ R.T)   # corners in viewport's camera plane
        corners = corners[:, :2] / corners[:, 2:3]

        warp_horizontal, warp_vertical = abs(1.0 / intrinsics[0, 0]), abs(1.0 / intrinsics[1, 1])
        for i in range(4):
            intersection, _ = utils3d.np.ray_intersection(
                np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
                corners[i - 1], corners[i] - corners[i - 1],
            )
            warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
        tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)

        # get target view intrinsics
        fx, fy = 1.0 / tgt_horizontal, 1.0 / tgt_vertical
        tgt_intrinsics = utils3d.np.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)
        
        # do homogeneous transformation with the rotation and intrinsics
        # 4.1 The image and depth is resized first to approximately the same pixel size as the target image with PIL's antialiasing resampling
        tgt_pixel_w, tgt_pixel_h = tgt_horizontal / tgt_width, tgt_vertical / tgt_height        # (should be exactly the same for x and y axes)
        rescaled_w, rescaled_h = int(raw_width * raw_pixel_w / tgt_pixel_w), int(raw_height * raw_pixel_h / tgt_pixel_h)
        image = np.array(Image.fromarray(image).resize((rescaled_w, rescaled_h), Image.Resampling.LANCZOS))

        depth, depth_mask = utils3d.np.masked_nearest_resize(depth, mask=depth_mask, size=(rescaled_h, rescaled_w))
        distance = norm3d(utils3d.np.depth_map_to_point_map(depth, intrinsics=intrinsics))
        segmentation_mask = cv2.resize(segmentation_mask, (rescaled_w, rescaled_h), interpolation=cv2.INTER_NEAREST) if segmentation_mask is not None else None
        if normal is not None:
            # NOTE: `utils3d.np.masked_nearest_resize` takes size as (H, W) -- the
            # opposite of cv2's (W, H) and of `mask_aware_nearest_resize_numpy`.
            normal, normal_mask = utils3d.np.masked_nearest_resize(normal, mask=normal_mask, size=(rescaled_h, rescaled_w))

        # 4.2 calculate homography warping
        transform = intrinsics @ np.linalg.inv(R) @ np.linalg.inv(tgt_intrinsics)
        uv_tgt = utils3d.np.uv_map(tgt_height, tgt_width)
        pts = np.concatenate([uv_tgt, np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1) @ transform.T
        uv_remap = pts[:, :, :2] / (pts[:, :, 2:3] + 1e-12)
        pixel_remap = utils3d.np.uv_to_pixel(uv_remap, (rescaled_h, rescaled_w)).astype(np.float32)
        
        tgt_image = cv2.remap(image, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR)
        tgt_distance = cv2.remap(distance, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST)
        tgt_ray_length = utils3d.np.unproject_cv(uv_tgt, np.ones_like(uv_tgt[:, :, 0]), intrinsics=tgt_intrinsics)
        tgt_ray_length = (tgt_ray_length[:, :, 0] ** 2 + tgt_ray_length[:, :, 1] ** 2 + tgt_ray_length[:, :, 2] ** 2) ** 0.5
        tgt_depth = tgt_distance / (tgt_ray_length + 1e-12)
        tgt_depth_mask = cv2.remap(depth_mask.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
        tgt_segmentation_mask = cv2.remap(segmentation_mask, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) if segmentation_mask is not None else None
        tgt_normal = cv2.remap(normal, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR) @ R.T if normal is not None else None
        tgt_normal_mask = cv2.remap(normal_mask.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0 if normal_mask is not None else None

        # NOTE: cv2.resize / cv2.remap take size as (W, H), unlike the
        # `masked_nearest_resize` call above.
        local_mask = instance.get('local_mask', None)
        if local_mask is not None:
            local_mask = cv2.resize(local_mask.astype(np.uint8), (rescaled_w, rescaled_h), interpolation=cv2.INTER_NEAREST)
            tgt_local_mask = cv2.remap(local_mask, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
        else:
            tgt_local_mask = None

        local_segmentation = instance.get('local_segmentation', None)
        if local_segmentation is not None:
            # int32 round-trips through resize/remap under INTER_NEAREST.
            local_segmentation = cv2.resize(local_segmentation, (rescaled_w, rescaled_h), interpolation=cv2.INTER_NEAREST)
            tgt_local_segmentation = cv2.remap(local_segmentation, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST)
        else:
            tgt_local_segmentation = None

        # drop depth greater than drop_max_depth
        max_depth = np.nanquantile(np.where(tgt_depth_mask, tgt_depth, np.nan), 0.01) * self.drop_max_depth
        tgt_depth_mask &= tgt_depth <= max_depth
        tgt_depth = np.nan_to_num(tgt_depth, nan=0.0)

        if self.depth_unit is not None:
            tgt_depth *= self.depth_unit
        
        if not np.any(tgt_depth_mask):
            # always make sure that mask is not empty, otherwise the loss calculation will crash
            tgt_depth_mask = np.ones_like(tgt_depth_mask)
            tgt_depth = np.ones_like(tgt_depth)
            instance['label_type'] = 'invalid'
        
        tgt_pts = utils3d.np.unproject_cv(uv_tgt, tgt_depth, intrinsics=tgt_intrinsics)

        # Process segmentation labels
        if segmentation_mask is not None:
            for k in ['undefined', 'unannotated', 'background', 'sky']:
                if k in segmentation_labels:
                    del segmentation_labels[k]
            seg_id2count = dict(zip(*np.unique(tgt_segmentation_mask, return_counts=True)))
            sorted_labels = sorted(segmentation_labels.keys(), key=lambda x: seg_id2count.get(segmentation_labels[x], 0), reverse=True)
            segmentation_labels = {k: segmentation_labels[k] for k in sorted_labels[:self.max_segments] if seg_id2count.get(segmentation_labels[k], 0) >= self.min_seg_area}

        instance.update({
            'image': torch.from_numpy(tgt_image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'depth': torch.from_numpy(tgt_depth).float(),
            'depth_mask': torch.from_numpy(tgt_depth_mask).bool(),
            'intrinsics': torch.from_numpy(tgt_intrinsics).float(),
            'points': torch.from_numpy(tgt_pts).float(),
            'normal': torch.from_numpy(tgt_normal).float() if tgt_normal is not None else None,
            'normal_mask': torch.from_numpy(tgt_normal_mask).bool() if tgt_normal_mask is not None else None,
            'segmentation_mask': torch.from_numpy(tgt_segmentation_mask).long() if tgt_segmentation_mask is not None else None,
            'segmentation_labels': segmentation_labels,
            'local_mask': torch.from_numpy(tgt_local_mask).bool() if tgt_local_mask is not None else None,
            'local_segmentation': torch.from_numpy(tgt_local_segmentation).long() if tgt_local_segmentation is not None else None,
            'is_metric': self.depth_unit is not None,
            'has_sharp_boundary': self.has_sharp_boundary,
        })
        
        instance = {k: v for k, v in instance.items() if v is not None}
        
        return instance

    def start(self):
        self.pipeline.start()
    
    def stop(self):
        self.pipeline.stop()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get(self):
        return self.pipeline.get()