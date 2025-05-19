import os
from pathlib import Path
import json
import time
import random
from typing import *
import traceback
import itertools
from numbers import Number
import io

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.v2.functional as TF
import utils3d
from tqdm import tqdm

from ..utils import pipeline
from ..utils.io import *
from ..utils.geometry_numpy import mask_aware_nearest_resize_numpy, harmonic_mean_numpy, norm3d, depth_occlusion_edge_numpy, depth_of_field


class TrainDataLoaderPipeline:
    def __init__(self, config: dict, batch_size: int, num_load_workers: int = 4, num_process_workers: int = 8, buffer_size: int = 8):
        self.config = config

        self.batch_size = batch_size
        self.clamp_max_depth = config['clamp_max_depth']
        self.fov_range_absolute = config.get('fov_range_absolute', 0.0)
        self.fov_range_relative = config.get('fov_range_relative', 0.0)
        self.center_augmentation = config.get('center_augmentation', 0.0)
        self.image_augmentation = config.get('image_augmentation', [])
        self.depth_interpolation = config.get('depth_interpolation', 'bilinear')

        if 'image_sizes' in config:
            self.image_size_strategy = 'fixed'
            self.image_sizes = config['image_sizes']
        elif 'aspect_ratio_range' in config and 'area_range' in config:
            self.image_size_strategy = 'aspect_area'
            self.aspect_ratio_range = config['aspect_ratio_range']
            self.area_range = config['area_range']
        else:
            raise ValueError('Invalid image size configuration')

        # Load datasets
        self.datasets = {}
        for dataset in tqdm(config['datasets'], desc='Loading datasets'):
            name = dataset['name']
            content = Path(dataset['path'], dataset.get('index', '.index.txt')).joinpath().read_text()
            filenames = content.splitlines()
            self.datasets[name] = {
                **dataset,
                'path': dataset['path'],
                'filenames': filenames,
            }
        self.dataset_names = [dataset['name'] for dataset in config['datasets']]
        self.dataset_weights = [dataset['weight'] for dataset in config['datasets']]

        # Build pipeline
        self.pipeline = pipeline.Sequential([
            self._sample_batch,
            pipeline.Unbatch(),
            pipeline.Parallel([self._load_instance] * num_load_workers),
            pipeline.Parallel([self._process_instance] * num_process_workers),
            pipeline.Batch(self.batch_size),
            self._collate_batch,
            pipeline.Buffer(buffer_size),
        ])

        self.invalid_instance = {
            'intrinsics': np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
            'image': np.zeros((256, 256, 3), dtype=np.uint8),
            'depth': np.ones((256, 256), dtype=np.float32),
            'depth_mask': np.ones((256, 256), dtype=bool),
            'depth_mask_inf': np.zeros((256, 256), dtype=bool),
            'label_type': 'invalid',
        }

    def _sample_batch(self):
        batch_id = 0
        last_area = None
        while True:
            # Depending on the sample strategy, choose a dataset and a filename
            batch_id += 1
            batch = []
            
            # Sample instances
            for _ in range(self.batch_size):
                dataset_name = random.choices(self.dataset_names, weights=self.dataset_weights)[0]
                filename = random.choice(self.datasets[dataset_name]['filenames'])

                path = Path(self.datasets[dataset_name]['path'], filename)

                instance = {
                    'batch_id': batch_id,
                    'seed': random.randint(0, 2 ** 32 - 1),
                    'dataset': dataset_name,
                    'filename': filename,
                    'path': path,
                    'label_type': self.datasets[dataset_name]['label_type'],
                }
                batch.append(instance)

            # Decide the image size for this batch
            if self.image_size_strategy == 'fixed':
                width, height = random.choice(self.config['image_sizes'])
            elif self.image_size_strategy == 'aspect_area':
                area = random.uniform(*self.area_range)
                aspect_ratio_ranges = [self.datasets[instance['dataset']].get('aspect_ratio_range', self.aspect_ratio_range) for instance in batch]
                aspect_ratio_range = (min(r[0] for r in aspect_ratio_ranges), max(r[1] for r in aspect_ratio_ranges))
                aspect_ratio = random.uniform(*aspect_ratio_range)
                width, height = int((area * aspect_ratio) ** 0.5), int((area / aspect_ratio) ** 0.5)
            else:
                raise ValueError('Invalid image size strategy')
            
            for instance in batch:
                instance['width'], instance['height'] = width, height
            
            yield batch

    def _load_instance(self, instance: dict):
        try:
            image = read_image(Path(instance['path'], 'image.jpg'))
            depth, _ = read_depth(Path(instance['path'], self.datasets[instance['dataset']].get('depth', 'depth.png')))
            
            meta = read_meta(Path(instance['path'], 'meta.json'))
            intrinsics = np.array(meta['intrinsics'], dtype=np.float32)
            depth_mask = np.isfinite(depth)
            depth_mask_inf = np.isinf(depth)
            depth = np.nan_to_num(depth, nan=1, posinf=1, neginf=1)
            data = {
                'image': image,
                'depth': depth,
                'depth_mask': depth_mask,
                'depth_mask_inf': depth_mask_inf,
                'intrinsics': intrinsics
            }
            instance.update({
                **data,
            })
        except Exception as e:
            print(f"Failed to load instance {instance['dataset']}/{instance['filename']} because of exception:", e)
            instance.update(self.invalid_instance)
        return instance

    def _process_instance(self, instance: Dict[str, Union[np.ndarray, str, float, bool]]):
        image, depth, depth_mask, depth_mask_inf, intrinsics, label_type = instance['image'], instance['depth'], instance['depth_mask'], instance['depth_mask_inf'], instance['intrinsics'], instance['label_type']
        depth_unit = self.datasets[instance['dataset']].get('depth_unit', None)

        raw_height, raw_width = image.shape[:2]
        raw_horizontal, raw_vertical = abs(1.0 / intrinsics[0, 0]), abs(1.0 / intrinsics[1, 1])
        raw_fov_x, raw_fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
        raw_pixel_w, raw_pixel_h = raw_horizontal / raw_width, raw_vertical / raw_height
        tgt_width, tgt_height = instance['width'], instance['height']
        tgt_aspect = tgt_width / tgt_height
        
        rng = np.random.default_rng(instance['seed'])

        # 1. set target fov
        center_augmentation = self.datasets[instance['dataset']].get('center_augmentation', self.center_augmentation)
        fov_range_absolute_min, fov_range_absolute_max = self.datasets[instance['dataset']].get('fov_range_absolute', self.fov_range_absolute)
        fov_range_relative_min, fov_range_relative_max = self.datasets[instance['dataset']].get('fov_range_relative', self.fov_range_relative)
        tgt_fov_x_min = min(fov_range_relative_min * raw_fov_x, fov_range_relative_min * utils3d.focal_to_fov(utils3d.fov_to_focal(raw_fov_y) / tgt_aspect))
        tgt_fov_x_max = min(fov_range_relative_max * raw_fov_x, fov_range_relative_max * utils3d.focal_to_fov(utils3d.fov_to_focal(raw_fov_y) / tgt_aspect))
        tgt_fov_x_min, tgt_fov_x_max = max(np.deg2rad(fov_range_absolute_min), tgt_fov_x_min), min(np.deg2rad(fov_range_absolute_max), tgt_fov_x_max)
        tgt_fov_x = rng.uniform(min(tgt_fov_x_min, tgt_fov_x_max), tgt_fov_x_max)
        tgt_fov_y = utils3d.focal_to_fov(utils3d.numpy.fov_to_focal(tgt_fov_x) * tgt_aspect)

        # 2. set target image center (principal point) and the corresponding z-direction in raw camera space
        center_dtheta = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_x - tgt_fov_x)
        center_dphi = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_y - tgt_fov_y)
        cu, cv = 0.5 + 0.5 * np.tan(center_dtheta) / np.tan(raw_fov_x / 2), 0.5 + 0.5 *  np.tan(center_dphi) / np.tan(raw_fov_y / 2)
        direction = utils3d.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=intrinsics)[0]

        # 3. obtain the rotation matrix for homography warping
        R = utils3d.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

        # 4. shrink the target view to fit into the warped image
        corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
        corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(intrinsics).T @ R.T)   # corners in viewport's camera plane
        corners = corners[:, :2] / corners[:, 2:3]
        tgt_horizontal, tgt_vertical = np.tan(tgt_fov_x / 2) * 2, np.tan(tgt_fov_y / 2) * 2
        warp_horizontal, warp_vertical = float('inf'), float('inf')
        for i in range(4):
            intersection, _ = utils3d.numpy.ray_intersection(
                np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
                corners[i - 1], corners[i] - corners[i - 1],
            )
            warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
        tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)
        
        # 5. obtain the target intrinsics
        fx, fy = 1 / tgt_horizontal, 1 / tgt_vertical
        tgt_intrinsics = utils3d.numpy.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)

        # 6. do homogeneous transformation 
        # 6.1 The image and depth are resized first to approximately the same pixel size as the target image with PIL's antialiasing resampling
        tgt_pixel_w, tgt_pixel_h = tgt_horizontal / tgt_width, tgt_vertical / tgt_height        # (should be exactly the same for x and y axes)
        rescaled_w, rescaled_h = int(raw_width * raw_pixel_w / tgt_pixel_w), int(raw_height * raw_pixel_h / tgt_pixel_h)
        image = np.array(Image.fromarray(image).resize((rescaled_w, rescaled_h), Image.Resampling.LANCZOS))

        fg_edge_mask, bg_edge_mask = depth_occlusion_edge_numpy(depth, mask=depth_mask, kernel_size=5, tol=0.01)
        edge_mask = fg_edge_mask | bg_edge_mask
        _, depth_mask_nearest, resize_index = mask_aware_nearest_resize_numpy(None, depth_mask, (rescaled_w, rescaled_h), return_index=True)
        depth_nearest = depth[resize_index]
        distance_nearest = norm3d(utils3d.numpy.depth_to_points(depth_nearest, intrinsics=intrinsics))
        edge_mask = edge_mask[resize_index]

        if self.depth_interpolation == 'bilinear':
            depth_mask_bilinear = cv2.resize(depth_mask.astype(np.float32), (rescaled_w, rescaled_h), interpolation=cv2.INTER_LINEAR)
            depth_bilinear = 1 / cv2.resize(1 / depth, (rescaled_w, rescaled_h), interpolation=cv2.INTER_LINEAR)
            distance_bilinear = norm3d(utils3d.numpy.depth_to_points(depth_bilinear, intrinsics=intrinsics))

        depth_mask_inf = cv2.resize(depth_mask_inf.astype(np.uint8), (rescaled_w, rescaled_h), interpolation=cv2.INTER_NEAREST) > 0

        # 6.2 calculate homography warping
        transform = intrinsics @ np.linalg.inv(R) @ np.linalg.inv(tgt_intrinsics)
        uv_tgt = utils3d.numpy.image_uv(width=tgt_width, height=tgt_height)
        pts = np.concatenate([uv_tgt, np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1) @ transform.T
        uv_remap = pts[:, :, :2] / (pts[:, :, 2:3] + 1e-12)
        pixel_remap = utils3d.numpy.uv_to_pixel(uv_remap, width=rescaled_w, height=rescaled_h).astype(np.float32)
        
        tgt_image = cv2.remap(image, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LANCZOS4)
        tgt_ray_length = norm3d(utils3d.numpy.unproject_cv(uv_tgt, np.ones_like(uv_tgt[:, :, 0]), intrinsics=tgt_intrinsics))
        tgt_depth_mask_nearest = cv2.remap(depth_mask_nearest.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
        tgt_depth_nearest = cv2.remap(distance_nearest, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) / tgt_ray_length
        tgt_edge_mask = cv2.remap(edge_mask.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
        if self.depth_interpolation == 'bilinear':
            tgt_depth_mask_bilinear = cv2.remap(depth_mask_bilinear, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR)
            tgt_depth_bilinear = cv2.remap(distance_bilinear, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR) / tgt_ray_length
            tgt_depth = np.where((tgt_depth_mask_bilinear == 1) & ~tgt_edge_mask, tgt_depth_bilinear, tgt_depth_nearest)
        else:
            tgt_depth = tgt_depth_nearest
        tgt_depth_mask = tgt_depth_mask_nearest
        
        tgt_depth_mask_inf = cv2.remap(depth_mask_inf.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0

        # always make sure that mask is not empty
        if tgt_depth_mask.sum() / tgt_depth_mask.size < 0.001:
            tgt_depth_mask = np.ones_like(tgt_depth_mask)
            tgt_depth = np.ones_like(tgt_depth)
            instance['label_type'] = 'invalid'

        # Flip augmentation
        if rng.choice([True, False]):
            tgt_image = np.flip(tgt_image, axis=1).copy()
            tgt_depth = np.flip(tgt_depth, axis=1).copy()
            tgt_depth_mask = np.flip(tgt_depth_mask, axis=1).copy()
            tgt_depth_mask_inf = np.flip(tgt_depth_mask_inf, axis=1).copy()
        
        # Color augmentation
        image_augmentation = self.datasets[instance['dataset']].get('image_augmentation', self.image_augmentation)
        if 'jittering' in image_augmentation:
            tgt_image = torch.from_numpy(tgt_image).permute(2, 0, 1)
            tgt_image = TF.adjust_brightness(tgt_image, rng.uniform(0.7, 1.3))
            tgt_image = TF.adjust_contrast(tgt_image, rng.uniform(0.7, 1.3))
            tgt_image = TF.adjust_saturation(tgt_image, rng.uniform(0.7, 1.3))
            tgt_image = TF.adjust_hue(tgt_image, rng.uniform(-0.1, 0.1))
            tgt_image = TF.adjust_gamma(tgt_image, rng.uniform(0.7, 1.3))
            tgt_image = tgt_image.permute(1, 2, 0).numpy()
        if 'dof' in image_augmentation:
            if rng.uniform() < 0.5:
                dof_strength = rng.integers(12)
                tgt_disp = np.where(tgt_depth_mask_inf, 0, 1 / tgt_depth)
                disp_min, disp_max = tgt_disp[tgt_depth_mask].min(), tgt_disp[tgt_depth_mask].max()
                tgt_disp = cv2.inpaint(tgt_disp, (~tgt_depth_mask & ~tgt_depth_mask_inf).astype(np.uint8), 3, cv2.INPAINT_TELEA).clip(disp_min, disp_max)
                dof_focus = rng.uniform(disp_min, disp_max)
                tgt_image = depth_of_field(tgt_image, tgt_disp, dof_focus, dof_strength)
        if 'shot_noise' in image_augmentation:
            if rng.uniform() < 0.5: 
                k = np.exp(rng.uniform(np.log(100), np.log(10000))) / 255
                tgt_image = (rng.poisson(tgt_image * k) / k).clip(0, 255).astype(np.uint8)
        if 'jpeg_loss' in image_augmentation:
            if rng.uniform() < 0.5: 
                tgt_image = cv2.imdecode(cv2.imencode('.jpg', tgt_image, [cv2.IMWRITE_JPEG_QUALITY, rng.integers(20, 100)])[1], cv2.IMREAD_COLOR)
        if 'blurring' in image_augmentation:
            if rng.uniform() < 0.5:    
                ratio = rng.uniform(0.25, 1)
                tgt_image = cv2.resize(cv2.resize(tgt_image, (int(tgt_width * ratio), int(tgt_height * ratio)), interpolation=cv2.INTER_AREA), (tgt_width, tgt_height), interpolation=rng.choice([cv2.INTER_LINEAR_EXACT, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))

        # convert depth to metric if necessary
        if depth_unit is not None:
            tgt_depth *= depth_unit
            instance['is_metric'] = True
        else:
            instance['is_metric'] = False

        # clamp depth maximum values
        max_depth = np.nanquantile(np.where(tgt_depth_mask, tgt_depth, np.nan), 0.01) * self.clamp_max_depth
        tgt_depth = np.clip(tgt_depth, 0, max_depth)
        tgt_depth = np.nan_to_num(tgt_depth, nan=1.0)

        if self.datasets[instance['dataset']].get('finite_depth_mask', None) == "only_known":
            tgt_depth_mask_fin = tgt_depth_mask
        else:
            tgt_depth_mask_fin = ~tgt_depth_mask_inf

        instance.update({
            'image': torch.from_numpy(tgt_image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'depth': torch.from_numpy(tgt_depth).float(),
            'depth_mask': torch.from_numpy(tgt_depth_mask).bool(),
            'depth_mask_fin': torch.from_numpy(tgt_depth_mask_fin).bool(),
            'depth_mask_inf': torch.from_numpy(tgt_depth_mask_inf).bool(),
            'intrinsics': torch.from_numpy(tgt_intrinsics).float(),
        })
        
        return instance

    def _collate_batch(self, instances: List[Dict[str, Any]]):
        batch = {k: torch.stack([instance[k] for instance in instances], dim=0) for k in ['image', 'depth', 'depth_mask', 'depth_mask_fin', 'depth_mask_inf', 'intrinsics']}
        batch = {
            'label_type': [instance['label_type'] for instance in instances],
            'is_metric': [instance['is_metric'] for instance in instances],
            'info': [{'dataset': instance['dataset'], 'filename': instance['filename']} for instance in instances],
            **batch,
        }
        return batch
    
    def get(self) -> Dict[str, Union[torch.Tensor, str]]:
        return self.pipeline.get()

    def start(self):
        self.pipeline.start()

    def stop(self):
        self.pipeline.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pipeline.terminate()
        self.pipeline.join()
        return False


