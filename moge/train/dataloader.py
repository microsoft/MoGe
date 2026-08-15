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
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d
import pipeline
from tqdm import tqdm


from ..utils.io import *
from ..utils.geometry_numpy import mask_aware_nearest_resize_numpy, harmonic_mean_numpy, norm3d, depth_occlusion_edge_numpy
from ..utils.data_augmentation import sample_perspective, warp_perspective, image_color_augmentation
from ..utils.tools import catch_exception


_NO_BUCKET = object()


class _SourceBatchGroup(pipeline.Batch):
    """Regroup surviving instances back into the source batch they came from.

    ``_sample_batch`` emits fixed-size batches whose instances share one target
    ``(height, width)``; the upstream ``Filter`` may then drop some of them as
    invalid. Since the ``Parallel`` nodes in between are FIFO, every instance of
    a source batch arrives contiguously, so only one bucket is ever open: a new
    ``batch_id`` means the previous source batch is complete.

    Incomplete buckets are dropped. That matches what the trainer effectively
    saw before this was keyed by source batch -- a short bucket was keyed by
    ``(height, width)``, which is derived from two continuous random draws and
    so practically never recurred, meaning the bucket could never be completed
    and was retained forever. The behaviour is the same; the unbounded
    retention is not. To train on the survivors instead of dropping them, put
    the bucket to the output in ``_discard_incomplete`` below.
    """
    def __init__(self, batch_size: int, name: Optional[str] = None):
        super().__init__(batch_size=batch_size, patience=None, name=name)
        self.num_incomplete_batches = 0
        self.num_dropped_instances = 0

    def _discard_incomplete(self, bucket: list):
        if bucket:
            self.num_incomplete_batches += 1
            self.num_dropped_instances += len(bucket)

    def loop(self):
        from pipeline.components import EndOfInput, ExceptionInNode
        from pipeline.queue import ShutDown
        bucket: list = []
        bucket_id = _NO_BUCKET
        try:
            while True:
                item = self.input.get()
                if isinstance(item, (EndOfInput, ExceptionInNode)):
                    # A full bucket is emitted as soon as it fills, so anything
                    # still open here is necessarily incomplete.
                    self._discard_incomplete(bucket)
                    bucket, bucket_id = [], _NO_BUCKET
                    self.output.put(item)
                    continue
                item_id = item.get('batch_id', _NO_BUCKET)
                if item_id != bucket_id:
                    self._discard_incomplete(bucket)
                    bucket, bucket_id = [], item_id
                bucket.append(item)
                if len(bucket) >= self.batch_size:
                    self.output.put(bucket)
                    bucket, bucket_id = [], _NO_BUCKET
        except ShutDown:
            return

    def _default_name(self):
        return f"SourceBatchGroup(size={self.batch_size})"


class TrainDataLoaderPipeline:
    def __init__(self, config: dict, batch_size: int, buffer_size: int = 8, workspace: Path = None, seed: Optional[int] = None):
        self.config = config
        self.workspace = workspace
        self._rng = random.Random(seed)

        num_load_workers = int(os.environ.get('MOGE_NUM_LOAD_WORKERS', 4))
        num_process_workers = int(os.environ.get('MOGE_NUM_PROCESS_WORKERS', 8))

        self.batch_size = batch_size
        self.clamp_max_depth = config['clamp_max_depth']
        self.fov_range_absolute = config.get('fov_range_absolute', 0.0)
        self.fov_range_relative = config.get('fov_range_relative', 0.0)
        self.center_augmentation = config.get('center_augmentation', 0.0)
        self.image_augmentation = config.get('image_augmentation', [])
        self.depth_interpolation = config.get('depth_interpolation', 'bilinear')
        # Reject-and-resample: if the warped finite-depth ratio (probed at low
        # resolution) falls below ``min_valid_after_warp``, we resample the
        # perspective up to ``resample_max_retries`` extra times and keep the
        # best attempt. Per-dataset overrides are honored via the dataset cfg.
        self.min_valid_after_warp = config.get('min_valid_after_warp', 0.01)
        self.resample_max_retries = config.get('resample_max_retries', 8)

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

        # debug sample balance
        self.dataset_sample_count = {name: 0 for name in self.dataset_names}

        # Build pipeline
        # Regroup by source batch so that ``Filter`` dropping invalid
        # instances does not lead to mixed-size batches.
        self._batch_grouper = _SourceBatchGroup(self.batch_size)
        self.pipeline = pipeline.Sequential([
            self._sample_batch,
            pipeline.Unbatch(),
            pipeline.Parallel([self._load_instance] * num_load_workers),
            pipeline.Parallel([self._process_instance] * num_process_workers),
            pipeline.Filter(lambda instance: instance['label_type'] != 'invalid'),
            self._batch_grouper,
            self._collate_batch,
            pipeline.Buffer(buffer_size),
        ])

    def state_dict(self) -> dict:
        """Return the RNG state so training can resume with the same data order."""
        return {'rng_state': self._rng.getstate()}

    def load_state_dict(self, state_dict: dict):
        """Restore the RNG state from a previous ``state_dict()`` call."""
        self._rng.setstate(state_dict['rng_state'])

    def _get_invalid_instance(self, torch_instance: bool = False, height: int = 256, width: int = 256, gen_info: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor, str, float, bool]]:
        invalid_depth = np.ones((height, width), dtype=np.float32)
        invalid_instance = {
            'intrinsics': np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
            'image': np.zeros((3, height, width), dtype=np.float32),
            'normal': np.zeros((height, width, 3), dtype=np.float32),
            'depth': invalid_depth,
            'depth_mask_fin': np.isfinite(invalid_depth),
            'depth_mask_inf': np.isinf(invalid_depth),
            'label_type': 'invalid',
            'is_metric': False,
        }
        if gen_info:
            invalid_instance.update({
                'width': width,
                'height': height,
                'dataset': 'invalid',
                'filename': 'invalid',
                'path': 'invalid',
            })
        if torch_instance:
            return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in invalid_instance.items()}
        else:
            return invalid_instance

    def _sample_batch(self):
        rng = self._rng
        batch_id = 0
        last_area = None
        while True:
            # Depending on the sample strategy, choose a dataset and a filename
            batch_id += 1
            batch = []
            
            # Sample instances
            for _ in range(self.batch_size):
                dataset_name = rng.choices(self.dataset_names, weights=self.dataset_weights)[0]
                filename = rng.choice(self.datasets[dataset_name]['filenames'])

                self.dataset_sample_count[dataset_name] += 1

                path = Path(self.datasets[dataset_name]['path'], filename)

                instance = {
                    'batch_id': batch_id,
                    'seed': rng.randint(0, 2 ** 32 - 1),
                    'dataset': dataset_name,
                    'filename': filename,
                    'path': path,
                    'label_type': self.datasets[dataset_name]['label_type'],
                }
                batch.append(instance)

            # Decide the image size for this batch
            if self.image_size_strategy == 'fixed':
                width, height = rng.choice(self.config['image_sizes'])
            elif self.image_size_strategy == 'aspect_area':
                area = rng.uniform(*self.area_range)
                aspect_ratio_ranges = [self.datasets[instance['dataset']].get('aspect_ratio_range', self.aspect_ratio_range) for instance in batch]
                aspect_ratio_range = (min(r[0] for r in aspect_ratio_ranges), max(r[1] for r in aspect_ratio_ranges))
                aspect_ratio = rng.uniform(*aspect_ratio_range)
                width, height = int((area * aspect_ratio) ** 0.5), int((area / aspect_ratio) ** 0.5)
            else:
                raise ValueError('Invalid image size strategy')
            
            for instance in batch:
                instance['width'], instance['height'] = width, height
            
            yield batch

    def _load_instance(self, instance: dict):
        try:
            if instance is None:
                return None

            image = read_image(Path(instance['path'], 'image.jpg'))
            depth = read_depth(Path(instance['path'], self.datasets[instance['dataset']].get('depth', 'depth.png')))
            meta = read_json(Path(instance['path'], 'meta.json'))
            intrinsics = np.array(meta['intrinsics'], dtype=np.float32)

            has_metric_annotation = False
            if "metric_scale" in meta:
                depth *= meta["metric_scale"]
                has_metric_annotation = True
            elif "depth_scale" in meta:
                depth *= meta["depth_scale"]
                has_metric_annotation = True

            data = {
                'image': image,
                'depth': depth,
                'intrinsics': intrinsics,
                'path': str(instance['path']),
                'has_metric_annotation': has_metric_annotation,
            }
            instance.update({
                **data,
            })
            
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to load instance {instance['dataset']}/{instance['filename']} because of exception:", e)
            instance.update(self._get_invalid_instance())
        return instance

    def _process_instance(self, instance: Dict[str, Union[np.ndarray, str, float, bool]]):
        try:
            if instance is None:
                return self._get_invalid_instance(torch_instance=True, gen_info=True)

            if instance['label_type'] == 'invalid':
                instance.update(self._get_invalid_instance(torch_instance=True))
                return instance

            raw_image, raw_depth, raw_intrinsics = instance['image'], instance['depth'], instance['intrinsics']
            raw_normal, raw_normal_mask = utils3d.np.depth_map_to_normal_map(raw_depth, intrinsics=raw_intrinsics, mask=np.isfinite(raw_depth), edge_threshold=88)
            raw_normal = np.where(raw_normal_mask[..., None], raw_normal, np.nan)
            depth_unit = self.datasets[instance['dataset']].get('depth_unit', None)

            tgt_width, tgt_height = instance['width'], instance['height']
            tgt_aspect = tgt_width / tgt_height
            
            rng = np.random.default_rng(instance['seed'])

            # Sample perspective transformation with reject-and-resample on the
            # coarse warped finite-depth ratio. This avoids spending the
            # expensive full-res warp budget on viewpoints that land entirely
            # in NaN regions (sky / empty background).
            ds_cfg = self.datasets[instance['dataset']]
            min_valid_after_warp = ds_cfg.get('min_valid_after_warp', self.min_valid_after_warp)
            max_resample_retries = ds_cfg.get('resample_max_retries', self.resample_max_retries)
            raw_finite_mask = np.isfinite(raw_depth)
            probe_h = 64
            probe_w = max(1, int(round(probe_h * tgt_aspect)))

            best = {'ratio': -1.0, 'tgt_intrinsics': None, 'R': None, 'transform': None}
            for _attempt in range(max_resample_retries + 1):
                tgt_intrinsics_try, R_try = sample_perspective(
                    raw_intrinsics,
                    tgt_aspect=tgt_aspect,
                    center_augmentation=ds_cfg.get('center_augmentation', self.center_augmentation),
                    fov_range_absolute=ds_cfg.get('fov_range_absolute', self.fov_range_absolute),
                    fov_range_relative=ds_cfg.get('fov_range_relative', self.fov_range_relative),
                    rng=rng,
                )
                transform_try = tgt_intrinsics_try @ R_try @ np.linalg.inv(raw_intrinsics)
                probe = warp_perspective(
                    raw_finite_mask.astype(np.uint8), transform_try,
                    (probe_h, probe_w), interpolation='nearest',
                )
                ratio = float(probe.mean())
                if ratio > best['ratio']:
                    best.update(ratio=ratio, tgt_intrinsics=tgt_intrinsics_try, R=R_try, transform=transform_try)
                if ratio >= min_valid_after_warp:
                    break
            tgt_intrinsics, R, transform = best['tgt_intrinsics'], best['R'], best['transform']

            # Warp
            # - Warp image
            tgt_image = warp_perspective(raw_image, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
            # - Warp depth
            depth_edge_mask = utils3d.np.depth_map_edge(raw_depth, mask=np.isfinite(raw_depth), kernel_size=5, ltol=0.01)
            depth_bilinear_mask = np.isfinite(raw_depth) & ~depth_edge_mask
            warped_depth_bilinear_mask = warp_perspective(depth_bilinear_mask.astype(np.float32), transform, (tgt_height, tgt_width), interpolation='bilinear')
            warped_depth_nearest = warp_perspective(raw_depth, transform, (tgt_height, tgt_width), interpolation='nearest', sparse_mask=~np.isnan(raw_depth))
            warped_depth_bilinear = 1 / warp_perspective(1 / raw_depth, transform, (tgt_height, tgt_width), interpolation='bilinear')   # NOTE: Bilinear intepolation in disparity space maintains planar surfaces.
            warped_depth = np.where(warped_depth_bilinear_mask == 1., warped_depth_bilinear, warped_depth_nearest)
            # check if there is any zero in warped depth
            if np.any(warped_depth == 0):
                try:
                    print(f"Zero depth encountered for instance {instance['path']}. Dumping data.")
                    dump_path = self.workspace / 'failed_warp_dumps' / f"{instance['dataset']}_{instance['filename'].replace('/', '_')}.npz"
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dump_path, raw_image=raw_image, raw_depth=raw_depth, transform=transform, tgt_size=(tgt_height, tgt_width), warped_depth=warped_depth)
                except Exception as e:
                    print("Failed to dump warp data because of:", e)
                # fix zeros
                warped_depth = np.where(warped_depth > 0, warped_depth, np.nan)
            tgt_uvhomo = np.concatenate([utils3d.np.uv_map((tgt_height, tgt_width)), np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1)
            tgt_depth = warped_depth / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])
            # - Warp normal
            warped_normal = warp_perspective(raw_normal, transform, (tgt_height, tgt_width), interpolation='bilinear')
            tgt_normal = warped_normal @ R.T

            # always make sure that mask is not empty
            if np.isfinite(tgt_depth).sum() / tgt_depth.size < 0.001:
                tgt_depth = np.ones_like(tgt_depth)
                instance['label_type'] = 'invalid'
                # try:
                    # print(f"Insufficient valid depth after warping for instance {instance['path']}.")
                    # dump_path = self.workspace / 'insufficient_depths' / f"{instance['dataset']}_{instance['filename'].replace('/', '_')}.npz"
                    # dump_path.parent.mkdir(parents=True, exist_ok=True)
                    # np.savez_compressed(dump_path, raw_image=raw_image, raw_depth=raw_depth, transform=transform, tgt_size=(tgt_height, tgt_width), tgt_depth=tgt_depth, tgt_image=tgt_image)
                # except Exception as e:
                    # print("Failed to dump insufficient depth data because of:", e)

            # Flip augmentation
            if rng.choice([True, False]):
                tgt_image = np.flip(tgt_image, axis=1).copy()
                tgt_depth = np.flip(tgt_depth, axis=1).copy()
                tgt_normal = np.flip(tgt_normal, axis=1).copy() * [-1, 1, 1]
                # NOTE: if cx != 0.5, flip intrinsics accordingly. 
            
            # Color augmentation
            image_augmentation = self.datasets[instance['dataset']].get('image_augmentation', self.image_augmentation)
            tgt_image = image_color_augmentation(
                tgt_image, 
                augmentations=image_augmentation, 
                rng=rng, 
                depth=tgt_depth,
            )
            
            # Set metric flag.
            # - Default: a dataset is metric iff it declares a `depth_unit` (depth is in a known
            #   metric unit), and `depth_unit` scales the raw depth to meters.
            # - `metric_from_meta`: decide per-sample — only instances whose meta.json carried a
            #   `metric_scale`/`depth_scale` annotation (already applied in `_load_instance`) are
            #   metric; the rest are relative-scale (e.g. MegaDepth SfM samples without a recovered
            #   scale). `depth_unit`, if present, still applies as a unit conversion to all samples.
            if depth_unit is not None:
                tgt_depth *= depth_unit
            if self.datasets[instance['dataset']].get('metric_from_meta', False):
                instance['is_metric'] = bool(instance.get('has_metric_annotation', False))
            else:
                instance['is_metric'] = depth_unit is not None

            # Clip maximum depth
            max_depth = np.nanquantile(np.where(np.isfinite(tgt_depth), tgt_depth, np.nan), 0.01) * self.clamp_max_depth
            tgt_depth = np.where(np.isfinite(tgt_depth), np.clip(tgt_depth, 0, max_depth), tgt_depth)

            tgt_depth_mask_inf = np.isinf(tgt_depth)
            if self.datasets[instance['dataset']].get('finite_depth_mask', None) == "only_known":
                tgt_depth_mask_fin = np.isfinite(tgt_depth)
            else:
                tgt_depth_mask_fin = ~tgt_depth_mask_inf

            instance.update({
                'image': torch.from_numpy(tgt_image.astype(np.float32) / 255.0).permute(2, 0, 1),
                'depth': torch.from_numpy(tgt_depth).float(),
                'depth_mask_fin': torch.from_numpy(tgt_depth_mask_fin).bool(),
                'depth_mask_inf': torch.from_numpy(tgt_depth_mask_inf).bool(),
                "normal": torch.from_numpy(tgt_normal).float(),
                'intrinsics': torch.from_numpy(tgt_intrinsics).float(),
            })
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to process instance {instance['path']}: {e}")
            instance.update(self._get_invalid_instance(torch_instance=True))

        return instance

    def _collate_batch(self, instances: List[Dict[str, Any]]):
        try:
            inst_w, inst_h = 256, 256
            for instance in instances:
                if instance['label_type'] != 'invalid':
                    inst_w, inst_h = instance['width'], instance['height']
                    break

            for i in range(len(instances)):
                if instances[i]['label_type'] == 'invalid':
                    print(f"Replacing invalid instance at index {i} with size ({inst_w}, {inst_h}). Instance: dataset={instances[i].get('dataset', 'N/A')}, filename={instances[i].get('filename', 'N/A')}")
                    instances[i].update(self._get_invalid_instance(torch_instance=True, height=inst_h, width=inst_w))

            batch = {k: torch.stack([instance[k] for instance in instances], dim=0) for k in ['image', 'depth', 'depth_mask_fin', 'depth_mask_inf', 'normal', 'intrinsics']}
            batch = {
                'label_type': [instance['label_type'] for instance in instances],
                'is_metric': [instance['is_metric'] for instance in instances],
                'info': [{'dataset': instance['dataset'], 'filename': instance['filename']} for instance in instances],
                **batch,
            }
            return batch
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to collate batch: {e}")
            # show batch info and dump instances
            try:
                for instance in instances:
                    print(f" - Instance: dataset={instance.get('dataset', 'N/A')}, filename={instance.get('filename', 'N/A')}, label_type={instance.get('label_type', 'N/A')}")
                    # save
                    dump_path = Path('failed_instance_dumps', f"{instance.get('dataset', 'N/A')}_{instance.get('filename', 'N/A').replace('/', '_')}.npz")
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dump_path, **{k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in instance.items()})
            except Exception as e2:
                traceback.print_exc()
                print("Also failed to dump instances because of:", e2)
            # return batch of invalid instances
            batch_size = len(instances)
            batch = {k: torch.stack([self._get_invalid_instance(torch_instance=True, height=inst_h, width=inst_w, gen_info=True)[k] for _ in range(batch_size)], dim=0) for k in ['image', 'depth', 'depth_mask_fin', 'depth_mask_inf', 'normal', 'intrinsics']}
            batch = {
                'label_type': ['invalid'] * batch_size,
                'is_metric': [False] * batch_size,
                'info': [{'dataset': 'invalid', 'filename': 'invalid'} for _ in range(batch_size)],
                **batch,
            }
            return batch
    
    def get(self) -> Dict[str, Union[torch.Tensor, str]]:
        return self.pipeline.get()

    def profile(self) -> str:
        text = self.pipeline.profile()
        grouper = getattr(self, '_batch_grouper', None)
        if grouper is not None and grouper.num_incomplete_batches > 0:
            text += (
                f'\nDropped {grouper.num_incomplete_batches} incomplete batches '
                f'({grouper.num_dropped_instances} instances) whose siblings were invalid'
            )
        return text

    def start(self):
        self.pipeline.start()

    def stop(self):
        self.pipeline.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pipeline.stop()
        return False

    def get_dataset_sample_count(self) -> Dict[str, int]:
        return self.dataset_sample_count
