import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from typing import IO
import zipfile
import json
import io
from typing import *
from pathlib import Path
import re

import numpy as np
import cv2 

from .tools import timeit


LEGACY_SEGFORMER_CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
    'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
    'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
    'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
    'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
    'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
    'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
    'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
    'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
    'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
    'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag'
]
LEGACY_SEGFORMER_LABELS = {k: i for i, k in enumerate(LEGACY_SEGFORMER_CLASSES)}


def write_rgbd_zip(
    file: Union[IO, os.PathLike], 
    image: Union[np.ndarray, bytes], 
    depth: Union[np.ndarray, bytes], mask: Union[np.ndarray, bytes], 
    segmentation_mask: Union[np.ndarray, bytes] = None, segmentation_labels: Union[Dict[str, int], bytes] = None, 
    intrinsics: np.ndarray = None, 
    normal: np.ndarray = None, normal_mask: np.ndarray = None,
    meta: Union[Dict[str, Any], bytes] = None, 
    *, image_quality: int = 95, depth_type: Literal['linear', 'log', 'disparity'] = 'linear', depth_format: Literal['png', 'exr'] = 'png', depth_max_dynamic_range: float = 1e4, png_compression: int = 7
):
    """
    Write RGBD data as zip archive containing the image, depth, mask, segmentation_mask, and meta data.
    In the zip file there will be:
    - `meta.json`: The meta data as a JSON file.
    - `image.jpg`: The RGB image as a JPEG file.
    - `depth.png/exr`: The depth map as a PNG or EXR file, depending on the `depth_type`.
    - `mask.png` (optional): The mask as a uint8 PNG file.
    - `segmentation_mask.png` (optional): The segformer mask as a uint8/uint16 PNG file.

    You can provided those data as np.ndarray or bytes. If you provide them as np.ndarray, they will be properly processed and encoded.
    If you provide them as bytes, they will be written as is, assuming they are already encoded.
    """
    if meta is None:
        meta = {}
    elif isinstance(meta, bytes):
        meta = json.loads(meta.decode())

    if isinstance(image, bytes):
        image_bytes = image
    elif isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_bytes = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, image_quality])[1].tobytes()
    
    if isinstance(depth, bytes):
        depth_bytes = depth
    elif isinstance(depth, np.ndarray):
        meta['depth_type'] = depth_type
        if depth_type == 'linear':
            if depth.dtype == np.float16:
                depth_format = 'exr'
                depth_bytes = cv2.imencode('.exr', depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])[1].tobytes()
            elif np.issubdtype(depth.dtype, np.floating):
                depth_format = 'exr'
                depth_bytes = cv2.imencode('.exr', depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])[1].tobytes()
            elif depth.dtype in [np.uint8, np.uint16]:
                depth_format = 'png'
                depth_bytes = cv2.imencode('.png', depth, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1].tobytes()
        elif depth_type == 'log':
            depth_format = 'png'
            depth = depth.astype(np.float32)
            near = max(depth[mask].min(), 1e-3)
            far = min(depth[mask].max(), near * depth_max_dynamic_range)
            depth = ((np.log(depth.clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65535).astype(np.uint16)
            depth_bytes = cv2.imencode('.png', depth, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1].tobytes()
            meta['depth_near'] = float(near)
            meta['depth_far'] = float(far)
        elif depth_type == 'disparity':
            depth_format = 'png'
            depth = depth.astype(np.float32)
            depth = 1 / (depth + 1e-12)
            depth = (depth / depth[mask].max()).clip(0, 1)
            if np.unique(depth) < 200:
                depth = (depth * 255).astype(np.uint8)
            else:
                depth = (depth * 65535).astype(np.uint16)
            depth_bytes = cv2.imencode('.png', depth, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1].tobytes()
    
    if isinstance(mask, bytes):
        mask_bytes = mask
    elif isinstance(mask, np.ndarray):
        mask_bytes = cv2.imencode('.png', mask.astype(np.uint8) * 255)[1].tobytes()

    if segmentation_mask is not None:
        if isinstance(segmentation_mask, bytes):
            segmentation_mask_bytes = segmentation_mask
        else:
            segmentation_mask_bytes = cv2.imencode('.png', segmentation_mask)[1].tobytes()
        assert segmentation_labels is not None, "You provided a segmentation mask, but not the corresponding labels."
        if isinstance(segmentation_labels, bytes):
            segmentation_labels = json.loads(segmentation_labels)
        meta['segmentation_labels'] = segmentation_labels

    if intrinsics is not None:
        meta['intrinsics'] = intrinsics.tolist()

    if normal is not None:
        if isinstance(normal, bytes):
            normal_bytes = normal
        elif isinstance(normal, np.ndarray):
            normal = ((normal * [0.5, -0.5, -0.5] + 0.5).clip(0, 1) * 65535).astype(np.uint16)
            normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
            normal_bytes = cv2.imencode('.png', normal, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1].tobytes()
        if normal_mask is None:
            normal_mask = np.ones(image.shape[:2], dtype=bool)
        normal_mask_bytes = cv2.imencode('.png', normal_mask.astype(np.uint8) * 255)[1].tobytes()

    meta_bytes = meta if isinstance(meta, bytes) else json.dumps(meta).encode()

    with zipfile.ZipFile(file, 'w') as z:
        z.writestr('meta.json', meta_bytes)
        z.writestr('image.jpg', image_bytes)
        z.writestr(f'depth.{depth_format}', depth_bytes)
        z.writestr('mask.png', mask_bytes)
        if segmentation_mask is not None:
            z.writestr('segmentation_mask.png', segmentation_mask_bytes)
        if normal is not None:
            z.writestr('normal.png', normal_bytes)
            z.writestr('normal_mask.png', normal_mask_bytes)


def read_rgbd_zip(file: Union[str, Path, IO], return_bytes: bool = False) -> Dict[str, Union[np.ndarray, Dict[str, Any], bytes]]:   
    """
    Read an RGBD zip file and return the image, depth, mask, segmentation_mask, intrinsics, and meta data.
    
    ### Parameters:
    - `file: Union[str, Path, IO]`
        The file path or file object to read from.
    - `return_bytes: bool = False`
        If True, return the image, depth, mask, and segmentation_mask as raw bytes.

    ### Returns:
    - `Tuple[Dict[str, Union[np.ndarray, Dict[str, Any]]], Dict[str, bytes]]`
        A dictionary containing: (If missing, the value will be None; if return_bytes is True, the value will be bytes)
        - `image`: RGB numpy.ndarray of shape (H, W, 3).
        - `depth`: float32 numpy.ndarray of shape (H, W).
        - `mask`: bool numpy.ndarray of shape (H, W). 
        - `segformer_mask`: uint8 numpy.ndarray of shape (H, W).
        - `intrinsics`: float32 numpy.ndarray of shape (3, 3).
        - `meta`: Dict[str, Any].
    """
    # Load & extract archive
    with zipfile.ZipFile(file, 'r') as z:
        meta = z.read('meta.json')
        if not return_bytes:
            meta = json.loads(z.read('meta.json'))

        image = z.read('image.jpg')
        if not return_bytes:
            image = cv2.imdecode(np.frombuffer(z.read('image.jpg'), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth_name = next(s for s in z.namelist() if s.startswith('depth'))
        depth = z.read(depth_name)
        if not return_bytes:
            depth = cv2.imdecode(np.frombuffer(z.read(depth_name), np.uint8), cv2.IMREAD_UNCHANGED)
        
        if 'mask.png' in z.namelist():
            mask = z.read('mask.png')
            if not return_bytes:
                mask = cv2.imdecode(np.frombuffer(z.read('mask.png'), np.uint8), cv2.IMREAD_UNCHANGED) > 0
        else:
            mask = None

        if 'segformer_mask.png' in z.namelist():
            # NOTE: Legacy support for segformer_mask.png
            segmentation_mask = z.read('segformer_mask.png')
            segmentation_labels = None
            if not return_bytes:
                segmentation_mask = cv2.imdecode(np.frombuffer(segmentation_mask, np.uint8), cv2.IMREAD_UNCHANGED)
                segmentation_labels = LEGACY_SEGFORMER_LABELS
        elif 'segmentation_mask.png' in z.namelist():
            segmentation_mask = z.read('segmentation_mask.png')
            segmentation_labels = None
            if not return_bytes:
                segmentation_mask = cv2.imdecode(np.frombuffer(segmentation_mask, np.uint8), cv2.IMREAD_UNCHANGED)
                segmentation_labels = meta['segmentation_labels']
        else:
            segmentation_mask = None
            segmentation_labels = None
        
        if 'normal.png' in z.namelist():
            normal = z.read('normal.png')
            if not return_bytes:
                normal = cv2.imdecode(np.frombuffer(z.read('normal.png'), np.uint8), cv2.IMREAD_UNCHANGED)
                normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
                normal = (normal.astype(np.float32) / 65535 - 0.5) * [2.0, -2.0, -2.0]
                normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        
            if 'normal_mask.png' in z.namelist():
                normal_mask = z.read('normal_mask.png')
                normal_mask = cv2.imdecode(np.frombuffer(normal_mask, np.uint8), cv2.IMREAD_UNCHANGED) > 0
            else:
                normal_mask = np.ones(image.shape[:2], dtype=bool)
        else:
            normal, normal_mask = None, None

    # recover linear depth
    if not return_bytes:
        if mask is None:
            mask = np.ones(image.shape[:2], dtype=bool)
        if meta['depth_type'] == 'linear':
            depth = depth.astype(np.float32)
            mask = mask & (depth > 0)
        elif meta['depth_type'] == 'log':
            near, far = meta['depth_near'], meta['depth_far']
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 65535
            elif depth.dtype == np.uint8:
                depth = depth.astype(np.float32) / 255
            depth = near ** (1 - depth) * far ** depth
            mask = mask & ~np.isnan(depth)
        elif meta['depth_type'] == 'disparity':
            mask = mask & (depth > 0)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 65535
            elif depth.dtype == np.uint8:
                depth = depth.astype(np.float32) / 255
            depth = 1 / (depth + 1e-12)
    
    # intrinsics
    if not return_bytes and 'intrinsics' in meta:
        intrinsics = np.array(meta['intrinsics'], dtype=np.float32)
    else:
        intrinsics = None

    # depth unit
    if not return_bytes and 'depth_unit' in meta:
        depth_unit_str = meta['depth_unit']
        if r := re.match(r'([\d.]*)(\w*)', depth_unit_str):
            digits, unit = r.groups()
            depth_unit = float(digits or 1) * {'m': 1, 'cm': 0.01, 'mm': 0.001}[unit]
        else:
            depth_unit = None
    else:
        depth_unit = None

    return_dict = {
        'image': image,
        'depth': depth,
        'mask': mask,
        'segmentation_mask': segmentation_mask,
        'segmentation_labels': segmentation_labels,
        'normal': normal,
        'normal_mask': normal_mask,
        'intrinsics': intrinsics,
        'depth_unit': depth_unit,
        'meta': meta,
    }
    return_dict = {k: v for k, v in return_dict.items() if v is not None}
    
    return return_dict

def write_rgbxyz(file: Union[IO, Path], image: np.ndarray, points: np.ndarray, mask: np.ndarray = None, image_quality: int = 95):
    if isinstance(image, bytes):
        image_bytes = image
    elif isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_bytes = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, image_quality])[1].tobytes()

    if isinstance(points, bytes):
        points_bytes = points
    elif isinstance(points, np.ndarray):
        points_bytes = cv2.imencode('.exr', points.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])[1].tobytes()
    
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)
    if isinstance(mask, bytes):
        mask_bytes = mask
    elif isinstance(mask, np.ndarray):
        mask_bytes = cv2.imencode('.png', mask.astype(np.uint8) * 255)[1].tobytes()

    is_archive = hasattr(file, 'write') or Path(file).suffix == '.zip'
    if is_archive:
        with zipfile.ZipFile(file, 'w') as z:
            z.writestr('image.jpg', image_bytes)
            z.writestr('points.exr', points_bytes)
            if mask is not None:
                z.writestr('mask.png', mask_bytes)
    else:
        file = Path(file)
        file.mkdir(parents=True, exist_ok=True)
        with open(file / 'image.jpg', 'wb') as f:
            f.write(image_bytes)
        with open(file / 'points.exr', 'wb') as f:
            f.write(points_bytes)
        if mask is not None:
            with open(file / 'mask.png', 'wb') as f:
                f.write(mask_bytes)


def read_rgbxyz(file: Union[IO, str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    is_archive = hasattr(file, 'read') or Path(file).suffix == '.zip'
    if is_archive:
        with zipfile.ZipFile(file, 'r') as z:
            image = cv2.imdecode(np.frombuffer(z.read('image.jpg'), np.uint8), cv2.IMREAD_COLOR)
            points = cv2.imdecode(np.frombuffer(z.read('points.exr'), np.uint8), cv2.IMREAD_UNCHANGED)
            if 'mask.png' in z.namelist():
                mask = cv2.imdecode(np.frombuffer(z.read('mask.png'), np.uint8), cv2.IMREAD_UNCHANGED) > 0
            else:
                mask = np.ones(image.shape[:2], dtype=bool)
    else:
        file = Path(file)
        file.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(file / 'image.jpg'), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = cv2.imread(str(file / 'points.exr'), cv2.IMREAD_UNCHANGED)
        if (file /'mask.png').exists():
            mask = cv2.imread(str(file / 'mask.png'), cv2.IMREAD_UNCHANGED) > 0
        else:
            mask = np.ones(image.shape[:2], dtype=bool)

    return image, points, mask


def save_glb(
    save_path: Union[str, os.PathLike], 
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_uvs: np.ndarray,
    texture: np.ndarray,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        visual = trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs, 
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(texture),
                metallicFactor=0.5,
                roughnessFactor=1.0
            )
        ),
        process=False
    ).export(save_path)


def save_ply(
    save_path: Union[str, os.PathLike], 
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_colors: np.ndarray,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    ).export(save_path)