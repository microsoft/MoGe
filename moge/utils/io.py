import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from typing import IO
import zipfile
import json
import io
from typing import *
from pathlib import Path
import re
from PIL import Image, PngImagePlugin

import numpy as np
import cv2 

from .tools import timeit


def save_glb(
    save_path: Union[str, os.PathLike], 
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_uvs: np.ndarray,
    texture: np.ndarray,
    vertex_normals: Optional[np.ndarray] = None,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices, 
        vertex_normals=vertex_normals,
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
    vertex_normals: Optional[np.ndarray] = None,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        vertex_normals=vertex_normals,
        process=False
    ).export(save_path)


def read_image(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a image, return uint8 RGB array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()  
    else:
        data = path.read()
    image = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image


def write_image(path: Union[str, os.PathLike, IO], image: np.ndarray, quality: int = 95):
    """
    Write a image, input uint8 RGB array of shape (H, W, 3).
    """
    data = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_bytes(data)
    else:
        path.write(data)


def read_depth(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a depth image, return float32 depth array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    depth = np.array(pil_image)
    mask_nan, mask_inf = depth == 0, depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth 
    if 'unit' in pil_image.info:    # Legacy support for depth units
        unit = float(pil_image.info.get('unit'))
        depth = depth * unit
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth


def write_depth(
    path: Union[str, os.PathLike, IO], 
    depth: np.ndarray, 
    max_range: float = 1e5,
    compression_level: int = 7,
):
    """
    Encode and write a depth image as 16-bit PNG format.
    ## Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `depth: np.ndarray`
        The depth array, float32 array of shape (H, W). 
        May contain `NaN` for invalid values and `Inf` for infinite values.
    
    Depth values are encoded as follows:
    - 0: unknown
    - 1 ~ 65534: depth values in logarithmic
    - 65535: infinity
    
    metadata is stored in the PNG file as text fields:
    - `near`: the minimum depth value
    - `far`: the maximum depth value
    """
    mask_values, mask_nan, mask_inf = np.isfinite(depth), np.isnan(depth),np.isinf(depth)

    depth = depth.astype(np.float32)
    mask_finite = depth
    near = max(depth[mask_values].min(), 1e-5)
    far = max(near * 1.1, min(depth[mask_values].max(), near * max_range))
    depth = 1 + np.round((np.log(np.nan_to_num(depth, nan=0).clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    depth[mask_nan] = 0
    depth[mask_inf] = 65535

    pil_image = Image.fromarray(depth)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('near', str(near))
    pnginfo.add_text('far', str(far))
    pil_image.save(path, pnginfo=pnginfo, compress_level=compression_level)


def read_segmentation(path: Union[str, os.PathLike, IO]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Read a segmentation mask
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to read from.
    ### Returns:
    - `Tuple[np.ndarray, Dict[str, int]]`
        A tuple containing:
        - `mask`: uint8 or uint16 numpy.ndarray of shape (H, W).
        - `labels`: Dict[str, int]. The label mapping, a dictionary of {label_name: label_id}.
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    labels = json.loads(pil_image.info['labels']) if 'labels' in pil_image.info else None
    mask = np.array(pil_image)
    return mask, labels


def write_segmentation(path: Union[str, os.PathLike, IO], mask: np.ndarray, labels: Dict[str, int] = None, compression_level: int = 7):
    """
    Write a segmentation mask and label mapping, as PNG format.
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `mask: np.ndarray`
        The segmentation mask, uint8 or uint16 array of shape (H, W).
    - `labels: Dict[str, int] = None`
        The label mapping, a dictionary of {label_name: label_id}.
    - `compression_level: int = 7`
        The compression level for PNG compression.
    """
    assert mask.dtype == np.uint8 or mask.dtype == np.uint16, f"Unsupported dtype {mask.dtype}"
    pil_image = Image.fromarray(mask)
    pnginfo = PngImagePlugin.PngInfo()
    if labels is not None:
        labels_json = json.dumps(labels, ensure_ascii=True, separators=(',', ':'))
        pnginfo.add_text('labels', labels_json)
    pil_image.save(path, pnginfo=pnginfo, compress_level=compression_level)



def read_normal(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a normal image, return float32 normal array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    normal = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    mask_nan = np.all(normal == 0, axis=-1)
    normal = (normal.astype(np.float32) / 65535 - 0.5) * [2.0, -2.0, -2.0]
    normal = normal / (np.sqrt(np.square(normal[..., 0]) + np.square(normal[..., 1]) + np.square(normal[..., 2])) + 1e-12)
    normal[mask_nan] = np.nan
    return normal


def write_normal(path: Union[str, os.PathLike, IO], normal: np.ndarray, compression_level: int = 7) -> np.ndarray:
    """
    Write a normal image, input float32 normal array of shape (H, W, 3).
    """
    mask_nan = np.isnan(normal).any(axis=-1)
    normal = ((normal * [0.5, -0.5, -0.5] + 0.5).clip(0, 1) * 65535).astype(np.uint16)
    normal[mask_nan] = 0
    data = cv2.imencode('.png', cv2.cvtColor(normal, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, compression_level])[1].tobytes()
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_bytes(data)
    else:
        path.write(data)


def read_mask(path: Union[str, os.PathLike, IO[bytes]]) -> np.ndarray:
    """
    Read a binary mask, return bool array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    return mask > 0


def write_mask(path: Union[str, os.PathLike, IO[bytes]], mask: np.ndarray, compression_level: int = 7):
    """
    Write a binary mask, input bool array of shape (H, W).
    """
    assert mask.dtype == bool, f"Mask must be bool array, got {mask.dtype}"
    mask = (mask.astype(np.uint8) * 255).astype(np.uint8)
    data = cv2.imencode('.png', mask, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])[1].tobytes()
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_bytes(data)
    else:
        path.write(data)

    
JSON_TYPE = Union[str, int, float, bool, None, Dict[str, "JSON"], List["JSON"]]


def read_json(path: Union[str, os.PathLike, IO[str]]) -> JSON_TYPE:
    if isinstance(path, (str, os.PathLike)):
        text = Path(path).read_text()
    else:
        text = path.read()
    return json.loads(text)


def write_json(path: Union[str, os.PathLike, IO[str]], content: JSON_TYPE):
    text = json.dumps(content)
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_text(text)
    else:
        path.write(text)