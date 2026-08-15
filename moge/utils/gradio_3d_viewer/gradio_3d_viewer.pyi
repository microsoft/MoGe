from html import escape
import json
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote
import uuid

import gradio as gr
import numpy as np
from PIL import Image


_PACKAGE_DIR = Path(__file__).resolve().parent
_ASSET_DIR = _PACKAGE_DIR / 'assets'
_TEMPLATE = (_PACKAGE_DIR / 'viewer.html').read_text(encoding='utf-8')
_FILE_URL_PREFIX = '/gradio_api/file='

from gradio.events import Dependency

class DepthMap3DViewer(gr.HTML):
    """Gradio HTML component for an offline, client-rendered point cloud viewer."""

    payload_filenames = ('view_depth.bin', 'view_color.webp')

    def __init__(
        self,
        value=None,
        *,
        height: str = '60vh',
        point_scale: float = 1.4,
        label: str = '3D Point Map',
        **kwargs,
    ):
        self.viewer_height = height
        self.point_scale = point_scale
        gr.set_static_paths(paths=[_ASSET_DIR])
        super().__init__(value=value, label=label, **kwargs)

    @staticmethod
    def payload_paths(output_dir: Union[str, Path]) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        return tuple(output_dir / filename for filename in DepthMap3DViewer.payload_filenames)

    @staticmethod
    def _file_url(path: Path, cache_key: Optional[str] = None) -> str:
        url = f'{_FILE_URL_PREFIX}{quote(path.resolve().as_posix(), safe="/:")}'
        return f'{url}?v={cache_key}' if cache_key is not None else url

    def build(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        intrinsics: np.ndarray,
        output_dir: Union[str, Path],
        *,
        mask: Optional[np.ndarray] = None,
    ) -> str:
        """Write the viewer payload and return an iframe HTML value for this component.

        ``intrinsics`` must be normalized to image width and height. ``image`` must
        be an RGB uint8 array with the same spatial shape as ``depth``.
        """
        depth = np.asarray(depth)
        image = np.asarray(image)
        intrinsics = np.asarray(intrinsics)
        if depth.ndim != 2:
            raise ValueError(f'depth must have shape (H, W), got {depth.shape}')
        if image.shape != (*depth.shape, 3) or image.dtype != np.uint8:
            raise ValueError(f'image must be RGB uint8 with shape {(*depth.shape, 3)}, got {image.shape} {image.dtype}')
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError(f'intrinsics must be a finite (3, 3) array, got {intrinsics.shape}')
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != depth.shape:
                raise ValueError(f'mask must have shape {depth.shape}, got {mask.shape}')

        valid = np.isfinite(depth)
        if mask is not None:
            valid &= mask.astype(bool)
        depth_view = np.where(valid, depth, 0).astype(np.float32)

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        depth_path, color_path = self.payload_paths(output_dir)
        depth_path.write_bytes(
            np.ascontiguousarray(depth_view.reshape(-1).view(np.uint8).reshape(-1, 4).T).tobytes()
        )
        Image.fromarray(image).save(color_path, format='WEBP', lossless=True)

        height, width = depth.shape
        cache_key = uuid.uuid4().hex
        config = {
            'width': width,
            'height': height,
            'fx': float(intrinsics[0, 0]),
            'fy': float(intrinsics[1, 1]),
            'cx': float(intrinsics[0, 2]),
            'cy': float(intrinsics[1, 2]),
            'pointScale': self.point_scale,
            'depthUrl': self._file_url(depth_path, cache_key),
            'colorUrl': self._file_url(color_path, cache_key),
        }
        document = (
            _TEMPLATE
            .replace('__THREE_URL__', self._file_url(_ASSET_DIR / 'three.module.js'))
            .replace('__ORBIT_CONTROLS_URL__', self._file_url(_ASSET_DIR / 'OrbitControls.js'))
            .replace('/*__CONFIG__*/null', json.dumps(config).replace('</', '<\\/'))
        )
        return (
            f'<iframe srcdoc="{escape(document, quote=True)}" '
            f'style="display:block;box-sizing:border-box;width:100%;height:{escape(self.viewer_height, quote=True)};'
            f'border:1px solid var(--block-border-color,#e5e5e5);border-radius:8px;background:#fff"></iframe>'
        )
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component