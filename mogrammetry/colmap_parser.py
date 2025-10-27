"""
Robust COLMAP file parser with support for all camera models.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass


@dataclass
class Camera:
    """Camera model representation."""
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def fx(self) -> float:
        """Get focal length in x direction."""
        if self.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE']:
            return self.params[0]
        elif self.model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'FOV', 'THIN_PRISM_FISHEYE']:
            return self.params[0]
        else:
            return self.params[0]

    @property
    def fy(self) -> float:
        """Get focal length in y direction."""
        if self.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE']:
            return self.params[0]
        elif self.model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'FOV', 'THIN_PRISM_FISHEYE']:
            return self.params[1]
        else:
            return self.params[0]

    @property
    def cx(self) -> float:
        """Get principal point x coordinate."""
        if self.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE']:
            return self.params[1]
        elif self.model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'FOV', 'THIN_PRISM_FISHEYE']:
            return self.params[2]
        else:
            return self.width / 2.0

    @property
    def cy(self) -> float:
        """Get principal point y coordinate."""
        if self.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE']:
            return self.params[2]
        elif self.model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'FOV', 'THIN_PRISM_FISHEYE']:
            return self.params[3]
        else:
            return self.height / 2.0

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 intrinsic matrix."""
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K

    def get_distortion(self) -> np.ndarray:
        """Get distortion parameters [k1, k2, p1, p2, k3, k4, k5, k6]."""
        if self.model == 'PINHOLE' or self.model == 'SIMPLE_PINHOLE':
            return np.zeros(8, dtype=np.float32)
        elif self.model == 'SIMPLE_RADIAL':
            k1 = self.params[3] if len(self.params) > 3 else 0.0
            return np.array([k1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        elif self.model == 'OPENCV':
            # fx, fy, cx, cy, k1, k2, p1, p2
            k1, k2, p1, p2 = self.params[4:8] if len(self.params) >= 8 else (0, 0, 0, 0)
            return np.array([k1, k2, p1, p2, 0, 0, 0, 0], dtype=np.float32)
        elif self.model == 'FULL_OPENCV':
            # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            distortion = list(self.params[4:12]) if len(self.params) >= 12 else [0] * 8
            return np.array(distortion, dtype=np.float32)
        else:
            return np.zeros(8, dtype=np.float32)


@dataclass
class Image:
    """Image representation with camera pose."""
    id: int
    qvec: np.ndarray  # Quaternion [qw, qx, qy, qz]
    tvec: np.ndarray  # Translation [tx, ty, tz]
    camera_id: int
    name: str
    point2D_idxs: Optional[np.ndarray] = None
    point3D_ids: Optional[np.ndarray] = None

    def get_rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix from quaternion."""
        # Convert from COLMAP convention (qw, qx, qy, qz) to scipy (qx, qy, qz, qw)
        qw, qx, qy, qz = self.qvec
        return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    def get_extrinsic_matrix(self) -> np.ndarray:
        """Get 4x4 extrinsic matrix (world-to-camera)."""
        R = self.get_rotation_matrix()
        t = self.tvec
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        return extrinsic

    def get_camera_to_world_matrix(self) -> np.ndarray:
        """Get 4x4 camera-to-world matrix (inverse of extrinsic)."""
        R = self.get_rotation_matrix()
        t = self.tvec
        # C2W = [R^T | -R^T @ t]
        R_inv = R.T
        t_inv = -R_inv @ t
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_inv
        c2w[:3, 3] = t_inv
        return c2w

    def get_camera_center(self) -> np.ndarray:
        """Get camera center in world coordinates."""
        R = self.get_rotation_matrix()
        return -R.T @ self.tvec


class COLMAPParser:
    """Parser for COLMAP reconstruction files."""

    # Camera model parameter counts
    CAMERA_MODEL_PARAMS = {
        'SIMPLE_PINHOLE': 3,      # f, cx, cy
        'PINHOLE': 4,              # fx, fy, cx, cy
        'SIMPLE_RADIAL': 4,        # f, cx, cy, k
        'RADIAL': 5,               # f, cx, cy, k1, k2
        'OPENCV': 8,               # fx, fy, cx, cy, k1, k2, p1, p2
        'OPENCV_FISHEYE': 8,       # fx, fy, cx, cy, k1, k2, k3, k4
        'FULL_OPENCV': 12,         # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        'FOV': 5,                  # fx, fy, cx, cy, omega
        'SIMPLE_RADIAL_FISHEYE': 4,
        'RADIAL_FISHEYE': 5,
        'THIN_PRISM_FISHEYE': 12,
    }

    def __init__(self, model_path: str):
        """
        Initialize COLMAP parser.

        Args:
            model_path: Path to COLMAP model directory containing cameras.txt and images.txt
        """
        self.model_path = Path(model_path)
        self.cameras: Dict[int, Camera] = {}
        self.images: Dict[int, Image] = {}
        self.points3D: Optional[np.ndarray] = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"COLMAP model path does not exist: {model_path}")

    def parse_cameras(self, cameras_file: Optional[str] = None) -> Dict[int, Camera]:
        """
        Parse COLMAP cameras.txt file.

        Args:
            cameras_file: Path to cameras.txt (default: model_path/cameras.txt)

        Returns:
            Dictionary mapping camera_id to Camera object
        """
        if cameras_file is None:
            cameras_file = self.model_path / 'cameras.txt'
        else:
            cameras_file = Path(cameras_file)

        if not cameras_file.exists():
            raise FileNotFoundError(f"Cameras file not found: {cameras_file}")

        cameras = {}
        with open(cameras_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = np.array([float(p) for p in parts[4:]], dtype=np.float32)

                # Validate parameter count
                if model in self.CAMERA_MODEL_PARAMS:
                    expected = self.CAMERA_MODEL_PARAMS[model]
                    if len(params) != expected:
                        raise ValueError(
                            f"Camera {camera_id}: Expected {expected} parameters "
                            f"for {model} model, got {len(params)}"
                        )

                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params
                )

        self.cameras = cameras
        return cameras

    def parse_images(self, images_file: Optional[str] = None) -> Dict[int, Image]:
        """
        Parse COLMAP images.txt file.

        Args:
            images_file: Path to images.txt (default: model_path/images.txt)

        Returns:
            Dictionary mapping image_id to Image object
        """
        if images_file is None:
            images_file = self.model_path / 'images.txt'
        else:
            images_file = Path(images_file)

        if not images_file.exists():
            raise FileNotFoundError(f"Images file not found: {images_file}")

        images = {}
        with open(images_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        # Images.txt has two lines per image
        for i in range(0, len(lines), 2):
            # First line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            parts = lines[i].split()
            if len(parts) < 10:
                continue

            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = ' '.join(parts[9:])  # Handle spaces in filename

            # Second line: POINTS2D[] as (X, Y, POINT3D_ID) triplets
            point2D_data = []
            point3D_ids = []
            if i + 1 < len(lines):
                points_line = lines[i + 1].split()
                for j in range(0, len(points_line), 3):
                    if j + 2 < len(points_line):
                        x, y = float(points_line[j]), float(points_line[j + 1])
                        p3d_id = int(points_line[j + 2])
                        point2D_data.append([x, y])
                        point3D_ids.append(p3d_id)

            images[image_id] = Image(
                id=image_id,
                qvec=np.array([qw, qx, qy, qz], dtype=np.float32),
                tvec=np.array([tx, ty, tz], dtype=np.float32),
                camera_id=camera_id,
                name=name,
                point2D_idxs=np.array(point2D_data, dtype=np.float32) if point2D_data else None,
                point3D_ids=np.array(point3D_ids, dtype=np.int32) if point3D_ids else None
            )

        self.images = images
        return images

    def parse_points3D(self, points3D_file: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Parse COLMAP points3D.txt file (optional).

        Args:
            points3D_file: Path to points3D.txt (default: model_path/points3D.txt)

        Returns:
            Nx6 array of [X, Y, Z, R, G, B] or None if file doesn't exist
        """
        if points3D_file is None:
            points3D_file = self.model_path / 'points3D.txt'
        else:
            points3D_file = Path(points3D_file)

        if not points3D_file.exists():
            return None

        points = []
        with open(points3D_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] ...
                if len(parts) >= 7:
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(int, parts[4:7])
                    points.append([x, y, z, r, g, b])

        if points:
            self.points3D = np.array(points, dtype=np.float32)
            return self.points3D
        return None

    def parse_all(self) -> Tuple[Dict[int, Camera], Dict[int, Image], Optional[np.ndarray]]:
        """
        Parse all COLMAP files (cameras, images, and optionally points3D).

        Returns:
            Tuple of (cameras, images, points3D)
        """
        self.parse_cameras()
        self.parse_images()
        self.parse_points3D()

        return self.cameras, self.images, self.points3D

    def get_image_by_name(self, name: str) -> Optional[Image]:
        """Get image by filename."""
        for img in self.images.values():
            if img.name == name:
                return img
        return None

    def validate(self) -> List[str]:
        """
        Validate parsed COLMAP data.

        Returns:
            List of warning messages (empty if all OK)
        """
        warnings = []

        # Check if cameras were parsed
        if not self.cameras:
            warnings.append("No cameras found in COLMAP model")

        # Check if images were parsed
        if not self.images:
            warnings.append("No images found in COLMAP model")
            return warnings

        # Check that all referenced cameras exist
        for img_id, img in self.images.items():
            if img.camera_id not in self.cameras:
                warnings.append(
                    f"Image {img_id} ({img.name}) references non-existent camera {img.camera_id}"
                )

        # Check for degenerate cameras (zero focal length)
        for cam_id, cam in self.cameras.items():
            if cam.fx <= 0 or cam.fy <= 0:
                warnings.append(f"Camera {cam_id} has invalid focal length: fx={cam.fx}, fy={cam.fy}")

        # Check for suspicious image dimensions
        for cam_id, cam in self.cameras.items():
            if cam.width <= 0 or cam.height <= 0:
                warnings.append(f"Camera {cam_id} has invalid dimensions: {cam.width}x{cam.height}")

        return warnings
