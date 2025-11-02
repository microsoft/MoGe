"""
Scale and shift alignment solvers for affine-invariant geometry.

Implements ROE (Robust Outlier Estimation) and other alignment methods
to recover absolute scale from affine-invariant point maps.
"""

from typing import Tuple, Optional, Dict
import numpy as np
from scipy.optimize import least_squares, minimize
import cv2


class AlignmentSolver:
    """Base class for alignment solvers."""

    def __init__(
        self,
        method: str = 'roe',
        ransac_threshold: float = 0.1,
        ransac_iterations: int = 1000,
        truncation_threshold: float = 0.05,
        min_valid_points: int = 100,
        use_reprojection: bool = True
    ):
        """
        Initialize alignment solver.

        Args:
            method: Alignment method ('roe', 'ransac', 'least_squares')
            ransac_threshold: Inlier threshold for RANSAC
            ransac_iterations: Number of RANSAC iterations
            truncation_threshold: Truncation threshold for robust estimation
            min_valid_points: Minimum number of valid points required
            use_reprojection: Use reprojection error for alignment
        """
        self.method = method
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.truncation_threshold = truncation_threshold
        self.min_valid_points = min_valid_points
        self.use_reprojection = use_reprojection

    def solve(
        self,
        points_pred: np.ndarray,
        intrinsics: np.ndarray,
        mask: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, np.ndarray, Dict]:
        """
        Solve for scale and shift to align affine-invariant points.

        Args:
            points_pred: Predicted affine-invariant point map (H, W, 3)
            intrinsics: Camera intrinsic matrix (3, 3)
            mask: Valid pixel mask (H, W)
            image_shape: (height, width) if different from points_pred shape

        Returns:
            scale: Scale factor
            shift: 3D shift vector [tx, ty, tz]
            stats: Dictionary with alignment statistics
        """
        if self.method == 'roe':
            return self._solve_roe(points_pred, intrinsics, mask, image_shape)
        elif self.method == 'ransac':
            return self._solve_ransac(points_pred, intrinsics, mask, image_shape)
        elif self.method == 'least_squares':
            return self._solve_least_squares(points_pred, intrinsics, mask, image_shape)
        else:
            raise ValueError(f"Unknown alignment method: {self.method}")

    def _solve_roe(
        self,
        points_pred: np.ndarray,
        intrinsics: np.ndarray,
        mask: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]]
    ) -> Tuple[float, np.ndarray, Dict]:
        """
        Robust Outlier Estimation (ROE) solver.

        This implements the alignment strategy from the MoGe paper, using
        truncated L1 loss for robust estimation.
        """
        H, W = points_pred.shape[:2]
        if mask is None:
            mask = np.ones((H, W), dtype=bool)

        # Get valid points
        valid_points = points_pred[mask]  # (N, 3)
        if len(valid_points) < self.min_valid_points:
            # Fallback to simple depth-based alignment
            return self._fallback_alignment(points_pred, mask)

        # Create pixel coordinates
        v_coords, u_coords = np.mgrid[0:H, 0:W]
        u_valid = u_coords[mask].astype(np.float32)
        v_valid = v_coords[mask].astype(np.float32)

        # Extract intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Predicted points in camera frame
        X_pred, Y_pred, Z_pred = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

        # For affine-invariant alignment, we assume t_x = t_y = 0
        # and only solve for scale s and shift t_z
        # Aligned: X' = s * X_pred, Y' = s * Y_pred, Z' = s * Z_pred + t_z

        # Reprojection constraint:
        # u = (X' / Z') * fx + cx = (s * X_pred) / (s * Z_pred + t_z) * fx + cx
        # v = (Y' / Z') * fy + cy = (s * Y_pred) / (s * Z_pred + t_z) * fy + cy

        def compute_reprojection_error(params):
            """Compute reprojection error with truncation."""
            s, t_z = params

            # Aligned coordinates
            Z_aligned = s * Z_pred + t_z

            # Avoid division by zero or negative depths
            valid = Z_aligned > 0.01
            if np.sum(valid) < self.min_valid_points:
                return 1e10

            X_aligned = s * X_pred[valid]
            Y_aligned = s * Y_pred[valid]
            Z_aligned = Z_aligned[valid]

            # Project to image plane
            u_proj = (X_aligned / Z_aligned) * fx + cx
            v_proj = (Y_aligned / Z_aligned) * fy + cy

            # Compute errors
            u_err = np.abs(u_proj - u_valid[valid])
            v_err = np.abs(v_proj - v_valid[valid])

            # Truncated L1 loss
            u_err_trunc = np.minimum(u_err, self.truncation_threshold * W)
            v_err_trunc = np.minimum(v_err, self.truncation_threshold * H)

            total_error = np.mean(u_err_trunc + v_err_trunc)
            return total_error

        # Grid search for good initialization
        best_params = None
        best_error = float('inf')

        # Estimate initial scale from depth statistics
        z_median = np.median(Z_pred)
        z_std = np.std(Z_pred)

        # Search over reasonable scale and shift values
        s_candidates = np.linspace(0.5, 2.0, 10)
        tz_candidates = np.linspace(-z_median, z_median, 10)

        for s in s_candidates:
            for t_z in tz_candidates:
                error = compute_reprojection_error([s, t_z])
                if error < best_error:
                    best_error = error
                    best_params = [s, t_z]

        # Refine with optimization
        if self.use_reprojection and best_params is not None:
            result = minimize(
                compute_reprojection_error,
                best_params,
                method='Powell',
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            if result.success:
                scale, t_z = result.x
            else:
                scale, t_z = best_params
        else:
            scale, t_z = best_params if best_params else (1.0, 0.0)

        # Ensure reasonable values
        scale = np.clip(scale, 0.1, 10.0)

        shift = np.array([0.0, 0.0, t_z], dtype=np.float32)

        stats = {
            'method': 'roe',
            'scale': scale,
            'shift': shift.tolist(),
            'final_error': best_error,
            'num_valid_points': len(valid_points),
            'z_median': float(z_median),
            'z_std': float(z_std)
        }

        return scale, shift, stats

    def _solve_ransac(
        self,
        points_pred: np.ndarray,
        intrinsics: np.ndarray,
        mask: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]]
    ) -> Tuple[float, np.ndarray, Dict]:
        """RANSAC-based alignment solver."""
        H, W = points_pred.shape[:2]
        if mask is None:
            mask = np.ones((H, W), dtype=bool)

        valid_points = points_pred[mask]
        if len(valid_points) < self.min_valid_points:
            return self._fallback_alignment(points_pred, mask)

        # Use depth-only RANSAC for simplicity
        Z_pred = valid_points[:, 2]

        # Assume we want positive depths with reasonable distribution
        # Find scale and shift to make depths positive and well-distributed
        best_inliers = 0
        best_s, best_tz = 1.0, 0.0

        for _ in range(self.ransac_iterations):
            # Sample random scale and shift
            s = np.random.uniform(0.5, 2.0)
            t_z = np.random.uniform(-2.0, 2.0)

            Z_aligned = s * Z_pred + t_z

            # Count inliers (positive depths)
            inliers = np.sum(Z_aligned > 0.1)

            if inliers > best_inliers:
                best_inliers = inliers
                best_s = s
                best_tz = t_z

        scale = best_s
        shift = np.array([0.0, 0.0, best_tz], dtype=np.float32)

        stats = {
            'method': 'ransac',
            'scale': scale,
            'shift': shift.tolist(),
            'inliers': int(best_inliers),
            'num_valid_points': len(valid_points)
        }

        return scale, shift, stats

    def _solve_least_squares(
        self,
        points_pred: np.ndarray,
        intrinsics: np.ndarray,
        mask: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]]
    ) -> Tuple[float, np.ndarray, Dict]:
        """Least squares alignment solver."""
        H, W = points_pred.shape[:2]
        if mask is None:
            mask = np.ones((H, W), dtype=bool)

        valid_points = points_pred[mask]
        if len(valid_points) < self.min_valid_points:
            return self._fallback_alignment(points_pred, mask)

        # Simple least squares: ensure all Z are positive
        Z_pred = valid_points[:, 2]

        # Find shift to make minimum Z = 0.1
        t_z = 0.1 - np.min(Z_pred)
        s = 1.0

        # Optional: normalize scale based on median depth
        z_median = np.median(Z_pred + t_z)
        if z_median > 0:
            s = 1.0 / z_median  # Normalize to unit scale

        scale = s
        shift = np.array([0.0, 0.0, t_z], dtype=np.float32)

        stats = {
            'method': 'least_squares',
            'scale': scale,
            'shift': shift.tolist(),
            'num_valid_points': len(valid_points),
            'z_median': float(np.median(Z_pred))
        }

        return scale, shift, stats

    def _fallback_alignment(
        self,
        points_pred: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, np.ndarray, Dict]:
        """Fallback alignment when not enough valid points."""
        valid_points = points_pred[mask] if mask is not None else points_pred.reshape(-1, 3)

        if len(valid_points) == 0:
            # No valid points - return identity
            return 1.0, np.zeros(3, dtype=np.float32), {'method': 'fallback', 'error': 'no_valid_points'}

        # Simple heuristic: shift Z to be positive
        Z_pred = valid_points[:, 2]
        t_z = 0.1 - np.min(Z_pred)

        scale = 1.0
        shift = np.array([0.0, 0.0, t_z], dtype=np.float32)

        stats = {
            'method': 'fallback',
            'scale': scale,
            'shift': shift.tolist(),
            'num_valid_points': len(valid_points),
            'warning': 'insufficient_points'
        }

        return scale, shift, stats


def align_points(
    points: np.ndarray,
    scale: float,
    shift: np.ndarray
) -> np.ndarray:
    """
    Apply scale and shift to points.

    Args:
        points: Point array (..., 3)
        scale: Scale factor
        shift: 3D shift vector

    Returns:
        Aligned points
    """
    return scale * points + shift


def transform_points_to_world(
    points_camera: np.ndarray,
    extrinsic: np.ndarray
) -> np.ndarray:
    """
    Transform points from camera frame to world frame.

    Args:
        points_camera: Points in camera frame (..., 3)
        extrinsic: 4x4 camera-to-world transformation matrix

    Returns:
        Points in world frame
    """
    original_shape = points_camera.shape
    points_flat = points_camera.reshape(-1, 3)

    # Convert to homogeneous coordinates
    points_homo = np.hstack([points_flat, np.ones((len(points_flat), 1), dtype=np.float32)])

    # Transform
    points_world_homo = (extrinsic @ points_homo.T).T

    # Convert back to 3D
    points_world = points_world_homo[:, :3]

    return points_world.reshape(original_shape)
