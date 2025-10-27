"""
Point cloud fusion and merging utilities.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import open3d as o3d
from dataclasses import dataclass


@dataclass
class PointCloudData:
    """Container for point cloud with metadata."""
    points: np.ndarray  # (N, 3)
    colors: Optional[np.ndarray] = None  # (N, 3)
    normals: Optional[np.ndarray] = None  # (N, 3)
    confidence: Optional[np.ndarray] = None  # (N,)
    source_image_id: Optional[int] = None
    camera_position: Optional[np.ndarray] = None  # (3,)


class PointCloudFusion:
    """Advanced point cloud fusion with multiple merging strategies."""

    def __init__(
        self,
        voxel_size: Optional[float] = None,
        outlier_removal: str = 'statistical',
        statistical_nb_neighbors: int = 20,
        statistical_std_ratio: float = 2.0,
        radius_nb_points: int = 16,
        radius: float = 0.05,
        merge_strategy: str = 'weighted',
        overlap_threshold: float = 0.8
    ):
        """
        Initialize point cloud fusion.

        Args:
            voxel_size: Voxel size for downsampling (auto if None)
            outlier_removal: Method ('statistical', 'radius', 'both', 'none')
            statistical_nb_neighbors: Neighbors for statistical outlier removal
            statistical_std_ratio: Std ratio for statistical outlier removal
            radius_nb_points: Min neighbors for radius outlier removal
            radius: Search radius for radius outlier removal
            merge_strategy: Strategy for merging overlapping points
            overlap_threshold: Threshold for detecting overlaps
        """
        self.voxel_size = voxel_size
        self.outlier_removal = outlier_removal
        self.statistical_nb_neighbors = statistical_nb_neighbors
        self.statistical_std_ratio = statistical_std_ratio
        self.radius_nb_points = radius_nb_points
        self.radius = radius
        self.merge_strategy = merge_strategy
        self.overlap_threshold = overlap_threshold

    def merge_point_clouds(
        self,
        point_clouds: List[PointCloudData],
        remove_outliers: bool = True,
        estimate_normals: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """
        Merge multiple point clouds into one.

        Args:
            point_clouds: List of PointCloudData objects
            remove_outliers: Apply outlier removal
            estimate_normals: Estimate normals for final cloud

        Returns:
            merged_pcd: Merged Open3D point cloud
            stats: Statistics about the merge
        """
        if not point_clouds:
            raise ValueError("No point clouds to merge")

        stats = {
            'num_input_clouds': len(point_clouds),
            'total_input_points': sum(len(pc.points) for pc in point_clouds),
            'points_per_cloud': [len(pc.points) for pc in point_clouds]
        }

        # Convert to Open3D point clouds
        o3d_clouds = []
        for pc_data in point_clouds:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_data.points)

            if pc_data.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(pc_data.colors)

            if pc_data.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(pc_data.normals)

            o3d_clouds.append(pcd)

        # Merge clouds
        if self.merge_strategy == 'append':
            merged_pcd = self._merge_append(o3d_clouds)
        elif self.merge_strategy == 'weighted':
            merged_pcd = self._merge_weighted(o3d_clouds, point_clouds)
        elif self.merge_strategy == 'average':
            merged_pcd = self._merge_average(o3d_clouds)
        else:
            merged_pcd = self._merge_append(o3d_clouds)

        stats['points_after_merge'] = len(merged_pcd.points)

        # Outlier removal
        if remove_outliers:
            merged_pcd, outlier_stats = self._remove_outliers(merged_pcd)
            stats.update(outlier_stats)

        # Voxel downsampling
        if self.voxel_size is not None:
            points_before = len(merged_pcd.points)
            merged_pcd = merged_pcd.voxel_down_sample(self.voxel_size)
            stats['points_after_voxel_downsample'] = len(merged_pcd.points)
            stats['voxel_size'] = self.voxel_size
            stats['downsampling_ratio'] = len(merged_pcd.points) / points_before if points_before > 0 else 0

        # Estimate normals if requested
        if estimate_normals and not merged_pcd.has_normals():
            self._estimate_normals(merged_pcd)
            stats['normals_estimated'] = True

        stats['final_point_count'] = len(merged_pcd.points)

        return merged_pcd, stats

    def _merge_append(self, clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """Simple concatenation of point clouds."""
        merged = o3d.geometry.PointCloud()

        for cloud in clouds:
            merged += cloud

        return merged

    def _merge_weighted(
        self,
        clouds: List[o3d.geometry.PointCloud],
        cloud_data: List[PointCloudData]
    ) -> o3d.geometry.PointCloud:
        """
        Merge with weighted averaging in overlapping regions.
        """
        # For now, use simple append with auto voxel downsampling
        # In a full implementation, we'd detect overlaps and weight by confidence
        merged = self._merge_append(clouds)

        # Auto voxel size if not specified
        if self.voxel_size is None:
            # Estimate voxel size from point cloud density
            bbox = merged.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            auto_voxel_size = np.min(extent) / 100.0
            merged = merged.voxel_down_sample(auto_voxel_size)

        return merged

    def _merge_average(self, clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """Merge with averaging in overlapping regions."""
        # Similar to weighted, but equal weights
        return self._merge_append(clouds)

    def _remove_outliers(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """Remove outliers from point cloud."""
        stats = {}
        points_before = len(pcd.points)

        if self.outlier_removal == 'none':
            stats['outliers_removed'] = 0
            return pcd, stats

        if self.outlier_removal in ['statistical', 'both']:
            pcd, ind = pcd.remove_statistical_outlier(
                nb_neighbors=self.statistical_nb_neighbors,
                std_ratio=self.statistical_std_ratio
            )
            stats['statistical_outliers_removed'] = points_before - len(pcd.points)
            points_before = len(pcd.points)

        if self.outlier_removal in ['radius', 'both']:
            pcd, ind = pcd.remove_radius_outlier(
                nb_points=self.radius_nb_points,
                radius=self.radius
            )
            stats['radius_outliers_removed'] = points_before - len(pcd.points)

        stats['total_outliers_removed'] = stats.get('statistical_outliers_removed', 0) + \
                                          stats.get('radius_outliers_removed', 0)

        return pcd, stats

    def _estimate_normals(self, pcd: o3d.geometry.PointCloud, radius: Optional[float] = None):
        """Estimate normals for point cloud."""
        if radius is None:
            # Auto-estimate radius
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            radius = np.min(extent) / 50.0

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=30
            )
        )

        # Orient normals consistently
        try:
            pcd.orient_normals_consistent_tangent_plane(k=15)
        except:
            # Fallback if orientation fails
            pass


def create_point_cloud_from_depth(
    depth: np.ndarray,
    image: np.ndarray,
    intrinsics: np.ndarray,
    mask: Optional[np.ndarray] = None,
    extrinsic: Optional[np.ndarray] = None
) -> PointCloudData:
    """
    Create point cloud from depth map.

    Args:
        depth: Depth map (H, W)
        image: RGB image (H, W, 3) in [0, 1]
        intrinsics: Camera intrinsic matrix (3, 3)
        mask: Valid pixel mask (H, W)
        extrinsic: Camera-to-world transformation (4, 4)

    Returns:
        PointCloudData object
    """
    H, W = depth.shape

    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    # Create pixel coordinates
    v, u = np.mgrid[0:H, 0:W]

    # Extract intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Back-project to 3D
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Stack to (H, W, 3)
    points_camera = np.stack([X, Y, Z], axis=-1)

    # Transform to world if extrinsic provided
    if extrinsic is not None:
        points_flat = points_camera.reshape(-1, 3)
        points_homo = np.hstack([points_flat, np.ones((len(points_flat), 1))])
        points_world_homo = (extrinsic @ points_homo.T).T
        points_camera = points_world_homo[:, :3].reshape(H, W, 3)

    # Apply mask and flatten
    points = points_camera[mask]
    colors = image[mask] if image is not None else None

    # Get camera position
    camera_position = None
    if extrinsic is not None:
        # Camera center is -R^T @ t
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        camera_position = -R.T @ t

    return PointCloudData(
        points=points.astype(np.float32),
        colors=colors.astype(np.float32) if colors is not None else None,
        camera_position=camera_position
    )


def filter_points_by_depth_range(
    points: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 1000.0,
    camera_position: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Filter points by depth range.

    Args:
        points: Points (N, 3)
        min_depth: Minimum depth
        max_depth: Maximum depth
        camera_position: Camera position for depth computation

    Returns:
        Boolean mask of valid points
    """
    if camera_position is not None:
        # Compute depth from camera
        depths = np.linalg.norm(points - camera_position, axis=-1)
    else:
        # Use Z coordinate
        depths = points[:, 2]

    valid = (depths >= min_depth) & (depths <= max_depth)
    return valid


def compute_point_cloud_statistics(pcd: o3d.geometry.PointCloud) -> Dict:
    """Compute statistics about a point cloud."""
    points = np.asarray(pcd.points)

    stats = {
        'num_points': len(points),
        'has_colors': pcd.has_colors(),
        'has_normals': pcd.has_normals(),
    }

    if len(points) > 0:
        stats['centroid'] = np.mean(points, axis=0).tolist()
        stats['bbox_min'] = np.min(points, axis=0).tolist()
        stats['bbox_max'] = np.max(points, axis=0).tolist()
        stats['bbox_extent'] = (np.max(points, axis=0) - np.min(points, axis=0)).tolist()

        # Compute density estimate
        if len(points) > 1:
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            k = min(10, len(points))
            distances = []
            for i in range(min(1000, len(points))):
                [_, idx, dist] = pcd_tree.search_knn_vector_3d(points[i], k)
                if len(dist) > 1:
                    distances.append(np.mean(np.sqrt(dist[1:])))

            if distances:
                stats['avg_nearest_neighbor_distance'] = float(np.mean(distances))
                stats['density_estimate'] = 1.0 / (float(np.mean(distances)) ** 3) if np.mean(distances) > 0 else 0

    return stats
