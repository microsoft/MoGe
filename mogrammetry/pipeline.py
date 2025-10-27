"""
Main MoGrammetry pipeline for 3D reconstruction.
"""

from typing import Optional, List, Dict, Tuple
from pathlib import Path
import numpy as np
import torch
import cv2
import open3d as o3d
from tqdm import tqdm
import json

from moge.model import MoGeModel

from .config import MoGrammetryConfig
from .colmap_parser import COLMAPParser, Camera, Image
from .alignment import AlignmentSolver, align_points, transform_points_to_world
from .fusion import PointCloudFusion, PointCloudData, create_point_cloud_from_depth
from .mesh import MeshGenerator, TextureMapper, save_mesh
from .logger import setup_logger, ProgressLogger


class MoGrammetryPipeline:
    """Main pipeline for MoGrammetry reconstruction."""

    def __init__(self, config: MoGrammetryConfig):
        """
        Initialize MoGrammetry pipeline.

        Args:
            config: Configuration object
        """
        self.config = config
        config.validate()

        # Setup logging
        self.logger = setup_logger(
            level=config.log_level,
            log_file=config.log_file,
            colorize=True
        )
        self.progress_logger = ProgressLogger(self.logger)

        # Initialize components
        self.device = torch.device(config.processing.device if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Load MoGe model
        self.progress_logger.start_task("Loading MoGe model")
        self.moge_model = MoGeModel.from_pretrained(config.model_name).to(self.device).eval()
        self.progress_logger.end_task("Loading MoGe model")

        # Parse COLMAP data
        self.progress_logger.start_task("Parsing COLMAP data")
        self.colmap_parser = COLMAPParser(config.colmap_model_path)
        self.cameras, self.images, self.points3D = self.colmap_parser.parse_all()
        self.progress_logger.end_task(
            "Parsing COLMAP data",
            f"{len(self.cameras)} cameras, {len(self.images)} images"
        )

        # Validate COLMAP data
        warnings = self.colmap_parser.validate()
        for warning in warnings:
            self.logger.warning(warning)

        # Initialize processing modules
        self.alignment_solver = AlignmentSolver(
            method=config.alignment.method,
            ransac_threshold=config.alignment.ransac_threshold,
            ransac_iterations=config.alignment.ransac_iterations,
            truncation_threshold=config.alignment.truncation_threshold,
            min_valid_points=config.alignment.min_valid_points,
            use_reprojection=config.alignment.use_reprojection
        )

        self.fusion = PointCloudFusion(
            voxel_size=config.fusion.voxel_size,
            outlier_removal=config.fusion.outlier_removal,
            statistical_nb_neighbors=config.fusion.statistical_nb_neighbors,
            statistical_std_ratio=config.fusion.statistical_std_ratio,
            radius_nb_points=config.fusion.radius_nb_points,
            radius=config.fusion.radius,
            merge_strategy=config.fusion.merge_strategy,
            overlap_threshold=config.fusion.overlap_threshold
        )

        if config.output.save_mesh:
            self.mesh_generator = MeshGenerator(
                method=config.mesh.method,
                poisson_depth=config.mesh.poisson_depth,
                poisson_width=config.mesh.poisson_width,
                poisson_scale=config.mesh.poisson_scale,
                poisson_linear_fit=config.mesh.poisson_linear_fit,
                ball_pivoting_radii=config.mesh.ball_pivoting_radii,
                alpha=config.mesh.alpha,
                simplify_mesh=config.mesh.simplify_mesh,
                target_faces=config.mesh.target_faces
            )

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'num_cameras': len(self.cameras),
            'num_images': len(self.images),
            'num_sparse_points': len(self.points3D) if self.points3D is not None else 0,
            'processed_images': [],
            'failed_images': []
        }

    def run(self) -> Dict:
        """
        Run complete reconstruction pipeline.

        Returns:
            Dictionary with statistics and output paths
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting MoGrammetry Pipeline")
        self.logger.info("=" * 80)

        # Process all images
        point_clouds = self._process_all_images()

        if not point_clouds:
            self.logger.error("No point clouds generated. Exiting.")
            return self.stats

        # Merge point clouds
        merged_pcd, fusion_stats = self._merge_point_clouds(point_clouds)
        self.stats['fusion'] = fusion_stats

        # Save point cloud
        if self.config.output.save_point_cloud:
            self._save_point_cloud(merged_pcd)

        # Generate mesh
        if self.config.output.save_mesh:
            mesh, mesh_stats = self._generate_mesh(merged_pcd)
            self.stats['mesh'] = mesh_stats

            if mesh is not None:
                self._save_mesh(mesh)

        # Save report
        if self.config.output.export_report:
            self._save_report()

        # Print summary
        self._print_summary()

        return self.stats

    def _process_all_images(self) -> List[PointCloudData]:
        """Process all images with MoGe."""
        self.progress_logger.start_task("Processing images with MoGe")

        point_clouds = []
        image_dir = Path(self.config.image_dir)

        # Process images
        for img_id, img_data in tqdm(
            self.images.items(),
            desc="Processing images",
            disable=not self.config.progress_bar
        ):
            try:
                pc_data = self._process_single_image(img_id, img_data, image_dir)
                if pc_data is not None:
                    point_clouds.append(pc_data)
                    self.stats['processed_images'].append(img_data.name)
            except Exception as e:
                self.logger.error(f"Failed to process {img_data.name}: {e}")
                self.stats['failed_images'].append(img_data.name)

        self.progress_logger.end_task(
            "Processing images with MoGe",
            f"Processed {len(point_clouds)}/{len(self.images)} images"
        )

        return point_clouds

    def _process_single_image(
        self,
        img_id: int,
        img_data: Image,
        image_dir: Path
    ) -> Optional[PointCloudData]:
        """Process single image with MoGe."""
        # Load image
        img_path = image_dir / img_data.name
        if not img_path.exists():
            self.logger.warning(f"Image not found: {img_path}")
            return None

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        # Convert to tensor
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=self.device)
        image_tensor = image_tensor.permute(2, 0, 1)

        # Get camera info
        camera = self.cameras[img_data.camera_id]

        # Run MoGe inference
        with torch.no_grad():
            output = self.moge_model.infer(
                image_tensor,
                resolution_level=self.config.processing.resolution_level
            )

        # Extract outputs
        points_pred = output['points'].cpu().numpy()  # (H, W, 3)
        mask = output['mask'].cpu().numpy() > 0.5
        intrinsics_moge = output['intrinsics'].cpu().numpy()

        # Use COLMAP intrinsics instead of MoGe's estimated intrinsics
        intrinsics_colmap = camera.get_intrinsic_matrix()

        # Align affine-invariant points
        scale, shift, align_stats = self.alignment_solver.solve(
            points_pred,
            intrinsics_colmap,
            mask,
            (H, W)
        )

        # Apply alignment
        points_aligned = align_points(points_pred, scale, shift)

        # Get camera-to-world transformation
        c2w = img_data.get_camera_to_world_matrix()

        # Transform to world coordinates
        points_world = transform_points_to_world(points_aligned, c2w)

        # Create point cloud data
        pc_data = PointCloudData(
            points=points_world[mask].astype(np.float32),
            colors=(image / 255.0)[mask].astype(np.float32),
            source_image_id=img_id,
            camera_position=img_data.get_camera_center()
        )

        # Save intermediate results if requested
        if self.config.processing.save_intermediate:
            self._save_intermediate_results(img_data.name, points_world, mask, image, align_stats)

        return pc_data

    def _merge_point_clouds(
        self,
        point_clouds: List[PointCloudData]
    ) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """Merge all point clouds."""
        self.progress_logger.start_task("Merging point clouds")

        merged_pcd, stats = self.fusion.merge_point_clouds(
            point_clouds,
            remove_outliers=True,
            estimate_normals=True
        )

        self.progress_logger.end_task(
            "Merging point clouds",
            f"{stats['final_point_count']} points"
        )

        return merged_pcd, stats

    def _generate_mesh(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> Tuple[Optional[o3d.geometry.TriangleMesh], Dict]:
        """Generate mesh from point cloud."""
        self.progress_logger.start_task("Generating mesh")

        try:
            mesh, stats = self.mesh_generator.generate_mesh(pcd)
            self.progress_logger.end_task(
                "Generating mesh",
                f"{stats['faces_after_cleanup']} faces"
            )
            return mesh, stats
        except Exception as e:
            self.logger.error(f"Mesh generation failed: {e}")
            self.progress_logger.end_task("Generating mesh", "FAILED")
            return None, {'error': str(e)}

    def _save_point_cloud(self, pcd: o3d.geometry.PointCloud):
        """Save point cloud to file."""
        self.progress_logger.start_task("Saving point cloud")

        for fmt in self.config.output.formats:
            if fmt == 'ply':
                output_path = self.output_dir / 'point_cloud.ply'
                o3d.io.write_point_cloud(str(output_path), pcd)
                self.logger.info(f"Saved point cloud: {output_path}")
                self.stats['point_cloud_path'] = str(output_path)

        self.progress_logger.end_task("Saving point cloud")

    def _save_mesh(self, mesh: o3d.geometry.TriangleMesh):
        """Save mesh to file."""
        self.progress_logger.start_task("Saving mesh")

        for fmt in self.config.output.formats:
            output_path = self.output_dir / f'mesh.{fmt}'
            save_mesh(mesh, str(output_path), format=fmt)
            self.logger.info(f"Saved mesh: {output_path}")

            if 'mesh_paths' not in self.stats:
                self.stats['mesh_paths'] = []
            self.stats['mesh_paths'].append(str(output_path))

        self.progress_logger.end_task("Saving mesh")

    def _save_intermediate_results(
        self,
        image_name: str,
        points: np.ndarray,
        mask: np.ndarray,
        image: np.ndarray,
        align_stats: Dict
    ):
        """Save intermediate results for debugging."""
        intermediate_dir = self.output_dir / 'intermediate' / Path(image_name).stem
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Save alignment stats
        with open(intermediate_dir / 'alignment_stats.json', 'w') as f:
            json.dump(align_stats, f, indent=2)

        # Save mask
        cv2.imwrite(
            str(intermediate_dir / 'mask.png'),
            (mask * 255).astype(np.uint8)
        )

    def _save_report(self):
        """Save processing report."""
        self.progress_logger.start_task("Saving report")

        report_path = self.output_dir / 'reconstruction_report.json'

        report = {
            'config': self.config.to_dict(),
            'statistics': self.stats,
            'timing': self.progress_logger.get_summary()
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Saved report: {report_path}")
        self.progress_logger.end_task("Saving report")

    def _print_summary(self):
        """Print pipeline summary."""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Pipeline Summary")
        self.logger.info("=" * 80)

        self.logger.info(f"Input images: {self.stats['num_images']}")
        self.logger.info(f"Successfully processed: {len(self.stats['processed_images'])}")
        self.logger.info(f"Failed: {len(self.stats['failed_images'])}")

        if 'fusion' in self.stats:
            fusion_stats = self.stats['fusion']
            self.logger.info(f"Final point cloud: {fusion_stats['final_point_count']} points")

        if 'mesh' in self.stats and 'error' not in self.stats['mesh']:
            mesh_stats = self.stats['mesh']
            self.logger.info(f"Mesh: {mesh_stats['faces_after_cleanup']} faces")

        self.logger.info(f"Total time: {self.progress_logger.get_total_time():.2f}s")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)


def run_mogrammetry(config: MoGrammetryConfig) -> Dict:
    """
    Convenience function to run MoGrammetry pipeline.

    Args:
        config: Configuration object

    Returns:
        Statistics dictionary
    """
    pipeline = MoGrammetryPipeline(config)
    return pipeline.run()
