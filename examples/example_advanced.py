#!/usr/bin/env python3
"""
Advanced example with custom processing.

Demonstrates:
- Loading configuration from file
- Custom alignment and fusion settings
- Processing statistics
- Error handling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mogrammetry import MoGrammetryPipeline, MoGrammetryConfig
from mogrammetry.config import AlignmentConfig, FusionConfig, MeshConfig
import json


def main():
    # Create configuration with custom sub-configs
    config = MoGrammetryConfig(
        colmap_model_path='path/to/colmap/sparse/0',
        image_dir='path/to/images',
        output_dir='output/advanced_example',
        model_name='Ruicheng/moge-vitl',
        log_level='DEBUG',
        log_file='output/advanced_example/reconstruction.log'
    )

    # Customize alignment
    config.alignment = AlignmentConfig(
        method='roe',
        use_reprojection=True,
        truncation_threshold=0.03,
        min_valid_points=200,
        ransac_iterations=2000
    )

    # Customize fusion
    config.fusion = FusionConfig(
        voxel_size=0.01,  # 1cm voxels
        outlier_removal='both',  # Statistical + radius
        statistical_nb_neighbors=30,
        statistical_std_ratio=2.5,
        merge_strategy='weighted'
    )

    # Customize mesh
    config.mesh = MeshConfig(
        method='poisson',
        poisson_depth=10,  # High quality
        simplify_mesh=True,
        target_faces=500000
    )

    # Save configuration for reference
    config_path = Path(config.output_dir) / 'config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(config_path))
    print(f"Saved configuration to: {config_path}")

    # Run pipeline with error handling
    try:
        print("\nStarting advanced reconstruction...")
        pipeline = MoGrammetryPipeline(config)
        stats = pipeline.run()

        # Analyze statistics
        print("\n" + "=" * 80)
        print("RECONSTRUCTION STATISTICS")
        print("=" * 80)

        print(f"\nInput:")
        print(f"  Total images: {stats['num_images']}")
        print(f"  Cameras: {stats['num_cameras']}")

        print(f"\nProcessing:")
        print(f"  Successful: {len(stats['processed_images'])}")
        print(f"  Failed: {len(stats['failed_images'])}")

        if 'fusion' in stats:
            fusion = stats['fusion']
            print(f"\nPoint Cloud Fusion:")
            print(f"  Input points: {fusion['total_input_points']}")
            print(f"  After merge: {fusion['points_after_merge']}")
            if 'points_after_voxel_downsample' in fusion:
                print(f"  After downsampling: {fusion['points_after_voxel_downsample']}")
            print(f"  Final points: {fusion['final_point_count']}")
            if 'total_outliers_removed' in fusion:
                print(f"  Outliers removed: {fusion['total_outliers_removed']}")

        if 'mesh' in stats and 'error' not in stats['mesh']:
            mesh = stats['mesh']
            print(f"\nMesh Generation:")
            print(f"  Method: {mesh['method']}")
            print(f"  Vertices: {mesh['vertices_after_cleanup']}")
            print(f"  Faces: {mesh['faces_after_cleanup']}")
            if 'faces_after_simplification' in mesh:
                print(f"  Simplified faces: {mesh['faces_after_simplification']}")

        # Save detailed statistics
        stats_path = Path(config.output_dir) / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nDetailed statistics saved to: {stats_path}")

        print("\n✓ Reconstruction completed successfully!")

    except Exception as e:
        print(f"\n✗ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
