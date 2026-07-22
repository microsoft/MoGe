#!/usr/bin/env python3
"""
MoGrammetry Demo - Showcase the system capabilities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tempfile
import shutil
import numpy as np


def demo_configuration():
    """Demonstrate configuration system."""
    print("=" * 80)
    print("1. CONFIGURATION SYSTEM")
    print("=" * 80)

    from mogrammetry import MoGrammetryConfig

    # Create a configuration
    config = MoGrammetryConfig(
        colmap_model_path='/path/to/colmap/sparse/0',
        image_dir='/path/to/images',
        output_dir='/path/to/output'
    )

    print("\n✓ Created configuration with default settings:")
    print(f"  - Model: {config.model_name}")
    print(f"  - Model Version: {config.model_version}")
    print(f"  - Device: {config.processing.device}")
    print(f"  - Alignment method: {config.alignment.method}")
    print(f"  - Fusion voxel size: {config.fusion.voxel_size}")
    print(f"  - Mesh method: {config.mesh.method}")

    # Save to YAML
    temp_yaml = tempfile.mktemp(suffix='.yaml')
    config.to_yaml(temp_yaml)
    print(f"\n✓ Saved configuration to: {temp_yaml}")

    # Show YAML content
    with open(temp_yaml, 'r') as f:
        yaml_content = f.read()
    print("\nYAML Configuration:")
    print("-" * 80)
    print(yaml_content[:500] + "..." if len(yaml_content) > 500 else yaml_content)
    print("-" * 80)

    # Load from YAML
    loaded_config = MoGrammetryConfig.from_yaml(temp_yaml)
    print(f"\n✓ Loaded configuration from YAML")
    print(f"  - Verified: output_dir = {loaded_config.output_dir}")

    Path(temp_yaml).unlink()


def demo_colmap_parser():
    """Demonstrate COLMAP parser."""
    print("\n" + "=" * 80)
    print("2. COLMAP PARSER")
    print("=" * 80)

    from mogrammetry.colmap_parser import COLMAPParser, Camera, Image

    # Create synthetic COLMAP data
    temp_dir = Path(tempfile.mkdtemp(prefix='colmap_demo_'))

    # Create cameras.txt
    cameras_file = temp_dir / 'cameras.txt'
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with 3 camera models\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n")
        f.write("2 SIMPLE_RADIAL 1280 720 800.0 640.0 360.0 0.01\n")
        f.write("3 OPENCV 2048 1536 1200.0 1200.0 1024.0 768.0 0.01 0.001 0.0 0.0\n")

    # Create images.txt
    images_file = temp_dir / 'images.txt'
    with open(images_file, 'w') as f:
        f.write("# Image list with camera poses\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 image_0001.jpg\n")
        f.write("100.0 200.0 -1\n")
        f.write("2 0.9239 0.0 0.3827 0.0 1.5 0.0 0.0 1 image_0002.jpg\n")
        f.write("150.0 250.0 -1\n")
        f.write("3 0.7071 0.0 0.7071 0.0 3.0 0.0 0.0 2 image_0003.jpg\n")
        f.write("200.0 300.0 -1\n")

    # Parse
    print(f"\n✓ Created synthetic COLMAP data in: {temp_dir}")
    parser = COLMAPParser(str(temp_dir))
    cameras, images, points3D = parser.parse_all()

    print(f"\n✓ Parsed COLMAP reconstruction:")
    print(f"  - {len(cameras)} cameras")
    print(f"  - {len(images)} images")

    # Show camera details
    print("\nCamera Details:")
    for cam_id, cam in cameras.items():
        K = cam.get_intrinsic_matrix()
        print(f"  Camera {cam_id} ({cam.model}):")
        print(f"    - Resolution: {cam.width}x{cam.height}")
        print(f"    - Focal length: fx={cam.fx:.1f}, fy={cam.fy:.1f}")
        print(f"    - Principal point: cx={cam.cx:.1f}, cy={cam.cy:.1f}")

    # Show image poses
    print("\nImage Poses:")
    for img_id, img in images.items():
        R = img.get_rotation_matrix()
        center = img.get_camera_center()
        print(f"  Image {img_id} ({img.name}):")
        print(f"    - Camera: {img.camera_id}")
        print(f"    - Position: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"    - Rotation (euler): {np.rad2deg([np.arctan2(R[2,1], R[2,2]), np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)), np.arctan2(R[1,0], R[0,0])])}")

    # Validate
    warnings = parser.validate()
    if warnings:
        print(f"\n⚠ Validation warnings: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✓ No validation warnings - COLMAP data is valid!")

    # Cleanup
    shutil.rmtree(temp_dir)


def demo_alignment():
    """Demonstrate alignment solver."""
    print("\n" + "=" * 80)
    print("3. ALIGNMENT SOLVER")
    print("=" * 80)

    from mogrammetry.alignment import AlignmentSolver, align_points

    # Create synthetic affine-invariant points
    H, W = 100, 100
    points = np.random.randn(H, W, 3).astype(np.float32)
    points[:, :, 2] += 5.0  # Positive depth

    intrinsics = np.array([
        [1000.0, 0, 50.0],
        [0, 1000.0, 50.0],
        [0, 0, 1]
    ], dtype=np.float32)

    mask = np.ones((H, W), dtype=bool)

    print(f"\n✓ Created synthetic point cloud:")
    print(f"  - Shape: {points.shape}")
    print(f"  - Valid points: {np.sum(mask)}")
    print(f"  - Depth range: [{points[:,:,2].min():.2f}, {points[:,:,2].max():.2f}]")

    # Test different alignment methods
    methods = ['roe', 'ransac', 'least_squares']
    print(f"\nTesting {len(methods)} alignment methods:")

    for method in methods:
        solver = AlignmentSolver(method=method)
        scale, shift, stats = solver.solve(points, intrinsics, mask)

        print(f"\n  {method.upper()}:")
        print(f"    - Scale: {scale:.4f}")
        print(f"    - Shift: [{shift[0]:.4f}, {shift[1]:.4f}, {shift[2]:.4f}]")
        print(f"    - Valid points used: {stats.get('num_valid_points', 'N/A')}")

        # Apply alignment
        aligned = align_points(points, scale, shift)
        print(f"    - Aligned depth range: [{aligned[:,:,2].min():.2f}, {aligned[:,:,2].max():.2f}]")


def demo_fusion():
    """Demonstrate point cloud fusion."""
    print("\n" + "=" * 80)
    print("4. POINT CLOUD FUSION")
    print("=" * 80)

    from mogrammetry.fusion import PointCloudFusion, PointCloudData

    # Create 3 overlapping point clouds
    pc_data = []
    for i in range(3):
        points = np.random.randn(1000, 3).astype(np.float32) + i * 0.5
        colors = np.random.rand(1000, 3).astype(np.float32)
        pc = PointCloudData(
            points=points,
            colors=colors,
            source_image_id=i
        )
        pc_data.append(pc)

    print(f"\n✓ Created {len(pc_data)} point clouds:")
    for i, pc in enumerate(pc_data):
        print(f"  - Point cloud {i}: {len(pc.points)} points")

    # Fuse point clouds
    fusion = PointCloudFusion(
        voxel_size=0.1,
        outlier_removal='statistical',
        statistical_nb_neighbors=20,
        statistical_std_ratio=2.0
    )

    print("\nFusion settings:")
    print(f"  - Voxel size: {fusion.voxel_size}")
    print(f"  - Outlier removal: {fusion.outlier_removal}")
    print(f"  - Statistical neighbors: {fusion.statistical_nb_neighbors}")

    merged, stats = fusion.merge_point_clouds(pc_data, remove_outliers=True)

    print(f"\n✓ Fusion complete:")
    print(f"  - Input clouds: {stats['num_input_clouds']}")
    print(f"  - Total input points: {stats['total_input_points']}")
    print(f"  - After merge: {stats['points_after_merge']}")
    if 'points_after_outlier_removal' in stats:
        print(f"  - After outlier removal: {stats['points_after_outlier_removal']}")
    if 'points_after_voxel_downsample' in stats:
        print(f"  - After voxel downsampling: {stats['points_after_voxel_downsample']}")
    print(f"  - Final points: {stats['final_point_count']}")
    print(f"  - Reduction: {100 * (1 - stats['final_point_count'] / stats['total_input_points']):.1f}%")


def demo_mesh():
    """Demonstrate mesh generation."""
    print("\n" + "=" * 80)
    print("5. MESH GENERATION")
    print("=" * 80)

    from mogrammetry.mesh import MeshGenerator
    import open3d as o3d

    # Create synthetic point cloud with structure
    print("\n✓ Creating structured point cloud (sphere)...")
    pcd = o3d.geometry.PointCloud()

    # Create sphere points
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v)).flatten()
    y = np.outer(np.sin(u), np.sin(v)).flatten()
    z = np.outer(np.ones(len(u)), np.cos(v)).flatten()

    points = np.column_stack([x, y, z])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()

    print(f"  - Points: {len(pcd.points)}")
    print(f"  - Normals estimated: ✓")

    # Try ball pivoting (faster than Poisson for demo)
    generator = MeshGenerator(
        method='ball_pivoting',
        ball_pivoting_radii=[0.1, 0.2, 0.4]
    )

    print(f"\nMesh generation settings:")
    print(f"  - Method: {generator.method}")
    print(f"  - Ball pivoting radii: {generator.ball_pivoting_radii}")

    try:
        mesh, stats = generator.generate_mesh(pcd)
        print(f"\n✓ Mesh generated successfully:")
        print(f"  - Method: {stats['method']}")
        print(f"  - Vertices (before cleanup): {stats['vertices_before_cleanup']}")
        print(f"  - Faces (before cleanup): {stats['faces_before_cleanup']}")
        print(f"  - Vertices (after cleanup): {stats['vertices_after_cleanup']}")
        print(f"  - Faces (after cleanup): {stats['faces_after_cleanup']}")
    except RuntimeError as e:
        print(f"\n⚠ Ball pivoting failed on sphere (needs better density)")
        print(f"  In real use, Poisson reconstruction would work better")


def demo_logger():
    """Demonstrate logging system."""
    print("\n" + "=" * 80)
    print("6. LOGGING SYSTEM")
    print("=" * 80)

    from mogrammetry.logger import setup_logger, ProgressLogger

    # Setup logger
    logger = setup_logger(level='INFO', console=True, colorize=True)
    progress = ProgressLogger(logger)

    print("\n✓ Simulating pipeline workflow with progress tracking:\n")

    import time

    # Simulate tasks
    tasks = [
        ("Loading MoGe model", 0.2),
        ("Parsing COLMAP data", 0.1),
        ("Processing images (1/3)", 0.3),
        ("Processing images (2/3)", 0.3),
        ("Processing images (3/3)", 0.3),
        ("Merging point clouds", 0.2),
        ("Generating mesh", 0.4),
        ("Saving outputs", 0.1)
    ]

    for task_name, duration in tasks:
        progress.start_task(task_name)
        time.sleep(duration)
        progress.end_task(task_name)

    # Show summary
    summary = progress.get_summary()
    print(f"\nPipeline Summary:")
    print(f"  - Total time: {summary['total_time']:.2f}s")
    print(f"  - Tasks completed: {len(summary['tasks'])}")

    print("\nTask breakdown:")
    for task, info in summary['tasks'].items():
        print(f"  - {task}: {info['duration']:.2f}s")


def main():
    """Run all demos."""
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "MoGrammetry Demo" + " " * 37 + "║")
    print("║" + " " * 15 + "3D Reconstruction with MoGe + COLMAP" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        demo_configuration()
        demo_colmap_parser()
        demo_alignment()
        demo_fusion()
        demo_mesh()
        demo_logger()

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("\n✓ All components demonstrated successfully!")
        print("\nNext steps:")
        print("  1. Install torch for full pipeline: pip install torch torchvision")
        print("  2. Install MoGe: pip install moge")
        print("  3. Prepare COLMAP reconstruction + images")
        print("  4. Run: python scripts/mogrammetry_cli.py run <config.yaml>")
        print("\nFor more info, see: MOGRAMMETRY_README.md")

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
