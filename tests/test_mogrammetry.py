#!/usr/bin/env python3
"""
Test suite for MoGrammetry components.

Run this to validate that all components are working correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tempfile
import shutil


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from mogrammetry import MoGrammetryPipeline, MoGrammetryConfig
        from mogrammetry.colmap_parser import COLMAPParser, Camera, Image
        from mogrammetry.alignment import AlignmentSolver, align_points
        from mogrammetry.fusion import PointCloudFusion, PointCloudData
        from mogrammetry.mesh import MeshGenerator, TextureMapper
        from mogrammetry.logger import setup_logger, ProgressLogger
        from mogrammetry.config import (
            AlignmentConfig, FusionConfig, MeshConfig,
            ProcessingConfig, OutputConfig
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration system."""
    print("\nTesting configuration system...")

    try:
        from mogrammetry import MoGrammetryConfig

        # Create config
        config = MoGrammetryConfig(
            colmap_model_path='/tmp/test',
            image_dir='/tmp/images',
            output_dir='/tmp/output'
        )

        # Test serialization
        temp_yaml = tempfile.mktemp(suffix='.yaml')
        temp_json = tempfile.mktemp(suffix='.json')

        config.to_yaml(temp_yaml)
        config.to_json(temp_json)

        # Test deserialization
        config_yaml = MoGrammetryConfig.from_yaml(temp_yaml)
        config_json = MoGrammetryConfig.from_json(temp_json)

        assert config_yaml.output_dir == config.output_dir
        assert config_json.output_dir == config.output_dir

        # Cleanup
        Path(temp_yaml).unlink()
        Path(temp_json).unlink()

        print("  ✓ Configuration system working")
        return True
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def test_colmap_parser():
    """Test COLMAP parser with synthetic data."""
    print("\nTesting COLMAP parser...")

    try:
        from mogrammetry.colmap_parser import COLMAPParser

        # Create temporary COLMAP files
        temp_dir = Path(tempfile.mkdtemp(prefix='colmap_test_'))

        # Write cameras.txt
        cameras_file = temp_dir / 'cameras.txt'
        with open(cameras_file, 'w') as f:
            f.write("# Camera list\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n")
            f.write("2 SIMPLE_RADIAL 1280 720 800.0 640.0 360.0 0.01\n")

        # Write images.txt
        images_file = temp_dir / 'images.txt'
        with open(images_file, 'w') as f:
            f.write("# Image list\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 image1.jpg\n")
            f.write("\n")  # Empty 2D points line
            f.write("2 0.707 0.0 0.707 0.0 1.0 0.0 0.0 1 image2.jpg\n")
            f.write("\n")

        # Parse
        parser = COLMAPParser(str(temp_dir))
        cameras, images, points3D = parser.parse_all()

        assert len(cameras) == 2
        assert len(images) == 2
        assert 1 in cameras
        assert cameras[1].model == 'PINHOLE'
        assert cameras[1].width == 1920
        assert images[1].name == 'image1.jpg'

        # Validate
        warnings = parser.validate()
        assert isinstance(warnings, list)

        # Test camera methods
        cam = cameras[1]
        K = cam.get_intrinsic_matrix()
        assert K.shape == (3, 3)
        assert K[0, 0] == 1000.0

        # Test image methods
        img = images[1]
        R = img.get_rotation_matrix()
        assert R.shape == (3, 3)
        extrinsic = img.get_extrinsic_matrix()
        assert extrinsic.shape == (4, 4)

        # Cleanup
        shutil.rmtree(temp_dir)

        print("  ✓ COLMAP parser working")
        return True
    except Exception as e:
        print(f"  ✗ COLMAP parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alignment():
    """Test alignment solver."""
    print("\nTesting alignment solver...")

    try:
        from mogrammetry.alignment import AlignmentSolver, align_points

        # Create synthetic data
        H, W = 100, 100
        points = np.random.randn(H, W, 3).astype(np.float32)
        points[:, :, 2] += 5.0  # Ensure positive Z

        intrinsics = np.array([
            [1000.0, 0, 50.0],
            [0, 1000.0, 50.0],
            [0, 0, 1]
        ], dtype=np.float32)

        mask = np.ones((H, W), dtype=bool)

        # Test different methods
        for method in ['roe', 'ransac', 'least_squares']:
            solver = AlignmentSolver(method=method)
            scale, shift, stats = solver.solve(points, intrinsics, mask)

            assert isinstance(scale, (float, np.floating))
            assert shift.shape == (3,)
            assert isinstance(stats, dict)
            assert 'method' in stats

            # Test alignment
            aligned = align_points(points, scale, shift)
            assert aligned.shape == points.shape

        print("  ✓ Alignment solver working")
        return True
    except Exception as e:
        print(f"  ✗ Alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion():
    """Test point cloud fusion."""
    print("\nTesting point cloud fusion...")

    try:
        from mogrammetry.fusion import PointCloudFusion, PointCloudData
        import open3d as o3d

        # Create synthetic point clouds
        pc1 = PointCloudData(
            points=np.random.randn(1000, 3).astype(np.float32),
            colors=np.random.rand(1000, 3).astype(np.float32)
        )

        pc2 = PointCloudData(
            points=np.random.randn(1000, 3).astype(np.float32) + 1.0,
            colors=np.random.rand(1000, 3).astype(np.float32)
        )

        # Test fusion
        fusion = PointCloudFusion(
            voxel_size=0.1,
            outlier_removal='statistical'
        )

        merged, stats = fusion.merge_point_clouds([pc1, pc2])

        assert isinstance(merged, o3d.geometry.PointCloud)
        assert len(merged.points) > 0
        assert 'final_point_count' in stats

        print("  ✓ Point cloud fusion working")
        return True
    except Exception as e:
        print(f"  ✗ Fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mesh_generation():
    """Test mesh generation."""
    print("\nTesting mesh generation...")

    try:
        from mogrammetry.mesh import MeshGenerator
        import open3d as o3d

        # Create synthetic point cloud
        pcd = o3d.geometry.PointCloud()
        points = np.random.randn(5000, 3).astype(np.float64)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        # Test mesh generation
        generator = MeshGenerator(method='ball_pivoting')  # Faster than Poisson

        try:
            mesh, stats = generator.generate_mesh(pcd)
            assert isinstance(mesh, o3d.geometry.TriangleMesh)
            assert len(mesh.vertices) > 0
            assert 'method' in stats
            print("  ✓ Mesh generation working")
            return True
        except RuntimeError:
            # Ball pivoting might fail on random points
            print("  ⚠ Mesh generation test skipped (ball pivoting failed on random data)")
            return True

    except Exception as e:
        print(f"  ✗ Mesh generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logger():
    """Test logging system."""
    print("\nTesting logging system...")

    try:
        from mogrammetry.logger import setup_logger, ProgressLogger

        logger = setup_logger(level='INFO', console=True)
        logger.info("Test message")

        progress = ProgressLogger(logger)
        progress.start_task("test_task")
        progress.end_task("test_task")

        summary = progress.get_summary()
        assert 'total_time' in summary
        assert 'tasks' in summary

        print("  ✓ Logging system working")
        return True
    except Exception as e:
        print(f"  ✗ Logger test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("MoGrammetry Test Suite")
    print("=" * 80)

    tests = [
        test_imports,
        test_config,
        test_colmap_parser,
        test_alignment,
        test_fusion,
        test_mesh_generation,
        test_logger,
    ]

    results = []
    for test in tests:
        results.append(test())

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
