#!/usr/bin/env python3
"""
Basic example of using MoGrammetry pipeline.

This script demonstrates the simplest way to run a reconstruction.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mogrammetry import MoGrammetryPipeline, MoGrammetryConfig


def main():
    # Create configuration with required paths
    config = MoGrammetryConfig(
        colmap_model_path='path/to/colmap/sparse/0',  # Change this
        image_dir='path/to/images',                    # Change this
        output_dir='output/basic_example',
        model_name='Ruicheng/moge-vitl'
    )

    # Optional: Customize settings
    config.processing.resolution_level = 9
    config.alignment.method = 'roe'
    config.mesh.method = 'poisson'
    config.output.save_mesh = True
    config.output.save_point_cloud = True
    config.output.formats = ['ply', 'glb']

    # Run pipeline
    print("Starting MoGrammetry reconstruction...")
    pipeline = MoGrammetryPipeline(config)
    stats = pipeline.run()

    # Print results
    print("\nReconstruction complete!")
    print(f"Processed {len(stats['processed_images'])} images")
    print(f"Output directory: {config.output_dir}")


if __name__ == '__main__':
    main()
