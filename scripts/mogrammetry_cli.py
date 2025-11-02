#!/usr/bin/env python3
"""
MoGrammetry Command-Line Interface

A comprehensive CLI for running MoGrammetry 3D reconstruction pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from mogrammetry import MoGrammetryPipeline, MoGrammetryConfig


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    MoGrammetry: Integration of MoGe with COLMAP for Enhanced 3D Reconstruction

    Combine MoGe's accurate monocular geometry estimation with COLMAP's robust
    multi-view Structure-from-Motion to create dense, high-quality 3D reconstructions.
    """
    pass


@cli.command()
@click.option(
    '--colmap-model',
    type=click.Path(exists=True),
    required=True,
    help='Path to COLMAP model directory (containing cameras.txt and images.txt)'
)
@click.option(
    '--image-dir',
    type=click.Path(exists=True),
    required=True,
    help='Path to directory containing input images'
)
@click.option(
    '--output',
    type=click.Path(),
    required=True,
    help='Path to output directory'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default=None,
    help='Path to configuration file (YAML or JSON)'
)
@click.option(
    '--model',
    type=str,
    default='Ruicheng/moge-vitl',
    help='MoGe model name or path'
)
@click.option(
    '--resolution-level',
    type=click.IntRange(0, 9),
    default=9,
    help='MoGe inference resolution level (0-9, higher is better but slower)'
)
@click.option(
    '--alignment-method',
    type=click.Choice(['roe', 'ransac', 'least_squares']),
    default='roe',
    help='Alignment method for affine-invariant geometry'
)
@click.option(
    '--mesh-method',
    type=click.Choice(['poisson', 'ball_pivoting', 'alpha_shape']),
    default='poisson',
    help='Mesh reconstruction method'
)
@click.option(
    '--outlier-removal',
    type=click.Choice(['statistical', 'radius', 'both', 'none']),
    default='statistical',
    help='Outlier removal method'
)
@click.option(
    '--voxel-size',
    type=float,
    default=None,
    help='Voxel size for downsampling (auto if not specified)'
)
@click.option(
    '--save-mesh/--no-save-mesh',
    default=True,
    help='Generate and save mesh'
)
@click.option(
    '--save-point-cloud/--no-save-point-cloud',
    default=True,
    help='Save point cloud'
)
@click.option(
    '--formats',
    type=str,
    default='ply,glb',
    help='Output formats (comma-separated: ply, glb, obj)'
)
@click.option(
    '--save-intermediate',
    is_flag=True,
    help='Save intermediate results for debugging'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Logging level'
)
@click.option(
    '--log-file',
    type=click.Path(),
    default=None,
    help='Path to log file'
)
@click.option(
    '--device',
    type=str,
    default='cuda',
    help='Device for inference (cuda, cuda:0, cpu)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Validate configuration without running pipeline'
)
def run(
    colmap_model,
    image_dir,
    output,
    config,
    model,
    resolution_level,
    alignment_method,
    mesh_method,
    outlier_removal,
    voxel_size,
    save_mesh,
    save_point_cloud,
    formats,
    save_intermediate,
    log_level,
    log_file,
    device,
    dry_run
):
    """
    Run MoGrammetry reconstruction pipeline.

    This command processes a COLMAP reconstruction along with source images
    to generate a dense 3D point cloud and mesh using MoGe's monocular
    geometry estimation.

    Example:

        mogrammetry run \\
            --colmap-model ./colmap/sparse/0 \\
            --image-dir ./images \\
            --output ./output \\
            --save-mesh --save-point-cloud
    """
    # Load or create configuration
    if config:
        cfg = MoGrammetryConfig.from_yaml(config) if config.endswith(('.yaml', '.yml')) \
              else MoGrammetryConfig.from_json(config)

        # Override with command-line arguments
        cfg.colmap_model_path = colmap_model
        cfg.image_dir = image_dir
        cfg.output_dir = output
    else:
        # Create config from command-line arguments
        cfg = MoGrammetryConfig(
            colmap_model_path=colmap_model,
            image_dir=image_dir,
            output_dir=output,
            model_name=model,
            log_level=log_level,
            log_file=log_file
        )

        # Apply CLI overrides
        cfg.processing.resolution_level = resolution_level
        cfg.processing.device = device
        cfg.processing.save_intermediate = save_intermediate

        cfg.alignment.method = alignment_method

        cfg.mesh.method = mesh_method

        cfg.fusion.outlier_removal = outlier_removal
        cfg.fusion.voxel_size = voxel_size

        cfg.output.save_mesh = save_mesh
        cfg.output.save_point_cloud = save_point_cloud
        cfg.output.formats = [f.strip() for f in formats.split(',')]

    # Validate configuration
    try:
        cfg.validate()
        click.echo(click.style("✓ Configuration validated successfully", fg='green'))
    except ValueError as e:
        click.echo(click.style(f"✗ Configuration error: {e}", fg='red'), err=True)
        sys.exit(1)

    # Dry run: just validate and show config
    if dry_run:
        click.echo("\nConfiguration:")
        click.echo(f"  COLMAP model: {cfg.colmap_model_path}")
        click.echo(f"  Image directory: {cfg.image_dir}")
        click.echo(f"  Output directory: {cfg.output_dir}")
        click.echo(f"  Model: {cfg.model_name}")
        click.echo(f"  Alignment: {cfg.alignment.method}")
        click.echo(f"  Mesh method: {cfg.mesh.method}")
        click.echo(f"  Output formats: {', '.join(cfg.output.formats)}")
        click.echo("\nDry run complete. Use --no-dry-run to execute.")
        return

    # Run pipeline
    try:
        click.echo(click.style("\nStarting MoGrammetry pipeline...\n", fg='cyan', bold=True))
        pipeline = MoGrammetryPipeline(cfg)
        stats = pipeline.run()

        click.echo(click.style("\n✓ Pipeline completed successfully!", fg='green', bold=True))
        click.echo(f"\nOutput saved to: {output}")

    except Exception as e:
        click.echo(click.style(f"\n✗ Pipeline failed: {e}", fg='red', bold=True), err=True)
        if log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    '--output',
    type=click.Path(),
    required=True,
    help='Path to save configuration file'
)
@click.option(
    '--format',
    type=click.Choice(['yaml', 'json']),
    default='yaml',
    help='Configuration file format'
)
@click.option(
    '--preset',
    type=click.Choice(['default', 'fast', 'quality', 'balanced']),
    default='default',
    help='Configuration preset'
)
def create_config(output, format, preset):
    """
    Create a configuration file with sensible defaults.

    Presets:
      - default: Balanced settings for general use
      - fast: Faster processing with lower quality
      - quality: Highest quality, slower processing
      - balanced: Good trade-off between speed and quality

    Example:

        mogrammetry create-config --output config.yaml --preset quality
    """
    # Create config based on preset
    if preset == 'fast':
        cfg = MoGrammetryConfig()
        cfg.processing.resolution_level = 6
        cfg.mesh.poisson_depth = 7
        cfg.fusion.outlier_removal = 'none'
    elif preset == 'quality':
        cfg = MoGrammetryConfig()
        cfg.processing.resolution_level = 9
        cfg.mesh.poisson_depth = 10
        cfg.fusion.outlier_removal = 'both'
        cfg.mesh.simplify_mesh = False
    elif preset == 'balanced':
        cfg = MoGrammetryConfig()
        cfg.processing.resolution_level = 8
        cfg.mesh.poisson_depth = 9
        cfg.fusion.outlier_removal = 'statistical'
    else:  # default
        cfg = MoGrammetryConfig()

    # Save configuration
    if format == 'yaml':
        cfg.to_yaml(output)
    else:
        cfg.to_json(output)

    click.echo(click.style(f"✓ Configuration saved to: {output}", fg='green'))
    click.echo(f"  Preset: {preset}")
    click.echo("\nEdit this file to customize settings, then use:")
    click.echo(f"  mogrammetry run --config {output} --colmap-model <path> --image-dir <path> --output <path>")


@cli.command()
@click.argument('colmap_model', type=click.Path(exists=True))
def validate(colmap_model):
    """
    Validate COLMAP model files.

    Checks that cameras.txt and images.txt are properly formatted and
    contain valid data.

    Example:

        mogrammetry validate ./colmap/sparse/0
    """
    from mogrammetry.colmap_parser import COLMAPParser

    click.echo(f"Validating COLMAP model: {colmap_model}\n")

    try:
        parser = COLMAPParser(colmap_model)
        cameras, images, points3D = parser.parse_all()

        click.echo(click.style("✓ COLMAP model parsed successfully", fg='green'))
        click.echo(f"\n  Cameras: {len(cameras)}")
        click.echo(f"  Images: {len(images)}")
        if points3D is not None:
            click.echo(f"  3D points: {len(points3D)}")

        # Check for warnings
        warnings = parser.validate()
        if warnings:
            click.echo(click.style(f"\n⚠ {len(warnings)} warnings:", fg='yellow'))
            for warning in warnings:
                click.echo(f"  - {warning}")
        else:
            click.echo(click.style("\n✓ No validation warnings", fg='green'))

        # Show camera details
        click.echo("\nCamera details:")
        for cam_id, cam in cameras.items():
            click.echo(f"  Camera {cam_id}: {cam.model} {cam.width}x{cam.height}")
            click.echo(f"    f: {cam.fx:.2f}, {cam.fy:.2f}  c: {cam.cx:.2f}, {cam.cy:.2f}")

    except Exception as e:
        click.echo(click.style(f"✗ Validation failed: {e}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
def info():
    """
    Display information about MoGrammetry and system.
    """
    import torch
    import open3d
    import trimesh

    click.echo(click.style("MoGrammetry System Information\n", fg='cyan', bold=True))

    click.echo("Version: 1.0.0")
    click.echo("\nDependencies:")
    click.echo(f"  PyTorch: {torch.__version__}")
    click.echo(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"  CUDA version: {torch.version.cuda}")
        click.echo(f"  GPU: {torch.cuda.get_device_name(0)}")
    click.echo(f"  Open3D: {open3d.__version__}")
    click.echo(f"  Trimesh: {trimesh.__version__}")

    click.echo("\nFor more information, visit:")
    click.echo("  https://github.com/microsoft/MoGe")


if __name__ == '__main__':
    cli()
