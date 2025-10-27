#!/usr/bin/env python3
"""
MoGrammetry Gradio Web Interface

Interactive web interface for running MoGrammetry 3D reconstruction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import tempfile
import shutil
import zipfile
from typing import Tuple, Optional
import numpy as np
import open3d as o3d

from mogrammetry import MoGrammetryPipeline, MoGrammetryConfig


def process_reconstruction(
    colmap_zip: gr.File,
    images_zip: gr.File,
    resolution_level: int,
    alignment_method: str,
    mesh_method: str,
    outlier_removal: str,
    save_mesh: bool,
    save_point_cloud: bool
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Process reconstruction from uploaded files.

    Returns:
        point_cloud_file: Path to point cloud file
        mesh_file: Path to mesh file
        log: Processing log
    """
    log_messages = []

    def log(msg):
        log_messages.append(msg)
        print(msg)

    try:
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix='mogrammetry_'))
        log(f"Created temporary directory: {temp_dir}")

        # Extract COLMAP model
        colmap_dir = temp_dir / 'colmap'
        colmap_dir.mkdir()

        if colmap_zip is not None:
            log("Extracting COLMAP model...")
            with zipfile.ZipFile(colmap_zip, 'r') as zip_ref:
                zip_ref.extractall(colmap_dir)
            log(f"✓ Extracted COLMAP model")
        else:
            log("✗ No COLMAP model provided")
            return None, None, "\n".join(log_messages)

        # Extract images
        images_dir = temp_dir / 'images'
        images_dir.mkdir()

        if images_zip is not None:
            log("Extracting images...")
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            log(f"✓ Extracted images")
        else:
            log("✗ No images provided")
            return None, None, "\n".join(log_messages)

        # Find COLMAP model directory (might be nested)
        possible_model_dirs = list(colmap_dir.rglob('cameras.txt'))
        if not possible_model_dirs:
            log("✗ Could not find cameras.txt in COLMAP archive")
            return None, None, "\n".join(log_messages)

        colmap_model_path = possible_model_dirs[0].parent
        log(f"Found COLMAP model: {colmap_model_path}")

        # Output directory
        output_dir = temp_dir / 'output'
        output_dir.mkdir()

        # Create configuration
        log("\nCreating configuration...")
        config = MoGrammetryConfig(
            colmap_model_path=str(colmap_model_path),
            image_dir=str(images_dir),
            output_dir=str(output_dir),
            model_name='Ruicheng/moge-vitl',
            log_level='INFO',
            progress_bar=False
        )

        config.processing.resolution_level = resolution_level
        config.alignment.method = alignment_method
        config.mesh.method = mesh_method
        config.fusion.outlier_removal = outlier_removal
        config.output.save_mesh = save_mesh
        config.output.save_point_cloud = save_point_cloud
        config.output.formats = ['ply', 'glb']

        log("✓ Configuration created")

        # Run pipeline
        log("\n" + "=" * 80)
        log("Starting MoGrammetry Pipeline")
        log("=" * 80 + "\n")

        pipeline = MoGrammetryPipeline(config)
        stats = pipeline.run()

        log("\n" + "=" * 80)
        log("Pipeline Complete")
        log("=" * 80)

        # Get output files
        point_cloud_file = None
        mesh_file = None

        if save_point_cloud and (output_dir / 'point_cloud.ply').exists():
            point_cloud_file = str(output_dir / 'point_cloud.ply')
            log(f"\n✓ Point cloud saved: {point_cloud_file}")

        if save_mesh and (output_dir / 'mesh.glb').exists():
            mesh_file = str(output_dir / 'mesh.glb')
            log(f"✓ Mesh saved: {mesh_file}")

        log("\n✓ Reconstruction complete!")

        return point_cloud_file, mesh_file, "\n".join(log_messages)

    except Exception as e:
        log(f"\n✗ Error: {str(e)}")
        import traceback
        log("\nTraceback:")
        log(traceback.format_exc())
        return None, None, "\n".join(log_messages)


# Create Gradio interface
with gr.Blocks(title="MoGrammetry - 3D Reconstruction", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # MoGrammetry: Enhanced 3D Reconstruction

    Combine MoGe's monocular geometry estimation with COLMAP's multi-view reconstruction
    to create dense, high-quality 3D models.

    ## How to use:
    1. Upload your COLMAP model files (cameras.txt, images.txt) as a ZIP
    2. Upload your source images as a ZIP
    3. Configure processing parameters
    4. Click "Run Reconstruction"
    5. Download the results!
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Files")

            colmap_zip = gr.File(
                label="COLMAP Model (ZIP)",
                file_types=[".zip"],
                type="filepath"
            )

            gr.Markdown("""
            Upload a ZIP containing:
            - `cameras.txt`
            - `images.txt`
            - (optional) `points3D.txt`
            """)

            images_zip = gr.File(
                label="Images (ZIP)",
                file_types=[".zip"],
                type="filepath"
            )

            gr.Markdown("Upload a ZIP containing all source images.")

            gr.Markdown("### Parameters")

            resolution_level = gr.Slider(
                minimum=0,
                maximum=9,
                value=9,
                step=1,
                label="Resolution Level",
                info="Higher = better quality but slower (0-9)"
            )

            alignment_method = gr.Dropdown(
                choices=['roe', 'ransac', 'least_squares'],
                value='roe',
                label="Alignment Method",
                info="Method for scale recovery"
            )

            mesh_method = gr.Dropdown(
                choices=['poisson', 'ball_pivoting', 'alpha_shape'],
                value='poisson',
                label="Mesh Method",
                info="Surface reconstruction algorithm"
            )

            outlier_removal = gr.Dropdown(
                choices=['statistical', 'radius', 'both', 'none'],
                value='statistical',
                label="Outlier Removal",
                info="Method for removing outlier points"
            )

            with gr.Row():
                save_point_cloud = gr.Checkbox(
                    value=True,
                    label="Save Point Cloud"
                )
                save_mesh = gr.Checkbox(
                    value=True,
                    label="Save Mesh"
                )

            run_button = gr.Button("Run Reconstruction", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### Results")

            log_output = gr.Textbox(
                label="Processing Log",
                lines=20,
                max_lines=30,
                interactive=False
            )

            with gr.Row():
                point_cloud_output = gr.File(
                    label="Point Cloud (PLY)",
                    interactive=False
                )

                mesh_output = gr.File(
                    label="Mesh (GLB)",
                    interactive=False
                )

            gr.Markdown("""
            ### Next Steps:
            - Download the PLY file to view in CloudCompare, MeshLab, or Blender
            - Download the GLB file to view in online 3D viewers or import into game engines
            - The reconstruction report contains detailed statistics
            """)

    # Connect the button to the processing function
    run_button.click(
        fn=process_reconstruction,
        inputs=[
            colmap_zip,
            images_zip,
            resolution_level,
            alignment_method,
            mesh_method,
            outlier_removal,
            save_mesh,
            save_point_cloud
        ],
        outputs=[point_cloud_output, mesh_output, log_output]
    )

    gr.Markdown("""
    ---
    ## About MoGrammetry

    MoGrammetry combines:
    - **MoGe**: State-of-the-art monocular geometry estimation
    - **COLMAP**: Robust Structure-from-Motion camera alignment

    This hybrid approach produces dense 3D reconstructions that are:
    - ✓ More complete than traditional multi-view stereo
    - ✓ More accurate than pure learning-based methods
    - ✓ Faster than running dense MVS pipelines

    ### System Requirements:
    - GPU with 8GB+ VRAM recommended
    - 16GB+ system RAM
    - Modern web browser

    ### Supported COLMAP Formats:
    - Text format (cameras.txt, images.txt)
    - All standard camera models (PINHOLE, SIMPLE_RADIAL, OPENCV, etc.)

    For more information, see the [MoGe paper](https://arxiv.org/abs/2410.19115).
    """)


if __name__ == '__main__':
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
