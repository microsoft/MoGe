# MoGrammetry: Production-Ready 3D Reconstruction Pipeline

**MoGrammetry** is a complete, production-ready system that combines [MoGe](https://wangrc.site/MoGePage/)'s state-of-the-art monocular geometry estimation with [COLMAP](https://colmap.github.io/)'s robust Structure-from-Motion to create dense, high-quality 3D reconstructions.

## 🎯 Key Features

- **Dense Reconstruction**: Generate complete 3D models from images using MoGe's monocular depth estimation
- **Accurate Alignment**: Robust scale and shift recovery using ROE (Robust Outlier Estimation) solver
- **Multi-Strategy Fusion**: Intelligent point cloud merging with outlier removal and downsampling
- **Mesh Generation**: Create textured meshes using Poisson, Ball Pivoting, or Alpha Shape algorithms
- **Flexible Interface**: Use via command-line, Python API, or interactive web interface
- **Production Ready**: Comprehensive logging, error handling, and progress tracking
- **Extensible**: Modular architecture for easy customization and extension

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Web Interface](#web-interface)
- [Configuration](#configuration)
- [Pipeline Overview](#pipeline-overview)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## 🚀 Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0 with CUDA support (recommended)
- 8GB+ GPU VRAM (for large images)
- 16GB+ system RAM

### Install Dependencies

```bash
# Clone the repository (if not already done)
git clone https://github.com/microsoft/MoGe.git
cd MoGe

# Install core requirements
pip install -r requirements.txt

# Install additional dependencies for MoGrammetry
pip install gradio pyyaml
```

## ⚡ Quick Start

### 1. Prepare Your Data

You need:
- A COLMAP reconstruction (cameras.txt and images.txt)
- Source images used in the COLMAP reconstruction

```bash
your_project/
├── colmap/
│   └── sparse/
│       └── 0/
│           ├── cameras.txt
│           ├── images.txt
│           └── points3D.txt (optional)
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. Run Reconstruction

```bash
python scripts/mogrammetry_cli.py run \
    --colmap-model your_project/colmap/sparse/0 \
    --image-dir your_project/images \
    --output output/reconstruction \
    --save-mesh --save-point-cloud
```

### 3. View Results

Output directory will contain:
- `point_cloud.ply` - Dense 3D point cloud
- `mesh.ply` / `mesh.glb` - Textured 3D mesh
- `reconstruction_report.json` - Detailed statistics

View with:
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://www.cloudcompare.org/)
- [Blender](https://www.blender.org/)
- Any GLB viewer online

## 📖 Usage

### Command-Line Interface

The CLI provides the most comprehensive control over the pipeline.

#### Basic Usage

```bash
# Minimal command
python scripts/mogrammetry_cli.py run \
    --colmap-model <path> \
    --image-dir <path> \
    --output <path>

# With all options
python scripts/mogrammetry_cli.py run \
    --colmap-model colmap/sparse/0 \
    --image-dir images \
    --output output \
    --resolution-level 9 \
    --alignment-method roe \
    --mesh-method poisson \
    --outlier-removal statistical \
    --voxel-size 0.01 \
    --formats ply,glb,obj \
    --save-intermediate \
    --log-level INFO \
    --device cuda
```

#### Create Configuration File

```bash
# Create config with preset
python scripts/mogrammetry_cli.py create-config \
    --output config.yaml \
    --preset quality

# Use config file
python scripts/mogrammetry_cli.py run \
    --config config.yaml \
    --colmap-model <path> \
    --image-dir <path> \
    --output <path>
```

#### Validate COLMAP Data

```bash
python scripts/mogrammetry_cli.py validate colmap/sparse/0
```

#### System Information

```bash
python scripts/mogrammetry_cli.py info
```

### Python API

Use MoGrammetry directly in your Python code:

```python
from mogrammetry import MoGrammetryPipeline, MoGrammetryConfig

# Create configuration
config = MoGrammetryConfig(
    colmap_model_path='colmap/sparse/0',
    image_dir='images',
    output_dir='output',
    model_name='Ruicheng/moge-vitl'
)

# Customize settings
config.processing.resolution_level = 9
config.alignment.method = 'roe'
config.mesh.method = 'poisson'
config.output.save_mesh = True
config.output.save_point_cloud = True

# Run pipeline
pipeline = MoGrammetryPipeline(config)
stats = pipeline.run()

print(f"Generated {stats['fusion']['final_point_count']} points")
```

### Web Interface

Launch the interactive web interface:

```bash
python scripts/app_mogrammetry.py
```

Then open your browser to `http://localhost:7860`

Features:
- Upload COLMAP model and images as ZIP files
- Configure parameters with interactive controls
- View processing logs in real-time
- Download results directly

## ⚙️ Configuration

### Configuration Presets

- **default**: Balanced settings for general use
- **fast**: Faster processing (resolution_level=6, poisson_depth=7)
- **quality**: Highest quality (resolution_level=9, poisson_depth=10)
- **balanced**: Good trade-off

### Key Parameters

#### Processing
- `resolution_level` (0-9): MoGe inference resolution (higher = better, slower)
- `device`: 'cuda', 'cuda:0', or 'cpu'
- `save_intermediate`: Save per-image results for debugging

#### Alignment
- `method`: 'roe' (recommended), 'ransac', or 'least_squares'
- `use_reprojection`: Use reprojection error for alignment
- `truncation_threshold`: Threshold for robust estimation

#### Fusion
- `voxel_size`: Voxel size for downsampling (auto if None)
- `outlier_removal`: 'statistical', 'radius', 'both', or 'none'
- `merge_strategy`: 'weighted', 'average', or 'append'

#### Mesh
- `method`: 'poisson', 'ball_pivoting', or 'alpha_shape'
- `poisson_depth`: Octree depth for Poisson (7-10)
- `simplify_mesh`: Enable mesh simplification
- `target_faces`: Target face count for simplification

#### Output
- `formats`: List of output formats ['ply', 'glb', 'obj']
- `save_depth_maps`: Save individual depth maps
- `export_report`: Save detailed statistics

### Configuration File Example

```yaml
# config.yaml
colmap_model_path: colmap/sparse/0
image_dir: images
output_dir: output
model_name: Ruicheng/moge-vitl

processing:
  resolution_level: 9
  device: cuda
  save_intermediate: false

alignment:
  method: roe
  use_reprojection: true
  truncation_threshold: 0.05

fusion:
  voxel_size: null  # auto
  outlier_removal: statistical
  merge_strategy: weighted

mesh:
  method: poisson
  poisson_depth: 9
  simplify_mesh: false

output:
  save_point_cloud: true
  save_mesh: true
  formats:
    - ply
    - glb
  export_report: true

log_level: INFO
```

## 🔧 Pipeline Overview

### Step 1: COLMAP Parsing
- Reads cameras.txt and images.txt
- Supports all COLMAP camera models
- Validates data integrity

### Step 2: MoGe Inference
- Processes each image with MoGe model
- Generates affine-invariant point maps
- Predicts confidence masks

### Step 3: Alignment
- Solves for scale and shift using ROE
- Aligns predicted geometry with COLMAP intrinsics
- Uses reprojection error minimization

### Step 4: Transformation
- Converts camera-space points to world coordinates
- Uses COLMAP extrinsics (camera poses)
- Applies proper coordinate transformations

### Step 5: Fusion
- Merges point clouds from all images
- Removes statistical and radius outliers
- Applies voxel downsampling
- Estimates normals

### Step 6: Mesh Generation
- Constructs surface using selected method
- Removes degenerate triangles
- Optionally simplifies mesh
- Computes vertex normals

### Step 7: Export
- Saves point cloud and mesh
- Exports in multiple formats
- Generates reconstruction report

## 🎓 Advanced Features

### Custom Alignment Solver

```python
from mogrammetry.alignment import AlignmentSolver

solver = AlignmentSolver(
    method='roe',
    ransac_iterations=2000,
    truncation_threshold=0.03
)

scale, shift, stats = solver.solve(points, intrinsics, mask)
```

### Point Cloud Processing

```python
from mogrammetry.fusion import PointCloudFusion, PointCloudData

fusion = PointCloudFusion(
    voxel_size=0.01,
    outlier_removal='both'
)

merged, stats = fusion.merge_point_clouds(point_clouds)
```

### Custom Mesh Generation

```python
from mogrammetry.mesh import MeshGenerator

generator = MeshGenerator(
    method='poisson',
    poisson_depth=10,
    simplify_mesh=True,
    target_faces=100000
)

mesh, stats = generator.generate_mesh(point_cloud)
```

### Batch Processing

```python
import glob
from pathlib import Path

# Process multiple COLMAP models
colmap_models = glob.glob('projects/*/colmap/sparse/0')

for model_path in colmap_models:
    project_dir = Path(model_path).parent.parent.parent
    config = MoGrammetryConfig(
        colmap_model_path=model_path,
        image_dir=str(project_dir / 'images'),
        output_dir=str(project_dir / 'output')
    )
    pipeline = MoGrammetryPipeline(config)
    pipeline.run()
```

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory

**Problem**: CUDA out of memory error

**Solutions**:
- Reduce `resolution_level` (try 7 or 6)
- Process fewer images at once
- Use CPU: `--device cpu`
- Increase `voxel_size` for downsampling

#### Poor Alignment

**Problem**: Points don't align with COLMAP cameras

**Solutions**:
- Check COLMAP reconstruction quality
- Try different alignment method: `--alignment-method ransac`
- Verify image directory contains correct images
- Check camera model compatibility

#### Mesh Has Holes

**Problem**: Generated mesh has gaps

**Solutions**:
- Increase `poisson_depth` (try 10)
- Use more images with better coverage
- Try `ball_pivoting` method
- Reduce outlier removal aggressiveness

#### Slow Processing

**Problem**: Pipeline takes too long

**Solutions**:
- Lower `resolution_level` (7-8)
- Use `--preset fast` configuration
- Disable mesh generation: `--no-save-mesh`
- Skip intermediate saves

### Debug Mode

Enable detailed logging:

```bash
python scripts/mogrammetry_cli.py run \
    --log-level DEBUG \
    --log-file debug.log \
    --save-intermediate \
    ...
```

## 📚 API Reference

### Core Classes

#### `MoGrammetryConfig`
Configuration container for all pipeline settings.

Methods:
- `from_yaml(path)`: Load from YAML file
- `from_json(path)`: Load from JSON file
- `to_yaml(path)`: Save to YAML file
- `validate()`: Validate configuration

#### `MoGrammetryPipeline`
Main pipeline orchestrator.

Methods:
- `__init__(config)`: Initialize pipeline
- `run()`: Execute complete pipeline
- Returns: Statistics dictionary

#### `COLMAPParser`
Parser for COLMAP reconstruction files.

Methods:
- `parse_cameras()`: Parse cameras.txt
- `parse_images()`: Parse images.txt
- `parse_all()`: Parse all files
- `validate()`: Validate data

#### `AlignmentSolver`
Solves scale and shift for affine-invariant geometry.

Methods:
- `solve(points, intrinsics, mask)`: Solve alignment
- Returns: (scale, shift, stats)

#### `PointCloudFusion`
Merge and process multiple point clouds.

Methods:
- `merge_point_clouds(clouds)`: Merge point clouds
- Returns: (merged_pcd, stats)

#### `MeshGenerator`
Generate meshes from point clouds.

Methods:
- `generate_mesh(pcd)`: Generate mesh
- Returns: (mesh, stats)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Better texture mapping algorithms
- Support for more camera models
- Incremental reconstruction
- Multi-scale fusion
- GPU-accelerated fusion
- Better visualization tools

## 📄 License

MoGrammetry follows the same license as MoGe:
- Code: MIT License
- DINOv2 components: Apache 2.0 License

## 📞 Support

- **Issues**: Report bugs or request features on GitHub
- **Documentation**: Check this README and code comments
- **Examples**: See `examples/` directory

## 🙏 Acknowledgments

MoGrammetry builds upon:
- **MoGe** by Microsoft Research
- **COLMAP** for Structure-from-Motion
- **Open3D** for 3D processing
- **Trimesh** for mesh operations

## 📖 Citation

If you use MoGrammetry in your research, please cite:

```bibtex
@misc{wang2024moge,
    title={MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision},
    author={Wang, Ruicheng and Xu, Sicheng and Dai, Cassie and Xiang, Jianfeng and Deng, Yu and Tong, Xin and Yang, Jiaolong},
    year={2024},
    eprint={2410.19115},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2410.19115},
}
```

---

**MoGrammetry** - Making 3D reconstruction production-ready. 🚀
