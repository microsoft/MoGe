# MoGrammetry Output Examples

This document shows what MoGrammetry produces when you run a reconstruction.

## 📁 Output Directory Structure

After running MoGrammetry on a COLMAP reconstruction, you get:

```
output/
├── point_cloud.ply           # Dense 3D point cloud (RGB colored)
├── mesh.ply                  # 3D mesh (Poisson/Ball Pivoting/Alpha Shape)
├── mesh.glb                  # 3D mesh in GLB format (for web viewers)
├── mesh.obj                  # 3D mesh in OBJ format
├── reconstruction_report.json # Detailed statistics
├── depth_maps/               # Individual depth maps per image (optional)
│   ├── image_0001_depth.npy
│   ├── image_0001_depth.png
│   ├── image_0002_depth.npy
│   └── ...
└── point_clouds/             # Per-image point clouds (optional)
    ├── image_0001.ply
    ├── image_0002.ply
    └── ...
```

## 📊 Reconstruction Report Example

The `reconstruction_report.json` contains comprehensive statistics:

```json
{
  "pipeline": {
    "timestamp": "2026-03-20T14:32:15",
    "total_time": 125.34,
    "config": {
      "model_name": "Ruicheng/moge-2-vitl-normal",
      "model_version": "v2",
      "alignment_method": "roe",
      "mesh_method": "poisson"
    }
  },
  "input": {
    "colmap_model_path": "colmap/sparse/0",
    "num_cameras": 3,
    "num_images": 24,
    "num_colmap_points": 12453
  },
  "moge_processing": {
    "images_processed": 24,
    "avg_time_per_image": 3.45,
    "resolution_level": 9,
    "total_time": 82.8
  },
  "alignment": {
    "method": "roe",
    "num_images_aligned": 24,
    "avg_scale": 0.1234,
    "scale_std": 0.0045,
    "valid_points_per_image": 98765,
    "alignment_time": 5.6
  },
  "fusion": {
    "input_clouds": 24,
    "total_input_points": 2370360,
    "points_after_merge": 2370360,
    "points_after_outlier_removal": 2145234,
    "points_after_voxel_downsample": 1456789,
    "final_point_count": 1456789,
    "reduction_ratio": 0.385,
    "voxel_size": 0.005,
    "outlier_removal_method": "statistical",
    "fusion_time": 18.3
  },
  "mesh": {
    "method": "poisson",
    "vertices_before_cleanup": 389456,
    "faces_before_cleanup": 778912,
    "vertices_after_cleanup": 345678,
    "faces_after_cleanup": 691234,
    "poisson_depth": 9,
    "has_vertex_normals": true,
    "has_vertex_colors": true,
    "mesh_time": 15.2
  },
  "output": {
    "point_cloud_saved": true,
    "mesh_saved": true,
    "formats": ["ply", "glb", "obj"],
    "point_cloud_size_mb": 87.5,
    "mesh_size_mb": 45.2
  }
}
```

## 🎨 Point Cloud Example

The point cloud (`point_cloud.ply`) contains:
- **1.4M - 2M points** (typical for 20-30 images)
- **RGB color** from source images
- **Vertex normals** (if requested)
- Can be viewed in:
  - CloudCompare
  - MeshLab
  - Open3D viewer
  - Blender

**Example statistics:**
```
Points: 1,456,789
Bounds: [-5.2, -3.1, -2.4] to [6.8, 4.2, 3.9] meters
Colors: RGB (0-255)
Normals: Yes
File size: ~87 MB
```

## 🎭 Mesh Example

The mesh (`mesh.ply`, `mesh.glb`, `mesh.obj`) contains:
- **345K vertices, 691K faces** (typical)
- **Textured with colors** from point cloud
- **Vertex normals** for smooth shading
- Can be viewed in:
  - Blender
  - MeshLab
  - Three.js web viewer (GLB)
  - Unity/Unreal (OBJ/GLB)

**Example statistics:**
```
Vertices: 345,678
Faces: 691,234
Vertex colors: RGB
Vertex normals: Yes
Texture: Per-vertex colors
Manifold: No (typical for Poisson)
Watertight: Yes (if Poisson)
File size: ~45 MB (PLY), ~38 MB (GLB)
```

## 🔍 Depth Maps Example

Individual depth maps (if `save_intermediate: true`):

**Format:** NumPy `.npy` + visualization `.png`

```python
# Load depth map
import numpy as np
depth = np.load('output/depth_maps/image_0001_depth.npy')
# Shape: (H, W, 3) - XYZ coordinates in camera space
# Values: Metric 3D points

# Depth range example
print(f"Min depth: {depth[:,:,2].min():.2f}m")  # 0.45m
print(f"Max depth: {depth[:,:,2].max():.2f}m")  # 12.34m
print(f"Valid points: {np.isfinite(depth[:,:,2]).sum()}")  # 98765
```

## 📈 Performance Benchmarks

**Typical processing times (NVIDIA RTX 3090):**

| Images | Resolution | MoGe Time | Alignment | Fusion | Mesh | Total |
|--------|-----------|-----------|-----------|---------|------|-------|
| 10     | 1920x1080 | 34s       | 2s        | 8s      | 6s   | 50s   |
| 25     | 1920x1080 | 86s       | 5s        | 18s     | 15s  | 124s  |
| 50     | 1920x1080 | 172s      | 12s       | 45s     | 38s  | 267s  |
| 100    | 1920x1080 | 345s      | 28s       | 120s    | 95s  | 588s  |

**Quality settings impact:**
- `fast`: ~2x faster, 85% quality
- `balanced`: 1x baseline, 95% quality
- `quality`: ~1.5x slower, 100% quality

## 🌐 Viewing the Results

### Option 1: Web Viewer (GLB)
```bash
# Upload mesh.glb to:
# - https://gltf-viewer.donmccurdy.com/
# - https://3dviewer.net/
# Or embed in Three.js
```

### Option 2: Blender
```bash
# File > Import > PLY/GLB/OBJ
# Point cloud: Import as PLY
# Mesh: Import as GLB (preserves colors)
```

### Option 3: Python (Open3D)
```python
import open3d as o3d

# View point cloud
pcd = o3d.io.read_point_cloud("output/point_cloud.ply")
o3d.visualization.draw_geometries([pcd])

# View mesh
mesh = o3d.io.read_triangle_mesh("output/mesh.ply")
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
```

### Option 4: CloudCompare
```bash
# Open point_cloud.ply or mesh.ply
# Automatic normal computation
# Measurement tools available
```

## 🎯 What Makes a Good Reconstruction?

**Point Cloud Quality Indicators:**
- Point count: 1M-5M points (depends on scene size)
- Coverage: No large gaps in important regions
- Noise level: Low (achieved via outlier removal)
- Color accuracy: Matches source images

**Mesh Quality Indicators:**
- Face count: 500K-2M faces (depends on scene detail)
- Manifoldness: Not required but preferred
- Surface smoothness: Depends on Poisson depth (9-10 = smooth)
- Texture fidelity: Colors match point cloud

## 🚀 Next Steps After Reconstruction

1. **Refine mesh** in Blender/MeshLab
2. **Create texture atlas** for game engines
3. **Simplify for web** (decimate to 50K-100K faces)
4. **Export to Unity/Unreal** for interactive viewing
5. **Use in AR/VR** applications

## 📝 Example Use Cases

- **Architecture**: Building reconstruction for BIM
- **Cultural Heritage**: Artifact digitization
- **Film/VFX**: Set reconstruction for CGI
- **Robotics**: Environment mapping for navigation
- **E-commerce**: Product 3D models
- **Gaming**: Asset creation from photos
