# MoGrammetry - Windows Installation Guide

## 🪟 Quick Start for Windows

### Prerequisites

1. **Python 3.8+** - [Download from python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
   - Verify: `python --version` in Command Prompt

2. **Git for Windows** - [Download](https://git-scm.com/download/win)
   - Verify: `git --version`

3. **NVIDIA GPU** (recommended) with CUDA support
   - Check: NVIDIA Control Panel → System Information
   - Download [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) if needed

---

## 📥 Step 1: Clone Your Repository

Open **Command Prompt** or **PowerShell**:

```bash
# Navigate to where you want the project
cd C:\Users\YourName\Documents

# Clone YOUR repository (not Microsoft's!)
git clone https://github.com/jenkinsm13/MoGrammetry.git
cd MoGrammetry

# Switch to the feature branch with all fixes
git checkout claude/develop-core-concept-011CUY7FQZkBnDwHyiQ1M4ub
```

---

## 🔧 Step 2: Install Dependencies

### Option A: With GPU Support (Recommended)

```bash
# Install PyTorch with CUDA support (for NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install MoGrammetry dependencies
pip install -r requirements.txt

# Install MoGe
pip install moge

# Install Open3D for 3D processing
pip install open3d

# Install additional packages
pip install click gradio pyyaml scipy trimesh pillow
```

### Option B: CPU Only (Slower)

```bash
# Install PyTorch CPU version
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
pip install moge open3d click gradio pyyaml scipy trimesh pillow
```

---

## ✅ Step 3: Verify Installation

```bash
# Run the test suite
python tests\test_mogrammetry.py
```

Expected output:
```
================================================================================
MoGrammetry Test Suite
================================================================================
Testing imports...
  ✓ All lightweight imports successful
  ✓ Pipeline import successful (torch available)

...

Passed: 7/7
✓ All tests passed!
```

---

## 🎨 Step 4: Run the Demo

```bash
# Run the comprehensive demo
python demo_mogrammetry.py
```

This will demonstrate all components without needing actual COLMAP data.

---

## 🚀 Step 5: Run Your First Reconstruction

### 5.1: Prepare Your Data

You need:
1. **COLMAP reconstruction** - folder with `cameras.txt`, `images.txt`, `points3D.txt`
2. **Original images** - the photos used for COLMAP

Example structure:
```
my_project/
├── colmap/
│   └── sparse/
│       └── 0/
│           ├── cameras.txt
│           ├── images.txt
│           └── points3D.txt
└── images/
    ├── IMG_0001.jpg
    ├── IMG_0002.jpg
    └── ...
```

### 5.2: Create Configuration

```bash
# Generate a config file
python scripts\mogrammetry_cli.py create-config my_config.yaml

# Edit my_config.yaml in Notepad or VS Code
notepad my_config.yaml
```

Update these paths (use Windows paths):
```yaml
colmap_model_path: C:/Users/YourName/my_project/colmap/sparse/0
image_dir: C:/Users/YourName/my_project/images
output_dir: C:/Users/YourName/output/reconstruction
```

### 5.3: Run Reconstruction

```bash
# Run the pipeline
python scripts\mogrammetry_cli.py run my_config.yaml
```

Or use command-line arguments:
```bash
python scripts\mogrammetry_cli.py run ^
    --colmap-model "C:\path\to\colmap\sparse\0" ^
    --image-dir "C:\path\to\images" ^
    --output "C:\path\to\output"
```

### 5.4: View Results

Your output folder will contain:
- `point_cloud.ply` - Dense 3D point cloud
- `mesh.ply` - 3D mesh
- `mesh.glb` - Web-friendly mesh format
- `reconstruction_report.json` - Statistics

**View with:**
- **CloudCompare** (free) - [Download](https://www.cloudcompare.org/)
- **MeshLab** (free) - [Download](https://www.meshlab.net/)
- **Blender** (free) - [Download](https://www.blender.org/)
- **Online GLB Viewer** - Upload `mesh.glb` to https://gltf-viewer.donmccurdy.com/

---

## 🌐 Alternative: Use Web Interface

```bash
# Start the Gradio web app
python scripts\app_mogrammetry.py
```

Then open your browser to: `http://localhost:7860`

Upload your COLMAP data + images through the web UI!

---

## 🔍 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"
Edit your config file:
```yaml
processing:
  resolution_level: 6  # Lower = less memory (0-9)
  device: cuda
```

Or use CPU:
```yaml
processing:
  device: cpu
```

### "No module named 'open3d'"
```bash
pip install open3d
```

### Import errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt
pip install moge open3d click gradio pyyaml scipy trimesh pillow
```

### COLMAP data not found
- Use **forward slashes** in paths: `C:/Users/...` not `C:\Users\...`
- Or use **double backslashes**: `C:\\Users\\...`
- Check that `cameras.txt`, `images.txt`, `points3D.txt` exist

---

## 📊 Performance Tips (Windows)

### GPU Acceleration
1. Update NVIDIA drivers to latest
2. Install CUDA Toolkit 11.8 or 12.x
3. Use PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Memory Management
- Close other programs while processing
- Use `resolution_level: 6` for 8GB GPU
- Use `resolution_level: 9` for 24GB GPU
- Enable `save_intermediate: false` to save disk space

### Faster Processing
Use the "fast" preset:
```bash
python scripts\mogrammetry_cli.py run my_config.yaml --preset fast
```

---

## 🎯 Example Commands for Windows

### Quick reconstruction (fast preset)
```batch
python scripts\mogrammetry_cli.py run ^
    --colmap-model "C:\data\colmap\sparse\0" ^
    --image-dir "C:\data\images" ^
    --output "C:\output\quick" ^
    --preset fast
```

### High-quality reconstruction
```batch
python scripts\mogrammetry_cli.py run ^
    --colmap-model "C:\data\colmap\sparse\0" ^
    --image-dir "C:\data\images" ^
    --output "C:\output\hq" ^
    --preset quality ^
    --save-mesh ^
    --save-point-cloud
```

### Check COLMAP data validity
```batch
python scripts\mogrammetry_cli.py validate "C:\data\colmap\sparse\0"
```

### Get system info
```batch
python scripts\mogrammetry_cli.py info
```

---

## 📁 Recommended Folder Structure (Windows)

```
C:\MoGrammetry\
├── MoGrammetry\          # Cloned repository
│   ├── mogrammetry\      # Python package
│   ├── scripts\          # CLI and web app
│   └── tests\            # Test suite
│
└── Projects\             # Your reconstruction projects
    ├── building\
    │   ├── colmap\sparse\0\
    │   ├── images\
    │   └── output\
    └── object\
        ├── colmap\sparse\0\
        ├── images\
        └── output\
```

---

## 🆘 Getting Help

If you encounter issues:

1. **Check test suite**: `python tests\test_mogrammetry.py`
2. **Run demo**: `python demo_mogrammetry.py`
3. **Validate COLMAP data**: `python scripts\mogrammetry_cli.py validate <path>`
4. **Check system**: `python scripts\mogrammetry_cli.py info`
5. **See examples**: Check `examples\` folder

---

## 🎓 Next Steps

1. ✅ Install Python and Git
2. ✅ Clone your repository
3. ✅ Install dependencies
4. ✅ Run tests to verify
5. ✅ Try the demo
6. 🎯 Run on your COLMAP data!

See `MOGRAMMETRY_README.md` for detailed usage and `EXAMPLE_OUTPUT.md` for what to expect.

---

## 💡 Pro Tips

- Use **VS Code** with Python extension for editing configs
- Enable **Windows Terminal** for better command-line experience
- Use **GPU-Z** to monitor GPU usage during processing
- Save presets in separate YAML files for different quality levels
- Use **PowerShell** instead of CMD for better path handling

Happy reconstructing! 🎉
