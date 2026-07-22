@echo off
REM MoGrammetry Windows Setup Script
REM Run this after cloning the repository

echo ========================================
echo MoGrammetry Windows Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [OK] Python found
python --version

echo.
echo Installing dependencies...
echo.

REM Detect NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [INFO] NVIDIA GPU not detected, installing CPU version
    pip install torch torchvision
) else (
    echo [OK] NVIDIA GPU detected, installing with CUDA support
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo Installing MoGrammetry dependencies...
pip install -r requirements.txt

echo.
echo Installing additional packages...
pip install moge open3d click gradio pyyaml scipy trimesh pillow

echo.
echo ========================================
echo Running tests...
echo ========================================
python tests\test_mogrammetry.py

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run demo: python demo_mogrammetry.py
echo 2. Create config: python scripts\mogrammetry_cli.py create-config my_config.yaml
echo 3. Run reconstruction: python scripts\mogrammetry_cli.py run my_config.yaml
echo.
echo See WINDOWS_SETUP.md for detailed instructions
echo.
pause
