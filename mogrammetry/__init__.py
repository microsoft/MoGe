"""
MoGrammetry: Integration of MoGe with COLMAP for Enhanced 3D Reconstruction

This package combines MoGe's accurate monocular geometry estimation with COLMAP's
robust multi-view Structure-from-Motion to create dense, high-quality 3D reconstructions.

Main components:
- pipeline: Core reconstruction pipeline
- alignment: Scale and shift alignment solvers
- fusion: Point cloud fusion and merging
- mesh: Mesh generation and texturing
- config: Configuration management
- utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "MoGrammetry Contributors"

# Lazy imports to avoid pulling in heavy dependencies (torch, open3d) at package import time.
# This allows lightweight modules (config, colmap_parser, logger) to be imported independently.
from .config import MoGrammetryConfig


def __getattr__(name):
    if name == 'MoGrammetryPipeline':
        from .pipeline import MoGrammetryPipeline
        return MoGrammetryPipeline
    raise AttributeError(f"module 'mogrammetry' has no attribute {name!r}")


__all__ = ['MoGrammetryPipeline', 'MoGrammetryConfig']
