"""
A package for common utility functions in 3D computer graphics and vision. Providing NumPy utilities in `utils3d.numpy`, PyTorch utilities in `utils3d.torch`, and IO utilities in `utils3d.io`.
"""
import importlib
from typing import TYPE_CHECKING

try:
    from ._unified import *
except ImportError:
    pass

__all__ = ['numpy', 'torch', 'io']

def __getattr__(name: str):
    return globals().get(name, importlib.import_module(f'.{name}', __package__)) 

if TYPE_CHECKING:
    from . import torch
    from . import numpy
    from . import io