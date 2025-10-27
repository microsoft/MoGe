"""
Configuration management for MoGrammetry pipeline.
"""

from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml
import json


@dataclass
class AlignmentConfig:
    """Configuration for alignment solver."""
    method: Literal['roe', 'ransac', 'least_squares'] = 'roe'
    ransac_threshold: float = 0.1
    ransac_iterations: int = 1000
    truncation_threshold: float = 0.05
    min_valid_points: int = 100
    use_reprojection: bool = True
    optimize_focal: bool = False


@dataclass
class FusionConfig:
    """Configuration for point cloud fusion."""
    voxel_size: Optional[float] = None  # Auto if None
    outlier_removal: Literal['statistical', 'radius', 'both', 'none'] = 'statistical'
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    radius_nb_points: int = 16
    radius: float = 0.05
    max_points_per_image: Optional[int] = None  # No limit if None
    merge_strategy: Literal['append', 'average', 'weighted'] = 'weighted'
    overlap_threshold: float = 0.8  # For identifying overlapping regions


@dataclass
class MeshConfig:
    """Configuration for mesh generation."""
    method: Literal['poisson', 'ball_pivoting', 'alpha_shape'] = 'poisson'
    poisson_depth: int = 9
    poisson_width: float = 0.0
    poisson_scale: float = 1.1
    poisson_linear_fit: bool = False
    ball_pivoting_radii: list = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.04])
    alpha: float = 0.03
    texture_size: int = 4096
    texture_method: Literal['average', 'max_weight', 'mvs'] = 'mvs'
    simplify_mesh: bool = False
    target_faces: Optional[int] = None


@dataclass
class ProcessingConfig:
    """Configuration for processing options."""
    resolution_level: int = 9  # MoGe inference resolution
    batch_size: int = 1
    max_workers: int = 4
    use_gpu: bool = True
    device: str = 'cuda'
    fp16: bool = False
    cache_dir: Optional[str] = None
    resume_from_cache: bool = True
    save_intermediate: bool = False


@dataclass
class OutputConfig:
    """Configuration for output formats and options."""
    save_point_cloud: bool = True
    save_mesh: bool = True
    save_depth_maps: bool = False
    save_normal_maps: bool = False
    save_confidence_maps: bool = False
    formats: list = field(default_factory=lambda: ['ply', 'glb'])
    coordinate_system: Literal['colmap', 'opencv', 'opengl'] = 'colmap'
    export_cameras: bool = True
    export_report: bool = True


@dataclass
class MoGrammetryConfig:
    """Main configuration for MoGrammetry pipeline."""

    # Paths
    colmap_model_path: Optional[str] = None
    image_dir: Optional[str] = None
    output_dir: str = './output'
    model_name: str = 'Ruicheng/moge-vitl'

    # Sub-configurations
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Logging
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    log_file: Optional[str] = None
    progress_bar: bool = True
    verbose: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MoGrammetryConfig':
        """Create config from dictionary."""
        # Handle nested configs
        if 'alignment' in config_dict:
            config_dict['alignment'] = AlignmentConfig(**config_dict['alignment'])
        if 'fusion' in config_dict:
            config_dict['fusion'] = FusionConfig(**config_dict['fusion'])
        if 'mesh' in config_dict:
            config_dict['mesh'] = MeshConfig(**config_dict['mesh'])
        if 'processing' in config_dict:
            config_dict['processing'] = ProcessingConfig(**config_dict['processing'])
        if 'output' in config_dict:
            config_dict['output'] = OutputConfig(**config_dict['output'])

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MoGrammetryConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'MoGrammetryConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        # Check required paths
        if self.colmap_model_path is None:
            errors.append("colmap_model_path is required")
        elif not Path(self.colmap_model_path).exists():
            errors.append(f"COLMAP model path does not exist: {self.colmap_model_path}")

        if self.image_dir is None:
            errors.append("image_dir is required")
        elif not Path(self.image_dir).exists():
            errors.append(f"Image directory does not exist: {self.image_dir}")

        # Check ranges
        if not (0 <= self.processing.resolution_level <= 9):
            errors.append("resolution_level must be between 0 and 9")

        if self.fusion.voxel_size is not None and self.fusion.voxel_size <= 0:
            errors.append("voxel_size must be positive")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return True

    @classmethod
    def create_default(cls, colmap_model_path: str, image_dir: str, output_dir: str) -> 'MoGrammetryConfig':
        """Create a default configuration with required paths."""
        return cls(
            colmap_model_path=colmap_model_path,
            image_dir=image_dir,
            output_dir=output_dir
        )


def load_config(config_path: Optional[str] = None, **overrides) -> MoGrammetryConfig:
    """
    Load configuration from file or create default, with optional overrides.

    Args:
        config_path: Path to YAML or JSON config file (optional)
        **overrides: Key-value pairs to override in config

    Returns:
        MoGrammetryConfig instance
    """
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.suffix in ['.yaml', '.yml']:
            config = MoGrammetryConfig.from_yaml(str(config_path))
        elif config_path.suffix == '.json':
            config = MoGrammetryConfig.from_json(str(config_path))
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    else:
        config = MoGrammetryConfig()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try nested configs
            for nested_config_name in ['alignment', 'fusion', 'mesh', 'processing', 'output']:
                nested_config = getattr(config, nested_config_name)
                if hasattr(nested_config, key):
                    setattr(nested_config, key, value)
                    break

    return config
