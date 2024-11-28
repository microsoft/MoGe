## 2024-11-28
### Added
- Supported user-provided camera FOV. See [scripts/infer.py](scripts/infer.py) --fov_x. 
  - Related issues: [#25](https://github.com/microsoft/MoGe/issues/25) and [#24](https://github.com/microsoft/MoGe/issues/24).
- Added inference scripts for panorama images. See [scripts/infer_panorama.py](scripts/infer_panorama.py).
  - Related issue: [#19](https://github.com/microsoft/MoGe/issues/19).

### Fixed
- Suppressed unnecessary numpy runtime warnings.
- Speficied recommended versions of requirements.

### Changed
- Move `app.py` and `infer.py` to [scripts/](scripts/)
- Improved edge removal. 