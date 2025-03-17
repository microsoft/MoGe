## 2024-11-28
### Added
- Supported user-provided camera FOV. See [scripts/infer.py](scripts/infer.py) --fov_x. 
  - Related issues: [#25](https://github.com/microsoft/MoGe/issues/25) and [#24](https://github.com/microsoft/MoGe/issues/24).
- Added inference scripts for panorama images. See [scripts/infer_panorama.py](scripts/infer_panorama.py).
  - Related issue: [#19](https://github.com/microsoft/MoGe/issues/19).

### Fixed
- Suppressed unnecessary numpy runtime warnings.
- Specified recommended versions of requirements.
  - Related issue: [#21](https://github.com/microsoft/MoGe/issues/21).

### Changed
- Moved `app.py` and `infer.py` to [scripts/](scripts/)
- Improved edge removal. 

## 2025-03-18
### Added
- Training and evaluation code. See [docs/train.md](docs/train.md) and [docs/eval.md](docs/eval.md).
- Supported installation via pip. Thanks to @fabiencastan and @jgoueslard
 for commits in the [#47](https://github.com/microsoft/MoGe/pull/47)
- Supported command-line usage when installed.

### Changed
- Moved `scripts/` into `moge/` for package installation and command-line usage.
- Renamed `moge.model.moge_model` to `moge.model.v1` for version management. 
  Now you can import the model class through `from moge.model.v1 import MoGeModel` or `from moge.model import import_model_class_by_version; MoGeModel = import_model_class_by_version('v1')`.
- Exposed `num_tokens` parameter in MoGe model.