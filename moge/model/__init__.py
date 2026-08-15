import importlib
from typing import *

if TYPE_CHECKING:
    from .v1 import MoGeModel as MoGeModelV1
    from .v2 import MoGeModel as MoGeModelV2
    from .v3 import MoGeModel as MoGeModelV3


def import_model_class_by_version(version: str) -> Type[Union['MoGeModelV1', 'MoGeModelV2', 'MoGeModelV3']]:
    assert version in ['v1', 'v2', 'v3'], f'Unsupported model version: {version}'
    
    try:
        module = importlib.import_module(f'.{version}', __package__)
    except ModuleNotFoundError:
        raise ValueError(f'Model version "{version}" not found.')

    cls = getattr(module, 'MoGeModel')
    return cls
