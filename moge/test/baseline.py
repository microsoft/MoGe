from typing import *

import click
import torch


class MGEBaselineInterface:
    """
    Abstract class for model wrapper to uniformize the interface of loading and inference across different models.
    """
    device: torch.device

    @click.command()
    @staticmethod
    def load(*args, **kwargs) -> "MGEBaselineInterface":
        """
        Customized static method to create an instance of the model wrapper from command line arguments. Decorated by `click.command()`
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented the load method.")       

    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        ### Parameters
            `image`: [B, 3, H, W] or [3, H, W],  RGB values in range [0, 1]
            `intrinsics`: [B, 3, 3] or [3, 3], camera intrinsics. Optional.
        
        ### Returns
            A dictionary containing:
            - `points_*`. point map output in OpenCV identity camera space.
                Supported suffixes: `metric`, `scale_invariant`, `affine_invariant`.
            - `depth_*`. depth map output
                Supported suffixes: `metric` (in meters), `scale_invariant`, `affine_invariant`.
            - `disparity_affine_invariant`. affine disparity map output
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented the infer method.")
    
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        If the model has a special evaluation mode, override this method to provide the evaluation mode inference.

        By default, this method simply calls `infer()`.
        """
        return self.infer(image, intrinsics)