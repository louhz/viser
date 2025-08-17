from typing import Union

import torch

# Type aliases Torch tensors; primarily for function inputs.

Scalar = Union[float, torch.Tensor]
"""Type alias for `Union[float, Tensor]`."""


__all__ = [
    "Scalar",
]