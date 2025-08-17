from typing import TYPE_CHECKING, Tuple, TypeVar, Union, cast

import torch

if TYPE_CHECKING:
    from .._base import MatrixLieGroup


T = TypeVar("T", bound="MatrixLieGroup")


def get_epsilon(dtype: torch.dtype) -> float:
    """Helper for grabbing type-specific precision constants.

    Args:
        dtype: Datatype.

    Returns:
        Output float.
    """
    if dtype == torch.float32:
        return 1e-5
    elif dtype == torch.float64:
        return 1e-10
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


TupleOfBroadcastable = TypeVar(
    "TupleOfBroadcastable",
    bound="Tuple[Union[MatrixLieGroup, torch.Tensor], ...]",
)


def broadcast_leading_axes(inputs: TupleOfBroadcastable) -> TupleOfBroadcastable:
    """Broadcast leading axes of tensors or MatrixLieGroup objects.
    Takes tuples of either:
      - a torch.Tensor of shape (*, D)
      - a MatrixLieGroup object
    Returns a tuple with each element broadcasted to a common batch shape.
    """
    from .._base import MatrixLieGroup

    # Extract raw tensors and their "trailing dims" (parameter dims or tensor's last dim)
    raw_and_suffix = []
    for x in inputs:
        if isinstance(x, MatrixLieGroup):
            params = x.parameters()               # Tensor of shape (*, D)
            suffix = (x.parameters_dim(),)        # trailing dims
            raw_and_suffix.append((params, suffix, x.__class__))
        else:
            suffix = x.shape[-1:]
            raw_and_suffix.append((x, suffix, None))

    # Verify suffix consistency
    for arr, suffix, _ in raw_and_suffix:
        assert arr.shape[-len(suffix):] == suffix, \
            f"Expected trailing dims {suffix}, got {arr.shape[-len(suffix):]}"

    # Compute broadcasted batch shape
    batch_shapes = [arr.shape[: -len(suffix)] for arr, suffix, _ in raw_and_suffix]
    common_batch = torch.broadcast_shapes(*batch_shapes)

    # Broadcast each tensor to (common_batch + suffix)
    broadcasted = [
        torch.broadcast_to(arr, common_batch + suffix)
        for arr, suffix, _ in raw_and_suffix
    ]

    # Re-wrap into MatrixLieGroup if needed
    out: Tuple[Union[MatrixLieGroup, torch.Tensor], ...] = tuple(
        cls(b) if cls is not None else b
        for (b, (_, _, cls)) in zip(broadcasted, raw_and_suffix)
    )
    return cast(TupleOfBroadcastable, out)
