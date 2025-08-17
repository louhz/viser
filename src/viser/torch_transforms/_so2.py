from __future__ import annotations

import dataclasses
from typing import Tuple

import torch
from torch import Tensor
from typing_extensions import override

from . import _base, hints
from .utils import broadcast_leading_axes


@dataclasses.dataclass(frozen=True)
class SO2(
    _base.SOBase,
    matrix_dim=2,
    parameters_dim=2,
    tangent_dim=1,
    space_dim=2,
):
    """Special orthogonal group for 2D rotations. Broadcasting rules follow torch semantics.

    Internal parameterization is `(cos, sin)`. Tangent parameterization is `(omega,)`.
    """

    unit_complex: Tensor
    """Internal parameters `(cos, sin)`. Shape should be `(*, 2)`."""

    @override
    def __repr__(self) -> str:
        uc = torch.round(self.unit_complex, 5)
        return f"{self.__class__.__name__}(unit_complex={uc})"

    @staticmethod
    def from_radians(theta: hints.Scalar) -> SO2:
        """Construct a rotation object from a scalar angle."""
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        return SO2(unit_complex=torch.stack([cos, sin], dim=-1))

    def as_radians(self) -> Tensor:
        """Compute a scalar angle from a rotation object."""
        radians = self.log()[..., 0]
        return radians

    @classmethod
    @override
    def identity(
        cls, batch_axes: Tuple[int, ...] = (), dtype: torch.dtype = torch.float64
    ) -> SO2:
        ones = torch.ones(batch_axes, dtype=dtype)
        zeros = torch.zeros(batch_axes, dtype=dtype)
        uc = torch.stack([ones, zeros], dim=-1)
        return SO2(unit_complex=uc)

    @classmethod
    @override
    def from_matrix(cls, matrix: Tensor) -> SO2:
        assert matrix.shape[-2:] == (2, 2)
        # extract cos, sin from first column
        uc = matrix[..., :2, 0]
        return SO2(unit_complex=uc)

    @override
    def as_matrix(self) -> Tensor:
        uc = self.unit_complex
        cos = uc[..., 0]
        sin = uc[..., 1]
        # build [[cos, -sin], [sin, cos]]
        row1 = torch.stack([cos, -sin], dim=-1)
        row2 = torch.stack([sin, cos], dim=-1)
        mat = torch.stack([row1, row2], dim=-2)
        assert mat.shape == (*self.get_batch_axes(), 2, 2)
        return mat

    @override
    def parameters(self) -> Tensor:
        return self.unit_complex

    @override
    def apply(self, target: Tensor) -> Tensor:
        assert target.shape[-1] == 2
        self_b, target_b = broadcast_leading_axes((self, target))
        return torch.einsum("...ij,...j->...i", self_b.as_matrix(), target_b)

    @override
    def multiply(self, other: SO2) -> SO2:
        uc = torch.einsum("...ij,...j->...i", self.as_matrix(), other.unit_complex)
        return SO2(unit_complex=uc)

    @classmethod
    @override
    def exp(cls, tangent: Tensor) -> SO2:
        assert tangent.shape[-1] == 1
        cos = torch.cos(tangent)
        sin = torch.sin(tangent)
        uc = torch.cat([cos, sin], dim=-1)
        return SO2(unit_complex=uc)

    @override
    def log(self) -> Tensor:
        # arctan2(sin, cos)
        sin = self.unit_complex[..., 1]
        cos = self.unit_complex[..., 0]
        ang = torch.atan2(sin, cos)
        return ang.unsqueeze(-1)

    @override
    def adjoint(self) -> Tensor:
        # For SO2, adjoint is 1x1 identity
        ones = torch.ones(*self.get_batch_axes(), 1, 1, dtype=self.unit_complex.dtype)
        return ones

    @override
    def inverse(self) -> SO2:
        uc = self.unit_complex.clone()
        uc = uc * torch.tensor([1.0, -1.0], dtype=uc.dtype)
        return SO2(unit_complex=uc)

    @override
    def normalize(self) -> SO2:
        norm = torch.linalg.norm(self.unit_complex, dim=-1, keepdim=True)
        uc = self.unit_complex / norm
        return SO2(unit_complex=uc)

    @classmethod
    @override
    def sample_uniform(
        cls,
        rng: torch.Generator,
        batch_axes: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
    ) -> SO2:
        # uniform angle in [0, 2pi)
        shape = batch_axes
        angles = torch.rand(shape, generator=rng, dtype=dtype) * (2 * torch.pi)
        return SO2.from_radians(angles)
