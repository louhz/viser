from __future__ import annotations

import dataclasses
from typing import Tuple, cast

import torch
from torch import Tensor
from typing_extensions import override

from . import _base, hints
from ._so2 import SO2
from .utils import broadcast_leading_axes, get_epsilon


@dataclasses.dataclass(frozen=True)
class SE2(
    _base.SEBase[SO2],
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=2,
):
    """Special Euclidean group for proper rigid transforms in 2D. Broadcasting
    rules follow torch semantics.

    Internal parameterization is `(cos, sin, x, y)`. Tangent parameterization is `(vx,
    vy, omega)`.
    """

    unit_complex_xy: Tensor
    """Internal parameters `(cos, sin, x, y)`, shape `(*, 4)`."""

    @override
    def __repr__(self) -> str:
        uc = torch.round(self.unit_complex_xy[..., :2], 5)
        xy = torch.round(self.unit_complex_xy[..., 2:], 5)
        return f"{self.__class__.__name__}(unit_complex={uc}, xy={xy})"

    @staticmethod
    def from_xy_theta(x: hints.Scalar, y: hints.Scalar, theta: hints.Scalar) -> SE2:
        """Construct from standard 2D pose."""
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        ucxy = torch.stack([cos, sin, x, y], dim=-1)
        return SE2(unit_complex_xy=ucxy)

    @classmethod
    @override
    def from_rotation_and_translation(
        cls,
        rotation: SO2,
        translation: Tensor,
    ) -> SE2:
        assert translation.shape[-1:] == (2,)
        rot, trans = broadcast_leading_axes((rotation, translation))
        ucxy = torch.cat([rot.unit_complex, trans], dim=-1)
        return SE2(unit_complex_xy=ucxy)

    @override
    def rotation(self) -> SO2:
        return SO2(unit_complex=self.unit_complex_xy[..., :2])

    @override
    def translation(self) -> Tensor:
        return self.unit_complex_xy[..., 2:]

    @classmethod
    @override
    def identity(
        cls, batch_axes: Tuple[int, ...] = (), dtype: torch.dtype = torch.float64
    ) -> SE2:
        vec = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
        ucxy = vec.expand(*batch_axes, 4)
        return SE2(unit_complex_xy=ucxy)

    @classmethod
    @override
    def from_matrix(cls, matrix: Tensor) -> SE2:
        assert matrix.shape[-2:] in [(3, 3), (2, 3)]
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[..., :2, :2]),
            translation=matrix[..., :2, 2],
        )

    @override
    def parameters(self) -> Tensor:
        return self.unit_complex_xy

    @override
    def as_matrix(self) -> Tensor:
        cos, sin, x, y = self.unit_complex_xy.movedim(-1, 0)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        stacked = torch.stack(
            [cos, -sin, x, sin, cos, y, zeros, zeros, ones],
            dim=-1
        )
        return stacked.view(*self.get_batch_axes(), 3, 3)

    @classmethod
    @override
    def exp(cls, tangent: Tensor) -> SE2:
        assert tangent.shape[-1:] == (3,)
        theta = tangent[..., 2]
        eps = get_epsilon(theta.dtype)
        use_taylor = torch.abs(theta) < eps
        safe_theta = torch.where(use_taylor, torch.ones_like(theta), theta)
        theta_sq = theta * theta

        sin_over_theta = torch.where(
            use_taylor,
            1.0 - theta_sq / 6.0,
            torch.sin(safe_theta) / safe_theta,
        )
        one_minus_cos_over_theta = torch.where(
            use_taylor,
            0.5 * theta - theta * theta_sq / 24.0,
            (1.0 - torch.cos(safe_theta)) / safe_theta,
        )

        V = torch.stack(
            [
                sin_over_theta,
                -one_minus_cos_over_theta,
                one_minus_cos_over_theta,
                sin_over_theta,
            ],
            dim=-1
        ).view(*tangent.shape[:-1], 2, 2)

        trans = torch.einsum("...ij,...j->...i", V, tangent[..., :2]).to(tangent.dtype)
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_radians(theta),
            translation=trans,
        )

    @override
    def log(self) -> Tensor:
        theta = self.rotation().log()[..., 0]
        cos = torch.cos(theta)
        cos_minus_one = cos - 1.0
        half_theta = theta / 2.0
        eps = get_epsilon(theta.dtype)
        use_taylor = torch.abs(cos_minus_one) < eps

        safe_cmo = torch.where(use_taylor, torch.ones_like(cos_minus_one), cos_minus_one)
        half_theta_over_tan_half = torch.where(
            use_taylor,
            1.0 - theta**2 / 12.0,
            -(half_theta * torch.sin(theta)) / safe_cmo,
        )

        V_inv = torch.stack(
            [
                half_theta_over_tan_half,
                half_theta,
                -half_theta,
                half_theta_over_tan_half,
            ],
            dim=-1
        ).view(*theta.shape, 2, 2)

        tangent = torch.cat(
            [
                torch.einsum("...ij,...j->...i", V_inv, self.translation()),
                theta.unsqueeze(-1),
            ],
            dim=-1
        )
        return tangent.to(self.unit_complex_xy.dtype)

    @override
    def adjoint(self) -> Tensor:
        cos, sin, x, y = self.unit_complex_xy.movedim(-1, 0)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        stacked = torch.stack(
            [cos, -sin, y, sin, cos, -x, zeros, zeros, ones],
            dim=-1
        )
        return stacked.view(*self.get_batch_axes(), 3, 3)

    @classmethod
    @override
    def sample_uniform(
        cls,
        rng: torch.Generator,
        batch_axes: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
    ) -> SE2:
        rot = SO2.sample_uniform(rng, batch_axes=batch_axes, dtype=dtype)
        trans = torch.empty(*batch_axes, 2, dtype=dtype).uniform_(-1.0, 1.0, generator=rng)
        return SE2.from_rotation_and_translation(rot, trans)
