from __future__ import annotations

import dataclasses
from typing import Tuple, cast

import torch
from torch import Tensor
from typing_extensions import override

from . import _base
from ._so3 import SO3
from .utils import broadcast_leading_axes, get_epsilon


def _skew(omega: Tensor) -> Tensor:
    """Returns the skew-symmetric form of a length-3 vector."""
    # omega: (..., 3)
    wx, wy, wz = torch.moveaxis(omega, -1, 0)
    zeros = torch.zeros_like(wx)
    return torch.stack(
        [zeros, -wz, wy,
         wz, zeros, -wx,
         -wy, wx, zeros],
        dim=-1,
    ).reshape((*omega.shape[:-1], 3, 3))


@dataclasses.dataclass(frozen=True)
class SE3(
    _base.SEBase[SO3],
    matrix_dim=4,
    parameters_dim=7,
    tangent_dim=6,
    space_dim=3,
):
    """Special Euclidean group for proper rigid transforms in 3D. Broadcasting
    rules follow PyTorch semantics."""

    wxyz_xyz: Tensor
    """Internal parameters: quaternion (w,x,y,z) + translation (x,y,z), shape (..., 7)."""

    @override
    def __repr__(self) -> str:
        quat = torch.round(self.wxyz_xyz[..., :4], decimals=5)
        trans = torch.round(self.wxyz_xyz[..., 4:], decimals=5)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={trans})"

    @classmethod
    @override
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        translation: Tensor,
    ) -> SE3:
        assert translation.shape[-1] == 3
        rotation, translation = broadcast_leading_axes((rotation, translation))
        return SE3(wxyz_xyz=torch.cat([rotation.wxyz, translation], dim=-1))

    @override
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @override
    def translation(self) -> Tensor:
        return self.wxyz_xyz[..., 4:]

    @classmethod
    @override
    def identity(
        cls, batch_axes: Tuple[int, ...] = (), dtype: torch.dtype = torch.float64, device: torch.device | None = None
    ) -> SE3:
        vec = torch.tensor([1.0, 0, 0, 0, 0, 0, 0], dtype=dtype, device=device)
        return SE3(
            wxyz_xyz=vec.broadcast_to((*batch_axes, 7))
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: Tensor) -> SE3:
        assert matrix.shape[-2:] in [(4, 4), (3, 4)]
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[..., :3, :3]),
            translation=matrix[..., :3, 3],
        )

    @override
    def as_matrix(self) -> Tensor:
        batch = self.get_batch_axes()
        dtype = self.wxyz_xyz.dtype
        device = self.wxyz_xyz.device
        out = torch.zeros((*batch, 4, 4), dtype=dtype, device=device)
        out[..., :3, :3] = self.rotation().as_matrix()
        out[..., :3, 3] = self.translation()
        out[..., 3, 3] = 1.0
        return out

    @override
    def parameters(self) -> Tensor:
        return self.wxyz_xyz

    @classmethod
    @override
    def exp(cls, tangent: Tensor) -> SE3:
        # tangent: (..., 6)
        assert tangent.shape[-1] == 6
        rotation = SO3.exp(tangent[..., 3:])

        theta2 = (tangent[..., 3:] ** 2).sum(dim=-1)
        use_taylor = theta2 < get_epsilon(theta2.dtype)

        # avoid zeros in denominator
        theta2_safe = torch.where(
            use_taylor, torch.ones_like(theta2), theta2
        )
        theta_safe = torch.sqrt(theta2_safe)

        skew_omega = _skew(tangent[..., 3:])
        eye3 = torch.eye(3, dtype=tangent.dtype, device=tangent.device)

        V = torch.where(
            use_taylor.unsqueeze(-1).unsqueeze(-1),
            rotation.as_matrix(),
            eye3
            + ((1 - torch.cos(theta_safe)) / theta2_safe).unsqueeze(-1).unsqueeze(-1) * skew_omega
            + ((theta_safe - torch.sin(theta_safe)) / (theta2_safe * theta_safe))
              .unsqueeze(-1).unsqueeze(-1)
              * torch.einsum("...ij,...jk->...ik", skew_omega, skew_omega)
        )

        trans = torch.einsum("...ij,...j->...i", V, tangent[..., :3]).to(tangent.dtype)
        return SE3.from_rotation_and_translation(rotation=rotation, translation=trans)

    @override
    def log(self) -> Tensor:
        omega = self.rotation().log()
        theta2 = (omega ** 2).sum(dim=-1)
        use_taylor = theta2 < get_epsilon(theta2.dtype)

        skew_omega = _skew(omega)
        theta2_safe = torch.where(use_taylor, torch.ones_like(theta2), theta2)
        theta_safe = torch.sqrt(theta2_safe)
        half_theta = theta_safe / 2

        eye3 = torch.eye(3, dtype=omega.dtype, device=omega.device)
        V_inv = torch.where(
            use_taylor.unsqueeze(-1).unsqueeze(-1),
            eye3 - 0.5 * skew_omega + torch.einsum("...ij,...jk->...ik", skew_omega, skew_omega) / 12,
            eye3
            - 0.5 * skew_omega
            + (((1 - theta_safe * torch.cos(half_theta) / torch.sin(half_theta)) / theta2_safe)
               .unsqueeze(-1).unsqueeze(-1)
               * torch.einsum("...ij,...jk->...ik", skew_omega, skew_omega))
        )

        lin = torch.einsum("...ij,...j->...i", V_inv, self.translation())
        return torch.cat([lin, omega], dim=-1).to(self.wxyz_xyz.dtype)

    @override
    def adjoint(self) -> Tensor:
        R = self.rotation().as_matrix()
        t = self.translation()
        skew_t = _skew(t)
        upper = torch.cat([R, torch.einsum("...ij,...jk->...ik", skew_t, R)], dim=-1)
        lower = torch.cat([torch.zeros((*self.get_batch_axes(), 3, 3), device=R.device, dtype=R.dtype), R], dim=-1)
        return torch.cat([upper, lower], dim=-2)

    @classmethod
    @override
    def sample_uniform(
        cls,
        rng: torch.Generator,
        batch_axes: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> SE3:
        rot = SO3.sample_uniform(rng, batch_axes=batch_axes, dtype=dtype, device=device)
        shape = (*batch_axes, 3)
        trans = torch.empty(*shape, dtype=dtype, device=device).uniform_(-1.0, 1.0, generator=rng)
        return SE3.from_rotation_and_translation(rotation=rot, translation=trans)
