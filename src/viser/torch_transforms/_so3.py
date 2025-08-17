from __future__ import annotations

import dataclasses
from typing import NamedTuple, Tuple

import torch
from torch import Tensor
from typing_extensions import override

from . import _base, hints
from .utils import broadcast_leading_axes, get_epsilon


class RollPitchYaw(NamedTuple):
    """Struct containing roll, pitch, and yaw Euler angles."""
    roll: Tensor
    pitch: Tensor
    yaw: Tensor


@dataclasses.dataclass(frozen=True)
class SO3(
    _base.SOBase,
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=3,
):
    """Special orthogonal group for 3D rotations (PyTorch version)."""
    wxyz: Tensor  # shape (..., 4)

    @override
    def __repr__(self) -> str:
        wxyz = torch.round(self.wxyz, 5)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

    @staticmethod
    def from_x_radians(theta: hints.Scalar) -> SO3:
        zeros = torch.zeros_like(theta)
        return SO3.exp(torch.stack([theta, zeros, zeros], dim=-1))

    @staticmethod
    def from_y_radians(theta: hints.Scalar) -> SO3:
        zeros = torch.zeros_like(theta)
        return SO3.exp(torch.stack([zeros, theta, zeros], dim=-1))

    @staticmethod
    def from_z_radians(theta: hints.Scalar) -> SO3:
        zeros = torch.zeros_like(theta)
        return SO3.exp(torch.stack([zeros, zeros, theta], dim=-1))

    @staticmethod
    def from_rpy_radians(
        roll: hints.Scalar,
        pitch: hints.Scalar,
        yaw: hints.Scalar,
    ) -> SO3:
        # ZYX order: roll (X), then pitch (Y), then yaw (Z)
        return (
            SO3.from_z_radians(yaw)
            @ SO3.from_y_radians(pitch)
            @ SO3.from_x_radians(roll)
        )

    @staticmethod
    def from_quaternion_xyzw(xyzw: Tensor) -> SO3:
        assert xyzw.shape[-1] == 4
        # roll from xyzw → wxyz
        return SO3(wxyz=torch.roll(xyzw, shifts=1, dims=-1))

    def as_quaternion_xyzw(self) -> Tensor:
        return torch.roll(self.wxyz, shifts=-1, dims=-1)

    def as_rpy_radians(self) -> RollPitchYaw:
        return RollPitchYaw(
            roll=self.compute_roll_radians(),
            pitch=self.compute_pitch_radians(),
            yaw=self.compute_yaw_radians(),
        )

    def compute_roll_radians(self) -> Tensor:
        q0, q1, q2, q3 = self.wxyz.unbind(dim=-1)
        return torch.atan2(
            2 * (q0 * q1 + q2 * q3),
            1 - 2 * (q1 * q1 + q2 * q2),
        )

    def compute_pitch_radians(self) -> Tensor:
        q0, q1, q2, q3 = self.wxyz.unbind(dim=-1)
        return torch.arcsin(2 * (q0 * q2 - q3 * q1))

    def compute_yaw_radians(self) -> Tensor:
        q0, q1, q2, q3 = self.wxyz.unbind(dim=-1)
        return torch.atan2(
            2 * (q0 * q3 + q1 * q2),
            1 - 2 * (q2 * q2 + q3 * q3),
        )

    @classmethod
    @override
    def identity(
        cls,
        batch_axes: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> SO3:
        vec = torch.tensor([1.0, 0, 0, 0], dtype=dtype, device=device)
        return SO3(wxyz=vec.broadcast_to((*batch_axes, 4)))

    @classmethod
    @override
    def from_matrix(cls, M: Tensor) -> SO3:
        assert M.shape[-2:] == (3, 3)

        # Four-case method from Mike Day’s paper
        def _case0(m):
            t = 1 + m[...,0,0] - m[...,1,1] - m[...,2,2]
            q = torch.stack([
                m[...,2,1] - m[...,1,2],
                t,
                m[...,1,0] + m[...,0,1],
                m[...,0,2] + m[...,2,0],
            ], dim=-1)
            return t, q

        def _case1(m):
            t = 1 - m[...,0,0] + m[...,1,1] - m[...,2,2]
            q = torch.stack([
                m[...,0,2] - m[...,2,0],
                m[...,1,0] + m[...,0,1],
                t,
                m[...,2,1] + m[...,1,2],
            ], dim=-1)
            return t, q

        def _case2(m):
            t = 1 - m[...,0,0] - m[...,1,1] + m[...,2,2]
            q = torch.stack([
                m[...,1,0] - m[...,0,1],
                m[...,0,2] + m[...,2,0],
                m[...,2,1] + m[...,1,2],
                t,
            ], dim=-1)
            return t, q

        def _case3(m):
            t = 1 + m[...,0,0] + m[...,1,1] + m[...,2,2]
            q = torch.stack([
                t,
                m[...,2,1] - m[...,1,2],
                m[...,0,2] - m[...,2,0],
                m[...,1,0] - m[...,0,1],
            ], dim=-1)
            return t, q

        c0_t, c0_q = _case0(M)
        c1_t, c1_q = _case1(M)
        c2_t, c2_q = _case2(M)
        c3_t, c3_q = _case3(M)

        cond0 = M[...,2,2] < 0
        cond1 = M[...,0,0] > M[...,1,1]
        cond2 = M[...,0,0] < -M[...,1,1]

        t = torch.where(
            cond0,
            torch.where(cond1, c0_t, c1_t),
            torch.where(cond2, c2_t, c3_t),
        )
        q = torch.where(
            cond0.unsqueeze(-1),
            torch.where(cond1.unsqueeze(-1), c0_q, c1_q),
            torch.where(cond2.unsqueeze(-1), c2_q, c3_q),
        )
        wxyz = (q * 0.5 / torch.sqrt(t.unsqueeze(-1))).to(M.dtype)
        return SO3(wxyz=wxyz)

    @override
    def as_matrix(self) -> Tensor:
        # builds 3×3 from unit quaternion
        norm_sq = torch.sum(self.wxyz * self.wxyz, dim=-1, keepdim=True)
        q = self.wxyz * torch.sqrt(2.0 / norm_sq)
        qo = torch.einsum("...i,...j->...ij", q, q)
        return torch.stack([
            1 - qo[...,2,2] - qo[...,3,3],
            qo[...,1,2] - qo[...,3,0],
            qo[...,1,3] + qo[...,2,0],

            qo[...,1,2] + qo[...,3,0],
            1 - qo[...,1,1] - qo[...,3,3],
            qo[...,2,3] - qo[...,1,0],

            qo[...,1,3] - qo[...,2,0],
            qo[...,2,3] + qo[...,1,0],
            1 - qo[...,1,1] - qo[...,2,2],
        ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    @override
    def parameters(self) -> Tensor:
        return self.wxyz

    @override
    def apply(self, target: Tensor) -> Tensor:
        assert target.shape[-1] == 3
        self, target = broadcast_leading_axes((self, target))
        padded = torch.cat([
            torch.zeros(*self.get_batch_axes(), 1, device=target.device, dtype=target.dtype),
            target
        ], dim=-1)
        return (self @ SO3(wxyz=padded) @ self.inverse()).wxyz[...,1:]

    @override
    def multiply(self, other: SO3) -> SO3:
        w0, x0, y0, z0 = self.wxyz.unbind(dim=-1)
        w1, x1, y1, z1 = other.wxyz.unbind(dim=-1)
        q = torch.stack([
            -x0*x1 - y0*y1 - z0*z1 + w0*w1,
             x0*w1 + y0*z1 - z0*y1 + w0*x1,
            -x0*z1 + y0*w1 + z0*x1 + w0*y1,
             x0*y1 - y0*x1 + z0*w1 + w0*z1,
        ], dim=-1)
        return SO3(wxyz=q)

    @classmethod
    @override
    def exp(cls, tangent: Tensor) -> SO3:
        # tangent: (..., 3)
        theta2 = torch.sum(tangent * tangent, dim=-1)
        theta4 = theta2 * theta2
        use_t = theta2 < get_epsilon(theta2.dtype)

        safe_t2 = torch.where(use_t, torch.ones_like(theta2), theta2)
        safe_t = torch.sqrt(safe_t2)
        half = 0.5 * safe_t

        real = torch.where(
            use_t,
            1.0 - theta2/8.0 + theta4/384.0,
            torch.cos(half),
        )
        imag = torch.where(
            use_t,
            0.5 - theta2/48.0 + theta4/3840.0,
            torch.sin(half) / safe_t,
        )
        q = torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1) * tangent], dim=-1)
        return SO3(wxyz=q.to(tangent.dtype))

    @override
    def log(self) -> Tensor:
        w = self.wxyz[...,0]
        v = self.wxyz[...,1:]
        n2 = torch.sum(v*v, dim=-1)
        use_t = n2 < get_epsilon(n2.dtype)

        n_safe = torch.sqrt(torch.where(use_t, torch.ones_like(n2), n2))
        w_safe = torch.where(use_t, w, torch.ones_like(w))
        atan_nw = torch.atan2(torch.where(w<0, -n_safe, n_safe), w.abs())

        factor = torch.where(
            use_t,
            2.0 / w_safe - 2.0/3.0 * n2 / (w_safe**3),
            torch.where(
                w.abs() < get_epsilon(w.dtype),
                torch.sign(w).unsqueeze(-1) * (torch.pi/n_safe).unsqueeze(-1),
                (2.0 * atan_nw / n_safe).unsqueeze(-1),
            ).squeeze(-1),
        )
        return (factor.unsqueeze(-1) * v).to(self.wxyz.dtype)

    @override
    def adjoint(self) -> Tensor:
        return self.as_matrix()

    @override
    def inverse(self) -> SO3:
        inv = self.wxyz.clone()
        inv[...,1:] *= -1
        return SO3(wxyz=inv)

    @override
    def normalize(self) -> SO3:
        norm = torch.linalg.norm(self.wxyz, dim=-1, keepdim=True)
        return SO3(wxyz=self.wxyz / norm)

    @classmethod
    @override
    def sample_uniform(
        cls,
        rng: torch.Generator,
        batch_axes: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> SO3:
        # sample u1∈[0,1), u2,u3∈[0,2π)
        size = (*batch_axes, 3)
        u = torch.rand(size, generator=rng, dtype=dtype, device=device)
        u1, u2, u3 = u.unbind(dim=-1)
        u2 = u2 * (2*torch.pi)
        u3 = u3 * (2*torch.pi)
        a = torch.sqrt(1 - u1)
        b = torch.sqrt(u1)
        quat = torch.stack([
            a * torch.sin(u2),
            a * torch.cos(u2),
            b * torch.sin(u3),
            b * torch.cos(u3),
        ], dim=-1)
        return SO3(wxyz=quat)
