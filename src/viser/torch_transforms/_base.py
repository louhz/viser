import abc
from typing import ClassVar, Generic, Tuple, TypeVar, Union, overload, cast
from typing_extensions import Self, final, get_args, override

import torch
from torch import Tensor


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    matrix_dim: ClassVar[int]
    """Dimension of square matrix output from `.as_matrix()`."""

    parameters_dim: ClassVar[int]
    """Dimension of underlying parameters, `.parameters()`."""

    tangent_dim: ClassVar[int]
    """Dimension of tangent space."""

    space_dim: ClassVar[int]
    """Dimension of coordinates that can be transformed."""

    def __init__(self, parameters: Tensor):
        """Construct a group object from its underlying parameters."""
        raise NotImplementedError()

    def __init_subclass__(
        cls,
        matrix_dim: int = 0,
        parameters_dim: int = 0,
        tangent_dim: int = 0,
        space_dim: int = 0,
    ) -> None:
        """Set class properties for subclasses. We default to dummy values."""
        cls.matrix_dim = matrix_dim
        cls.parameters_dim = parameters_dim
        cls.tangent_dim = tangent_dim
        cls.space_dim = space_dim

    # Shared implementations.

    @overload
    def __matmul__(self, other: Self) -> Self: ...
    @overload
    def __matmul__(self, other: Tensor) -> Tensor: ...

    def __matmul__(self, other: Union[Self, Tensor]) -> Union[Self, Tensor]:
        """Switches between the group action (`.apply()`) and multiplication
        (`.multiply()`) based on the type of `other`."""
        if isinstance(other, Tensor):
            return self.apply(target=other)
        elif isinstance(other, MatrixLieGroup):
            assert self.space_dim == other.space_dim
            return self.multiply(other=other)  # type: ignore
        else:
            raise TypeError(f"Invalid argument type for `@`: {type(other)}")

    # Factory.

    @classmethod
    @abc.abstractmethod
    def identity(
        cls, batch_axes: Tuple[int, ...] = (), dtype: torch.dtype = torch.float64
    ) -> Self:
        """Returns identity element."""

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: Tensor) -> Self:
        """Get group member from matrix representation."""

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> Tensor:
        """Get transformation as a matrix."""

    @abc.abstractmethod
    def parameters(self) -> Tensor:
        """Get underlying representation."""

    # Operations.

    @abc.abstractmethod
    def apply(self, target: Tensor) -> Tensor:
        """Applies group action to a point."""

    @abc.abstractmethod
    def multiply(self, other: Self) -> Self:
        """Composes this transformation with another."""

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: Tensor) -> Self:
        """Computes exponential map on tangent."""

    @abc.abstractmethod
    def log(self) -> Tensor:
        """Computes logarithm map from group to tangent."""

    @abc.abstractmethod
    def adjoint(self) -> Tensor:
        """Computes the adjoint matrix."""

    @abc.abstractmethod
    def inverse(self) -> Self:
        """Computes the inverse transformation."""

    @abc.abstractmethod
    def normalize(self) -> Self:
        """Normalize/projection for numerical stability."""

    @classmethod
    @abc.abstractmethod
    def sample_uniform(
        cls,
        rng: torch.Generator,
        batch_axes: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
    ) -> Self:
        """Draw a uniform sample from the group."""

    @final
    def get_batch_axes(self) -> Tuple[int, ...]:
        """Return any leading batch axes in contained parameters."""
        return self.parameters().shape[:-1]


class SOBase(MatrixLieGroup):
    """Base class for special orthogonal groups."""


ContainedSOType = TypeVar("ContainedSOType", bound=SOBase)


class SEBase(Generic[ContainedSOType], MatrixLieGroup):
    """Base class for special Euclidean groups."""

    @classmethod
    @abc.abstractmethod
    def from_rotation_and_translation(
        cls,
        rotation: ContainedSOType,
        translation: Tensor,
    ) -> Self:
        """Construct a rigid transform from rotation and translation."""

    @final
    @classmethod
    def from_rotation(cls, rotation: ContainedSOType) -> Self:
        return cls.from_rotation_and_translation(
            rotation=rotation,
            translation=torch.zeros(
                (*rotation.get_batch_axes(), cls.space_dim),
                dtype=rotation.parameters().dtype,
            ),
        )

    @final
    @classmethod
    def from_translation(cls, translation: Tensor) -> Self:
        # Extract rotation class from type parameter.
        assert len(cls.__orig_bases__) == 1  # type: ignore
        identity_cls = get_args(cls.__orig_bases__[0])[0]  # type: ignore
        return cls.from_rotation_and_translation(
            rotation=identity_cls.identity(),
            translation=translation,
        )

    @abc.abstractmethod
    def rotation(self) -> ContainedSOType:
        """Returns the rotation component."""

    @abc.abstractmethod
    def translation(self) -> Tensor:
        """Returns the translation component."""

    @final
    @override
    def apply(self, target: Tensor) -> Tensor:
        return self.rotation() @ target + self.translation()  # type: ignore

    @final
    @override
    def multiply(self, other: Self) -> Self:  # type: ignore
        return type(self).from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=(self.rotation() @ other.translation()) + self.translation(),
        )

    @final
    @override
    def inverse(self) -> Self:
        R_inv = self.rotation().inverse()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    @final
    @override
    def normalize(self) -> Self:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
        )