from typing import Optional

import torch

from .vector_base import Vec3TypeX, unpack_3d_params, unpack_nd_params


class TorchVec3(torch.Tensor):
    def __new__(cls,
                x: Optional[Vec3TypeX] = None,
                y: Optional[float] = None,
                z: Optional[float] = None) -> 'TorchVec3':
        x, y, z = unpack_3d_params(x, y, z, default=0)
        vec3 = super().__new__(cls, (x, y, z))

        return vec3

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    @x.setter
    def x(self, x: float):
        self[0] = x

    @y.setter
    def y(self, y: float):
        self[1] = y

    @z.setter
    def z(self, z: float):
        self[2] = z

    def length(self):
        return torch.linalg.norm(self)

    def normalize(self) -> 'TorchVec3':
        return self / self.length()

    def homogenize(self) -> 'TorchVec4':
        return TorchVec4(self, 1)

    def tensor(self) -> torch.Tensor:
        return torch.Tensor(self)


class TorchVec4(torch.Tensor):
    def __new__(cls,
                x: Vec3TypeX,
                y: Optional[float] = None,
                z: Optional[float] = None,
                w: Optional[float] = None) -> 'TorchVec4':
        # TODO: Allow no arguments -> 0 vector
        x, y, z, w = unpack_nd_params(4, x, y, z, w)
        vec3 = super().__new__(cls, (x, y, z, w))

        return vec3

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    @property
    def w(self) -> float:
        return self[3]

    @x.setter
    def x(self, x: float):
        self[0] = x

    @y.setter
    def y(self, y: float):
        self[1] = y

    @z.setter
    def z(self, z: float):
        self[2] = z

    @w.setter
    def w(self, w: float):
        self[3] = w

    def length(self):
        return torch.linalg.norm(self)

    def normalize(self) -> 'TorchVec4':
        return self / self.length()

    def tensor(self) -> torch.Tensor:
        return torch.Tensor(self)