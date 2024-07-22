from typing import Union, Tuple, List, Optional

import numpy as np
import torch

Vec4Type = Union[Tuple[float, float, float, float], np.ndarray, torch.Tensor, 'Vec4', 'TorchVec4']
Vec4TypeX = Union[float, Vec4Type]  # Can be either the first coordinate ('x') or a full vector
Vec3Type = Union[Tuple[float, float, float], np.ndarray, List[float], torch.Tensor, 'Vec3', 'TorchVec3']
Vec3TypeX = Union[float, Vec3Type]  # Can be either the first coordinate ('x') or a full vector
FloatType = Union[float, np.ndarray]  # Can be single element array (as a result of a numpy operation)


# ==========================================================
# Helper methods for dealing with heterogeneous input types
# ==========================================================

def unpack_nd_params(
        n: int,
        *args,
        default: Optional = None) -> Tuple:
    try:
        if len(args[0]) == n:
            assert all([args[i] is None for i in range(1, n)]), \
                "When a container is passed, all other values must not be given"
            components = args[0]
        elif len(args[0]) == n - 1:
            assert n == 2 or all([args[i] is None for i in range(2, n)]), \
                "When a container is passed, all other values must not be given"
            components = list(args[0]) + [args[1]]
        else:
            raise ValueError(f"Passed container must contain {n} or {n - 1} values!")
    except TypeError:
        # x was actually a number
        components = [default if arg is None else arg for arg in args]
    return components


def unpack_3d_params(
        x: Optional[Vec3TypeX],
        y: float,
        z: float,
        default: Optional = None) -> Tuple[float, float, float]:
    return unpack_nd_params(3, x, y, z, default=default)


def unpack_single(value: FloatType) -> float:
    if isinstance(value, np.ndarray):
        assert value.size == 1, f"Expected array with exactly one element! Got {value.size}"
        return value.item()
    else:
        return value
