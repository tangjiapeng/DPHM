from typing import Optional, Union

import torch

from ..util.typing import to_base_tensor


class TorchIntrinsics(torch.Tensor):
    def __new__(cls,
                matrix_or_fx: Union[torch.Tensor, float] = torch.eye(3),
                fy: Optional[float] = None,
                cx: Optional[float] = None,
                cy: Optional[float] = None,
                s: Optional[float] = None,
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None):

        if not isinstance(matrix_or_fx, torch.Tensor) and not isinstance(matrix_or_fx, (float, int)):
            # Try to pack object into torch Tensor
            matrix_or_fx = torch.as_tensor(matrix_or_fx, device=device, dtype=dtype)

        if isinstance(matrix_or_fx, torch.Tensor) and device is None:
            # Use device of passed tensor
            device = matrix_or_fx.device

        if isinstance(matrix_or_fx, torch.Tensor) and dtype is None:
            # Use dtype of passed tensor
            dtype = matrix_or_fx.dtype

        if isinstance(matrix_or_fx, torch.Tensor) and matrix_or_fx.shape == (3, 3):
            assert fy is None and cx is None and cy is None and s is None, \
                "If a full intrinsics matrix is given, no other parameters should be specified!"
            # Have to use .to() call, as __new__ implicitly calls torch.Tensor which does not support device= keyword
            intrinsics = super().__new__(cls, matrix_or_fx).to(dtype).to(device)

        elif isinstance(matrix_or_fx, (float, int)) or isinstance(matrix_or_fx, torch.Tensor) and matrix_or_fx.shape == (1,):
            assert not (cx is None or cy is None), \
                "If a focal length is given, cx and cy have to be specified!"

            s = 0 if s is None else s
            fy = matrix_or_fx if fy is None else fy

            # Have to use .to() call, as __new__ implicitly calls torch.Tensor which does not support device= keyword
            intrinsics = super().__new__(cls, torch.zeros((3, 3))).to(dtype).to(device)
            if isinstance(matrix_or_fx, torch.Tensor) and matrix_or_fx.dtype == torch.float64:
                intrinsics = intrinsics.double()
            intrinsics[0, 0] = matrix_or_fx
            intrinsics[0, 1] = s
            intrinsics[0, 2] = cx
            intrinsics[1, 1] = fy
            intrinsics[1, 2] = cy
            intrinsics[2, 2] = 1
        else:
            print(matrix_or_fx, type(matrix_or_fx))
            raise ValueError("Either a full intrinsics matrix has to be given or fx, cx and cy")

        return intrinsics

    @property
    def fx(self) -> float:
        return self[0, 0]

    @property
    def fy(self) -> float:
        return self[1, 1]

    @property
    def cx(self) -> float:
        return self[0, 2]

    @property
    def cy(self) -> float:
        return self[1, 2]

    @property
    def s(self) -> float:
        return self[0, 1]

    def homogenize(self) -> torch.Tensor:
        homogenized = torch.eye(4)
        homogenized[:3, :3] = self
        return homogenized

    def tensor(self) -> torch.Tensor:
        return to_base_tensor(self)

    def __rmatmul__(self, other):
        assert not isinstance(other, TorchIntrinsics), "Matrix Multiplication between intrinsics does not make sense"
        return other @ to_base_tensor(self)

    def __matmul__(self, other):
        assert not isinstance(other, TorchIntrinsics), "Matrix Multiplication between intrinsics does not make sense"
        return to_base_tensor(self) @ other