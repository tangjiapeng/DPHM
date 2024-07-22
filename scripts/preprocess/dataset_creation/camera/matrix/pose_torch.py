from typing import Union, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

# ==========================================================
# Torch Matrices
# ==========================================================
from .pose_base import is_rotation_matrix
from ..util.typing import to_base_tensor
from ..vector.vector_base import Vec3TypeX, unpack_3d_params, Vec3Type
from ..vector.vector_numpy import Vec4
from ..vector.vector_torch import TorchVec3, TorchVec4


class TorchPose(torch.Tensor):

    def __new__(cls,
                matrix_or_rotation: Union[torch.Tensor, np.ndarray, List] = torch.eye(4),
                translation: Optional[torch.Tensor] = None):
        use_double = isinstance(matrix_or_rotation, torch.Tensor) and matrix_or_rotation.dtype == torch.float64

        def _cast(tensor: torch.Tensor):
            if use_double:
                return tensor.double()
            else:
                return tensor.float()

        if not isinstance(matrix_or_rotation, torch.Tensor):
            matrix_or_rotation = torch.asarray(matrix_or_rotation, dtype=torch.float32)

        if translation is not None and not isinstance(translation, torch.Tensor):
            translation = torch.asarray(translation, dtype=torch.float32)

        device = matrix_or_rotation.device

        if matrix_or_rotation.shape == (4, 4):
            # Full 4x4 Pose
            assert translation is None, "If a full pose is given, no translation should be specified!"
            assert (matrix_or_rotation[3, :] == torch.tensor([0, 0, 0, 1])).all(), \
                f"Last row of pose must be [0, 0, 0, 1]. Got {matrix_or_rotation[3, :]}"
            assert is_rotation_matrix(matrix_or_rotation[:3, :3]), \
                f"Specified matrix does not contain a valid rotation matrix! {matrix_or_rotation[:3, :3]}"

            pose = super().__new__(cls, matrix_or_rotation).to(device)
            pose = _cast(pose)

        elif matrix_or_rotation.shape == (3, 3):
            translation = translation.squeeze()
            # 3x3 rotation matrix + 3(x1) translation vector
            assert translation.shape == (3,), \
                "If a rotation matrix is given, the translation vector has to be 3 dimensional!"

            pose = super().__new__(cls, torch.zeros((4, 4))).to(device)
            pose = _cast(pose)

            pose[:3, :3] = _cast(matrix_or_rotation)
            pose[:3, 3] = _cast(translation)
            pose[3, 3] = 1
        elif matrix_or_rotation.shape == (3, 1) or matrix_or_rotation.shape == (3,):
            # 3(x1) Rodriguez vector + 3(x1) translation vector
            translation = translation.squeeze()
            assert translation.squeeze().shape == (3,), \
                "If a Rodriguez vector is given, the translation vector has to be 3 dimensional!"

            pose = super().__new__(cls, torch.zeros((4, 4))).to(device)
            pose = _cast(pose)

            # TODO: Continue
            pose[:3, :3] = angle_axis_to_rotation_matrix(matrix_or_rotation.unsqueeze(0))[0, :3, :3]
            pose[:3, 3] = _cast(translation)
            pose[3, 3] = 1
        else:
            raise ValueError("Either a full pose has to be given or a 3x3 rotation + 3x1 translation!")

        return pose

    @staticmethod
    def from_euler(
            euler_angles: Vec3Type,
            translation: Vec3Type = TorchVec3(),
            euler_mode: str = 'XYZ') -> 'TorchPose':
        return TorchPose(R.from_euler(euler_mode, euler_angles).as_matrix(), translation)

    @staticmethod
    def from_cv(cv_rotation_vector: TorchVec3, translation: TorchVec3) -> 'TorchPose':
        return TorchPose(cv_rotation_vector, translation)

    def get_rotation_matrix(self) -> torch.Tensor:
        return self[:3, :3]

    # def get_euler_angles(self, order: str) -> TorchVec3:
    #     return Vec3(R.from_matrix(self.get_rotation_matrix()).as_euler(order))

    def get_rodriguez_vector(self) -> TorchVec3:
        return TorchVec3(rotation_matrix_to_angle_axis(self[:3, :].unsqueeze(0))[0])

    # def get_quaternion(self) -> TorchVec4:
    #     return Vec4(R.from_matrix(self.get_rotation_matrix()).as_quat())

    def get_translation(self) -> TorchVec3:
        return TorchVec3(self[:3, 3])

    def set_translation(self, x: Vec3TypeX, y: Optional[float] = None, z: Optional[float] = None):
        x, y, z = unpack_3d_params(x, y, z)
        if x is not None:
            self[0, 3] = x
        if y is not None:
            self[1, 3] = y
        if z is not None:
            self[2, 3] = z

    def invert(self) -> 'TorchPose':
        inverted_rotation = self.get_rotation_matrix().T
        inverted_translation = -inverted_rotation @ self.get_translation()
        inverted_pose = TorchPose(inverted_rotation, inverted_translation)
        return inverted_pose

    def __rmatmul__(self, other):
        if isinstance(other, TorchPose):
            return super(TorchPose, self).__rmatmul__(other)
        else:
            return other @ to_base_tensor(self)

    def __matmul__(self, other):
        if isinstance(other, TorchPose):
            return super(TorchPose, self).__matmul__(other)
        else:
            return to_base_tensor(self) @ other

    def tensor(self) -> torch.Tensor:
        return to_base_tensor(self)


# ==========================================================
# Rotation Helper methods from torchgeometry
# ==========================================================


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`
    """

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis
