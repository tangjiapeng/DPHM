from typing import Optional, Union, List, Tuple
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..camera import CameraCoordinateConvention, PoseType
from .pose_base import is_rotation_matrix
from ..vector.vector_base import Vec3TypeX, FloatType, unpack_3d_params, Vec3Type
from ..vector.vector_numpy import Vec3, Vec4


class Pose(np.ndarray):
    camera_coordinate_convention: CameraCoordinateConvention
    pose_type: PoseType

    def __new__(cls, matrix_or_rotation: Union[np.ndarray, List] = np.eye(4), translation: Optional[Vec3Type] = None,
                camera_coordinate_convention: CameraCoordinateConvention = CameraCoordinateConvention.OPEN_CV,
                pose_type: PoseType = PoseType.WORLD_2_CAM):
        """
        Per default, Poses are assumed to be WORLD_2_CAM poses that transform world coordinates into an OPEN_CV
        camera space.

        Parameters
        ----------
            matrix_or_rotation:
                Either a full 4x4 affine transform matrix or a 3x3 rotation matrix
            translation:
                If `matrix_or_rotation` was only a 3x3 rotation matrix, ´translation` can be specified as a 3x1 vector
            camera_coordinate_convention:
                Indicates which coordinate convention is used for the camera space that this pose matrix maps into
            pose_type:
                Indicates whether this pose maps from WORLD -> CAM or CAM -> WORLD (inverse)
        """

        pose = super().__new__(cls, (4, 4), dtype=np.float32)

        pose.camera_coordinate_convention = camera_coordinate_convention
        pose.pose_type = pose_type

        if not isinstance(matrix_or_rotation, np.ndarray):
            matrix_or_rotation = np.asarray(matrix_or_rotation)

        if matrix_or_rotation.shape == (4, 4):
            # Full 4x4 Pose
            assert translation is None, "If a full pose is given, no translation should be specified!"
            assert (matrix_or_rotation[3, :] == [0, 0, 0, 1]).all(), \
                f"Last row of pose must be [0, 0, 0, 1]. Got {matrix_or_rotation[3, :]}"
            assert is_rotation_matrix(matrix_or_rotation[:3, :3]), \
                f"Specified matrix does not contain a valid rotation matrix! {matrix_or_rotation[:3, :3]}"

            pose[:] = matrix_or_rotation

        elif matrix_or_rotation.shape == (3, 3):
            # 3x3 rotation matrix + 3(x1) translation vector
            translation = np.asarray(translation)
            assert translation.squeeze().shape == (3,), \
                "If a rotation matrix is given, the translation vector has to be 3 dimensional!"

            pose[:3, :3] = matrix_or_rotation
            pose[:3, 3] = translation.squeeze()
            pose[3, :] = [0, 0, 0, 1]
        elif matrix_or_rotation.shape == (3, 1) or matrix_or_rotation.shape == (3,):
            # 3(x1) Rodriguez vector + 3(x1) translation vector
            assert translation.squeeze().shape == (3,), \
                "If a Rodriguez vector is given, the translation vector has to be 3 dimensional!"

            pose[:3, :3] = cv2.Rodrigues(matrix_or_rotation)[0]
            pose[:3, 3] = translation.squeeze()
            pose[3, :] = [0, 0, 0, 1]
        else:
            raise ValueError("Either a full pose has to be given or a 3x3 rotation + 3x1 translation!")

        return pose

    @staticmethod
    def from_euler(euler_angles: Vec3Type,
                   translation: Vec3Type = Vec3(),
                   euler_mode: str = 'XYZ',
                   camera_coordinate_convention: CameraCoordinateConvention = CameraCoordinateConvention.OPEN_CV,
                   pose_type: PoseType = PoseType.WORLD_2_CAM) -> 'Pose':
        return Pose(R.from_euler(euler_mode, euler_angles).as_matrix(),
                    translation,
                    camera_coordinate_convention=camera_coordinate_convention,
                    pose_type=pose_type)

    @staticmethod
    def from_rodriguez(rodriguez_vector: Vec3Type,
                       translation: Vec3Type = Vec3(),
                       camera_coordinate_convention: CameraCoordinateConvention = CameraCoordinateConvention.OPEN_CV,
                       pose_type: PoseType = PoseType.WORLD_2_CAM) -> 'Pose':
        return Pose(cv2.Rodrigues(rodriguez_vector)[0],
                    translation,
                    camera_coordinate_convention=camera_coordinate_convention,
                    pose_type=pose_type)

    def get_rotation_matrix(self) -> np.ndarray:
        return self[:3, :3]

    def get_euler_angles(self, order: str) -> Vec3:
        return Vec3(R.from_matrix(self.get_rotation_matrix()).as_euler(order))

    def get_rodriguez_vector(self) -> Vec3:
        return Vec3(cv2.Rodrigues(self.get_rotation_matrix())[0].squeeze())

    def get_quaternion(self) -> Vec4:
        return Vec4(R.from_matrix(self.get_rotation_matrix()).as_quat())

    def get_translation(self) -> Vec3:
        # assert self.pose_type == PoseType.CAM_2_WORLD, "camera position only makes sense for CAM_2_WORLD poses"
        return Vec3(self[:3, 3])

    def set_translation(self, x: Optional[Vec3TypeX] = None, y: Optional[FloatType] = None, z: Optional[FloatType] = None):
        x, y, z = unpack_3d_params(x, y, z)
        if x is not None:
            self[0, 3] = x
        if y is not None:
            self[1, 3] = y
        if z is not None:
            self[2, 3] = z

    def move(self, x: Optional[Vec3TypeX] = None, y: Optional[FloatType] = None, z: Optional[FloatType] = None,
             inplace: bool = True) -> 'Pose':
        x, y, z = unpack_3d_params(x, y, z, 0)

        if inplace:
            pose = self
        else:
            pose = self.copy()

        pose[0, 3] += x
        pose[1, 3] += y
        pose[2, 3] += z

        return pose

    def scale(self, scale: float) -> 'Pose':
        self[:3, 3] *= scale
        return self

    def set_rotation_matrix(self, rotation_matrix: np.ndarray):
        assert is_rotation_matrix(rotation_matrix), \
            f"Specified matrix does not contain a valid rotation matrix! {rotation_matrix}"
        self[:3, :3] = rotation_matrix

    def set_rotation_euler(self, order: str,
                           euler_x: Vec3TypeX = 0,
                           euler_y: Optional[float] = None,
                           euler_z: Optional[float] = None):
        euler_x, euler_y, euler_z = unpack_3d_params(euler_x, euler_y, euler_z, default=0)
        self[:3, :3] = R.from_euler(order, [euler_x, euler_y, euler_z]).as_matrix()

    def rotate_euler(self,
                     order: str,
                     euler_x: Vec3TypeX = 0,
                     euler_y: Optional[float] = None,
                     euler_z: Optional[float] = None,
                     inplace: bool = True) -> 'Pose':
        if inplace:
            pose = self
        else:
            pose = self.copy()

        euler_x, euler_y, euler_z = unpack_3d_params(euler_x, euler_y, euler_z, default=0)
        euler_rotation = Vec3(euler_x, euler_y, euler_z)
        current_euler_angles = pose.get_euler_angles(order)
        pose.set_rotation_euler(order, current_euler_angles + euler_rotation)

        return pose

    def invert(self) -> 'Pose':
        inverted_rotation = self.get_rotation_matrix().T
        inverted_translation = -inverted_rotation @ self.get_translation()
        inverted_pose = Pose(inverted_rotation,
                             inverted_translation,
                             camera_coordinate_convention=self.camera_coordinate_convention,
                             pose_type=self.pose_type.invert())
        return inverted_pose

    def negate_orientation_axis(self, axis: int):
        """
        Changes the coordinate convention of the camera space.
        E.g., if the pose is assumed to be in OpenCV camera space (x -> right, y -> down, z -> forward), then
        negate_orientation_axis(1) will make it DirectX (x -> right, y -> up, z -> forward).
        The function negates the column of the rotation matrix that corresponds to the given axis.
        The camera's position in world space is NOT affected by this transformation.
        Only the orientation of the camera frustum is changed.
        Assumes the current pose is cam2world.

        Parameters
        ----------
            axis: which axis to negate.
        """

        # negates the column of the rotation matrix
        self[:3, axis] *= -1

    def swap_axes(self, permutation: List[Union[int, str]], inplace: bool = True) -> 'Pose':
        """
        Negates/swaps entire rows of the pose matrix.
        Essentially, applies a rotation / flip operation around the world origin to the camera object.
        E.g., swap_axes(['-x', 'y', 'z']) will move an origin-facing camera that was at (3, 1, 1) to (-3, 1, 1).
        The moved camera will still face the origin afterwards (i.e., the camera is not just moved but also rotated).
        Alternatively, instead of flipping the camera along axis-aligned planes, swap_axes() can be interpreted as
        flipping the actual scene in world space while keeping the camera where it was.
        I.e., after applying the camera transformation the scene will be rendered as if it was flipped.
        Assumes the current pose is cam2world

        Parameters
        ----------
            permutation: 3-tuple of axis indicators (either 0, 1, 2 or x, y, z with optional '-' signs)
        """

        if inplace:
            pose = self
        else:
            pose = self.copy()

        axis_switcher = np.zeros((4, 4))
        axis_order = ['x', 'y', 'z']
        for idx, a in enumerate(permutation):
            v = 1
            if isinstance(a, int):
                ax = a
            else:
                # Possibility to also flip an axis via -x, -y etc.
                ax = axis_order.index(a[-1])  # Map x -> 0, y -> 1, z -> 2
                if a[0] == '-':
                    # Axis shall be flipped
                    v = -1

            axis_switcher[idx, ax] = v
        axis_switcher[3, 3] = 1

        if np.abs(np.linalg.det(axis_switcher) - 1) > 1e-6:
            print("[WARNING] swap_axis changes handedness!")

        # Negates / Flips rows
        pose[:, :] = axis_switcher @ pose

        return pose

    def change_camera_coordinate_convention(self,
                                            new_camera_coordinate_convention: CameraCoordinateConvention,
                                            inplace: bool = True) -> 'Pose':
        # TODO: Make this work for WORLD_2_CAM poses as well
        assert self.pose_type == PoseType.CAM_2_WORLD, "Camera coordinate conventions can only be changed on CAM_2_WORLD matrices"

        current_ccc = self.camera_coordinate_convention

        if inplace:
            pose = self
        else:
            pose = self.copy()

        if current_ccc.x_direction != new_camera_coordinate_convention.x_direction:
            pose.negate_orientation_axis(0)

        if current_ccc.y_direction != new_camera_coordinate_convention.y_direction:
            pose.negate_orientation_axis(1)

        if current_ccc.z_direction != new_camera_coordinate_convention.z_direction:
            pose.negate_orientation_axis(2)

        pose.camera_coordinate_convention = new_camera_coordinate_convention

        return pose

    def change_pose_type(self, new_pose_type: PoseType, inplace: bool = True) -> 'Pose':
        assert self.pose_type in {PoseType.CAM_2_WORLD, PoseType.WORLD_2_CAM}, "start pose must be either cam2world or world2cam"
        # print(self.pose_type)
        # print(new_pose_type,  new_pose_type in {PoseType.CAM_2_WORLD, PoseType.WORLD_2_CAM} )
        assert new_pose_type in {PoseType.CAM_2_WORLD, PoseType.WORLD_2_CAM}, "target pose type must be either cam2world or world2cam"

        if inplace:
            pose = self
        else:
            pose = self.copy()

        if pose.pose_type != new_pose_type:
            pose = pose.invert()

        return pose

    def get_look_direction(self) -> 'Vec3':
        # Assumes the current pose is cam2world
        # Assigns meaning to the coordinate axes. Hence, the coordinate system convention is important
        # Assumes an OpenCV camera coordinate system convention (x -> right, y -> down, z -> forward/look)

        assert self.pose_type == PoseType.CAM_2_WORLD
        forward_direction = self.camera_coordinate_convention.forward_direction
        axis = forward_direction.axis_id
        sign = forward_direction.sign()
        look_direction = sign * Vec3(self[:3, axis])

        return look_direction

    def get_up_direction(self) -> 'Vec3':
        # Assumes the current pose is cam2world
        # Assigns meaning to the coordinate axes. Hence, the coordinate system convention is important
        # Assumes an OpenCV camera coordinate system convention (x -> right, y -> down, z -> forward)

        assert self.pose_type == PoseType.CAM_2_WORLD
        up_direction = self.camera_coordinate_convention.up_direction
        axis = up_direction.axis_id
        sign = up_direction.sign()

        up_direction = sign * Vec3(self[:3, axis])

        return up_direction

    def look_at(self, at: Vec3, up: Vec3 = Vec3(0, 0, 1)):
        # This method assigns meaning to the coordinate axes. Hence, the coordinate system convention is important
        # We use the OpenCV camera coordinate system convention

        # Poses are always assumed to be cam2world
        # That way the translation part of the pose matrix is the location of the object in world space

        assert self.pose_type == PoseType.CAM_2_WORLD
        assert self.camera_coordinate_convention == CameraCoordinateConvention.OPEN_CV

        eye = self.get_translation()
        z_axis = (at - eye).normalize()  # Assumes z-axis is forward
        x_axis = z_axis.cross(up).normalize()  # Assumes y-axis is up
        y_axis = x_axis.cross(z_axis).normalize()

        # Important as otherwise rotation matrix has negative determinant (would be left-handed).
        # Makes it a [x, -y, z] OpenCV camera coordinate system
        # [x, y, -z] would be a Blender/OpenGL camera coordinate system
        y_axis = - y_axis

        self.set_rotation_matrix(np.array([x_axis, y_axis, z_axis]).T)
        # self.set_translation(np.dot(x_axis, eye), np.dot(y_axis, eye), np.dot(z_axis, eye))
        self.set_translation(eye)

    def calculate_rigid_transformation_to(self, other: 'Pose') -> 'Pose':
        """
        Let current cam2world pose = X and target cam2world pose = Y.
        Calculate a rigid transform T, s.t. T @ X = Y
        In other words, T shall rotate and move the coordinate system of X such that it aligns with the coordinate
        system of Y.

        Parameters
        ----------
            other:
                the target pose. Must be either cam2world or world2cam

        Returns
        -------
            The rigid transformation T (a cam2cam pose) that aligns the current pose with the specified pose.
        """

        assert self.pose_type in {PoseType.CAM_2_WORLD, PoseType.WORLD_2_CAM}, \
            "start pose must be either cam2world or world2cam"
        assert other.pose_type in {PoseType.CAM_2_WORLD, PoseType.WORLD_2_CAM}, \
            "target pose must be either cam2world or world2cam"

        # TODO: Continue
        source_cam2world = self.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
        target_cam2world = other.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)

        rigid_transformation = target_cam2world @ source_cam2world.invert()
        assert rigid_transformation.pose_type == PoseType.CAM_2_CAM

        return rigid_transformation

    def __rmatmul__(self, other):
        if isinstance(other, Pose):
            pose = super(Pose, self).__rmatmul__(other)
            pose.camera_coordinate_convention = self.camera_coordinate_convention
            pose.pose_type = self.pose_type @ other.pose_type
            return pose
        else:
            return other @ np.array(self)

    def __matmul__(self, other):
        # TODO: figure out why numpy operations automatically cast to Pose again
        if isinstance(other, Pose):
            pose = super(Pose, self).__matmul__(other)
            pose.camera_coordinate_convention = self.camera_coordinate_convention
            pose.pose_type = self.pose_type @ other.pose_type
            return pose
        else:
            return np.array(self) @ other

    def copy(self, order='C') -> 'Pose':
        pose = super(Pose, self).copy(order=order)
        pose.camera_coordinate_convention = self.camera_coordinate_convention
        pose.pose_type = self.pose_type

        return pose

    def numpy(self) -> np.ndarray:
        return np.array(self)

    def __repr__(self):
        representation = super(Pose, self).__repr__()
        representation = f"{representation}\n" \
                         f" > camera_coordinate_convention: {self.camera_coordinate_convention.name}\n" \
                         f" > pose_type: {self.pose_type.name}"
        return representation

    def __str__(self):
        string = super(Pose, self).__str__()
        string = f"{string}\n" \
                 f" > camera_coordinate_convention: {self.camera_coordinate_convention.name}\n" \
                 f" > pose_type: {self.pose_type.name}"
        return string
