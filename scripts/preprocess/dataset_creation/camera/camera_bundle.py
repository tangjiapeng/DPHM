from typing import List, Optional, Union, Tuple

import numpy as np
from .camera import PoseType

from .matrix import Pose
from .vector import Vec3, rotation_matrix_between_vectors, offset_vector_between_line_and_point, \
    angle_between_vectors


def calculate_look_center(cam_to_world_poses: List[Pose]) -> Vec3:
    a = []  # point on look-direction line (i.e., position of camera)
    d = []  # look direction

    for cam_to_world_pose in cam_to_world_poses:
        position = cam_to_world_pose.get_translation()
        look_direction = cam_to_world_pose.get_look_direction()

        a.append(position)
        d.append(look_direction)

    # See https://stackoverflow.com/questions/48154210/3d-point-closest-to-multiple-lines-in-3d-space
    M = np.zeros((3, 3))
    for k in range(3):
        ms = [d[i][k] * d[i] - (d[i].dot(d[i])) * Vec3.unit(k) for i in range(3)]
        M[k] = sum(ms)

    b = sum([d[i] * (a[i].dot(d[i])) - a[i] * (d[i].dot(d[i])) for i in range(3)])

    look_center = np.linalg.solve(M, b)

    return Vec3(look_center)


def align_poses(world_to_cam_poses: List[Pose],
                up: Optional[Vec3] = Vec3(0, 1, 0),
                look: Optional[Vec3] = Vec3(0, 0, -1),
                look_center: Optional[Vec3] = Vec3(0, 0, 0),
                cam_to_world: bool = False,
                inplace: bool = False,
                return_transformation: bool = False) -> Union[List[Pose], Tuple[List[Pose], Pose]]:
    """
    Calibration poses can be arbitrarily aligned. This method provides a utility to transform a set of camera poses
    such that their up/look directions and look center correspond to the specified values.
    calibration poses are expected in world_to_cam format and OpenCV coordinate convention
    (i.e., x -> right, y -> down, z -> forward).
    Per default, the set of camera poses will be transformed to look at the center in an OpenGL world
    (i.e., x -> right, y -> up, z -> backward).

    Parameters
    ----------
        world_to_cam_poses: the poses to transform
        up: where the up direction should point to
        look: where the look direction should point to
        look_center: where the look center of all cameras should fall into
        cam_to_world: whether the provided poses are already cam_to_world
        return_transformation:
            If specified, a 4x4 transformation matrix is returned that transforms the cam_to_world_poses into
            aligned space exactly as align_poses() would do. Apply as:
                transformation @ world_to_cam_poses[i].invert()

    Returns
    -------
        the re-aligned camera poses
    """

    if not inplace:
        world_to_cam_poses = [pose.copy() for pose in world_to_cam_poses]

    if cam_to_world:
        cam_to_world_poses = world_to_cam_poses
    else:
        cam_to_world_poses = [cam_pose.invert() for cam_pose in world_to_cam_poses]

    transformation = np.eye(4)

    # Align up direction
    if up is not None:
        up_directions = [cam_pose.get_up_direction() for cam_pose in cam_to_world_poses]
        average_up_direction = np.mean(up_directions, axis=0)
        align_up_rotation = rotation_matrix_between_vectors(average_up_direction, up)
        rotator_up = Pose(align_up_rotation, Vec3(), pose_type=PoseType.CAM_2_CAM)
        cam_to_world_poses = [rotator_up @ cam_pose for cam_pose in cam_to_world_poses]
        transformation = rotator_up

    # Align the look direction
    if look is not None:
        look_directions = [cam_pose.get_look_direction() for cam_pose in cam_to_world_poses]
        average_look_direction = np.mean(look_directions, axis=0)
        align_look_rotation = rotation_matrix_between_vectors(average_look_direction, look)
        rotator_look = Pose(align_look_rotation, Vec3(), pose_type=PoseType.CAM_2_CAM)
        cam_to_world_poses = [rotator_look @ cam_pose for cam_pose in cam_to_world_poses]
        transformation = rotator_look @ transformation

    # Align the look center
    if look_center is not None:
        original_look_center = calculate_look_center(cam_to_world_poses)
        offset_vector = look_center - original_look_center

        # look_directions = [cam_pose.get_look_direction() for cam_pose in cam_to_world_poses]
        # cameras_center = np.mean([cam_pose.get_translation() for cam_pose in cam_to_world_poses], axis=0)
        # average_look_direction = np.mean(look_directions, axis=0)
        # # TODO: This won't move cameras much if cameras_center is already at look_center
        # #   Would have to somehow find the point that is closest to all camera rays
        # offset_vector = offset_vector_between_line_and_point(cameras_center, average_look_direction, look_center)
        for cam_pose in cam_to_world_poses:
            cam_pose.move(offset_vector)

        mover = np.eye(4)
        mover[:3, 3] = offset_vector
        transformation = mover @ transformation

    if up is not None:
        # Aligning the look direction might mess up the up direction again
        up_directions = [cam_pose.get_up_direction() for cam_pose in cam_to_world_poses]
        average_up_direction = np.mean(up_directions, axis=0)
        angle = angle_between_vectors(average_up_direction, up)

        # TODO: Here we assume that look direction should be z axis. Correct would be to rotate around look direction
        rotator = Pose.from_euler(Vec3(0, 0, angle), pose_type=PoseType.CAM_2_CAM)
        cam_to_world_poses = [rotator @ cam_pose for cam_pose in cam_to_world_poses]
        transformation = rotator @ transformation

        up_directions_after_alignment = [cam_pose.get_up_direction() for cam_pose in cam_to_world_poses]
        average_up_direction_after_alignment = np.mean(up_directions_after_alignment, axis=0)
        assert average_up_direction_after_alignment.dot(up) > 0.9, \
            "Up directions could not be properly aligned. This can happen if the desired look direction is something else than the z-axis"

    if return_transformation:
        transformation = Pose(transformation, pose_type=PoseType.CAM_2_CAM)
        return cam_to_world_poses, transformation
    else:
        return cam_to_world_poses
