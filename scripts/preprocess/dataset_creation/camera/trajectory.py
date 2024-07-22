from math import cos, sin
from typing import Optional, List

import numpy as np

from .camera import PoseType
from .matrix import Pose
from .vector import Vec3


def point_around_axis(theta: float,
                      axis: Vec3 = Vec3(0, 0, 1)) -> Vec3:
    """
    Compute a point with unit distance from `axis` that is rotated by `theta` around it.
    It is somewhat arbitrary where `theta=0` lands.
     - (1, 0, 0) -> (0, 0, 1)
     - (0, 1, 0) -> (0, 0, -1)
     - (0, 0, 1) -> (0, 1, 0)

    Computed points via `point_around_axis()` will be centered around the origin.

    Parameters
    ----------
        theta: angle between [0, 2pi) specifying the rotation around the axis
        axis: the axis to rotate around
    """

    axis = Vec3(axis)
    non_parallel = Vec3(0, 1, 0) if (axis == Vec3(1, 0, 0)).all() else Vec3(1, 0, 0)
    v = axis.cross(non_parallel).normalize()
    v_rotated = cos(theta) * v + sin(theta) * (np.cross(axis, v)) + (1 - cos(theta)) * axis * (np.dot(axis, v))

    return v_rotated


def circle_around_axis(n_poses: int,
                       axis=Vec3(0, 0, 1),
                       up: Vec3 = Vec3(0, 0, 1),
                       move=Vec3(),
                       distance: float = 1.0,
                       distance_end: Optional[float] = None,
                       theta_from: float = 0,
                       theta_to: float = 2 * np.pi,
                       look_at: Vec3 = Vec3(0, 0, 0)) -> List[Pose]:
    """
    Computes `n_poses` many camera poses (cam2world) that circle with distance `distance` around the specified `axis`
    that is moved by `move`.
    Per default, one full circle is computed. With `theta_from` and `theta_to` one can specify parts of the circle
    or even multiple circulations around the axis.

    Parameters
    ----------
        n_poses:
            how many poses should be computed
        axis:
            The axis (direction) around which we rotate
        up:
            which direction should be up for the camera
        move:
            the location of the axis
        distance:
            distance of the pose locations from the axis
        distance_end:
            if specified, `distance` will be interpreted as a start distance for the first pose and distance_end
            defines the distance from the axis for the last pose. Distances in for poses in between are linearly
            interpolated. This gives a spiraling effect.
        theta_from:
            orientation of the first pose
        theta_to:
            orientation of the last pose
        look_at:
            all circle poses will look at the specified point in space. Per default, this is the origin
    """

    if distance_end is None:
        distance_end = distance
        distance_start = distance
    else:
        distance_start = distance

    poses = []
    for i_pose in range(n_poses):
        alpha = i_pose / n_poses
        theta = theta_from + alpha * (theta_from - theta_to)
        distance = distance_start + alpha * (distance_end - distance_start)
        location = distance * point_around_axis(theta, axis=axis)

        pose = Pose(pose_type=PoseType.CAM_2_WORLD)
        location += move
        pose.set_translation(location)
        pose.look_at(look_at, up=up)

        poses.append(pose)

    return poses
