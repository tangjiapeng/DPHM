from typing import Union, Tuple

import numpy as np

from .camera import CameraCoordinateConvention, PoseType
from .matrix import Pose, Intrinsics


def back_project(points: np.ndarray, depths: np.ndarray, cam_to_world_pose: Pose, intrinsics: Intrinsics) -> np.ndarray:
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert len(depths.shape) == 1
    N = points.shape[0]
    assert depths.shape[0] == N, "Need to have one depth value per point"
    assert cam_to_world_pose.camera_coordinate_convention == CameraCoordinateConvention.OPEN_CV
    assert cam_to_world_pose.pose_type == PoseType.CAM_2_WORLD

    p_screen = np.hstack([points, np.ones((points.shape[0], 1))])
    p_screen_canonical = p_screen @ intrinsics.invert().T
    p_cam = p_screen_canonical * np.expand_dims(depths, 1)
    p_cam_hom = np.hstack([p_cam, np.ones((p_cam.shape[0], 1))])
    p_world = p_cam_hom @ cam_to_world_pose.T

    return p_world[:, :3]


def project(points: np.ndarray, pose: Pose, intrinsics: Intrinsics) -> np.ndarray:
    """
    Projects 3D points onto the image plane defined by the camera pose and intrinsics.

    Parameters
    ----------
        points: 3D points [N, 3]
        pose: Camera pose. Can be either CAM_2_WORLD or WORLD_2_CAM
        intrinsics: Intrinsics of the camera

    Returns
    -------
        Projected points [N, 3] in image space where the third coordinate defines the depth from the camera space.
        Note that no occlusions are handled (difficult for point clouds) and points that would be outside the image
        won't be filtered.
    """

    assert len(points.shape) == 2
    assert points.shape[1] == 3
    cam_to_world_pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    cam_to_world_pose = cam_to_world_pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV,
                                                                              inplace=False)
    world_to_cam_pose = cam_to_world_pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)

    p_world = np.hstack([points, np.ones((points.shape[0], 1))])
    p_cam = p_world @ world_to_cam_pose.T
    depths = p_cam[:, [2]]  # Assuming OpenCV convention: depth is positive z-axis
    p_cam = p_cam / depths
    p_screen = p_cam[:, :3] @ intrinsics.T
    p_screen[:, 2] = np.squeeze(depths, 1)  # Return depth as third coordinate

    return p_screen


def draw_onto_image(image: np.ndarray,
                    points: np.ndarray,
                    values: Union[np.ndarray, Tuple]):
    projected_x = points[:, 0].round().astype(int)
    projected_y = points[:, 1].round().astype(int)
    valid_x = (0 <= projected_x) & (projected_x < image.shape[1])
    valid_y = (0 <= projected_y) & (projected_y < image.shape[0])
    valid_xy = valid_x & valid_y

    if isinstance(values, tuple):
        image[projected_y[valid_xy], projected_x[valid_xy]] = values
    else:
        image[projected_y[valid_xy], projected_x[valid_xy]] = values[valid_xy]
