from typing import Optional, Union, Tuple

import numpy as np
import pyvista as pv

# TODO: Points is not a float type. This can cause issues when transforming or applying filters.
#  Casting to ``np.float32``. Disable this by passing ``force_float=False``
import vtk

from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector import Vec3, Vec4
from pyvista import examples


def add_coordinate_axes(p: pv.Plotter,
                        max_length: int = 10,
                        ticks_frequency: float = 1,
                        line_width: float = 1,
                        tick_length: float = 0.2,
                        arrow_size: float = 0.5,
                        draw_labels: bool = True,
                        scale: float = 1):
    axis_color_map = {
        0: 'r',
        1: 'g',
        2: 'b'
    }

    max_length *= scale
    ticks_frequency *= scale
    line_width *= scale
    tick_length *= scale
    arrow_size *= scale

    def _add_tick_lines(axis_across: int, axis_tick: int):
        points = []
        labels = []

        for pos in np.arange(-max_length, max_length, ticks_frequency):
            if pos == 0:
                continue

            point_1 = Vec3()
            point_1[axis_across] = pos
            point_1[axis_tick] = - tick_length

            point_2 = Vec3()
            point_2[axis_across] = pos
            point_2[axis_tick] = tick_length

            # # Apply Scaling
            # point_1 *= scale
            # point_2 *= scale

            p.add_lines(np.array([point_1, point_2]), color=axis_color_map[axis_across], width=line_width)

            points.append(point_2)
            labels.append(pos if round(pos) == pos else f"{pos:.1f}")

        if draw_labels:
            p.add_point_labels(points, labels, fill_shape=False, shape=None, show_points=False,
                               text_color=axis_color_map[axis_across])

    def _add_arrow(axis: int):
        cone_center = Vec3()
        cone_center[axis] = max_length - arrow_size / 2
        # cone_center *= scale
        p.add_mesh(pv.Cone(center=cone_center, direction=cone_center, height=arrow_size, radius=arrow_size / 2),
                   color=axis_color_map[axis])

    # Axes
    p.add_lines(np.array([[-max_length, 0, 0], [max_length - arrow_size, 0, 0]]), color='r', width=line_width)
    p.add_lines(np.array([[0, -max_length, 0], [0, max_length - arrow_size, 0]]), color='g', width=line_width)
    p.add_lines(np.array([[0, 0, -max_length], [0, 0, max_length - arrow_size]]), color='b', width=line_width)

    # Ticks
    # x: ticks across y dim
    # y: ticks across x dim
    # z: ticks across x dim
    _add_tick_lines(0, 1)
    _add_tick_lines(1, 0)
    _add_tick_lines(2, 0)

    # Arrows
    _add_arrow(0)
    _add_arrow(1)
    _add_arrow(2)


def add_floor(p: pv.Plotter,
              square_size: float = 1,
              max_distance: float = 10,
              color='gray',
              axes: Tuple[int, int] = (0, 1)):
    idx1, idx2 = axes
    for i in np.arange(-max_distance, max_distance, square_size):
        for j in np.arange(-max_distance, max_distance, square_size):
            point_1 = Vec3()
            point_1[idx1] = i
            point_1[idx2] = j

            point_2 = Vec3()
            point_2[idx1] = i + square_size
            point_2[idx2] = j

            point_3 = Vec3()
            point_3[idx1] = i + square_size
            point_3[idx2] = j + square_size

            point_4 = Vec3()
            point_4[idx1] = i
            point_4[idx2] = j + square_size

            floor_square = pv.Rectangle([
                point_1, point_2, point_3, point_4
            ])
            # floor_square = pv.Rectangle([
            #     [x, y, 0],
            #     [x + square_size, y, 0],
            #     [x + square_size, y + square_size, 0],
            #     [x, y + square_size, 0]]
            # )
            p.add_mesh(floor_square, show_edges=True, opacity=0.2, color=color)


def add_camera_frustum(p: pv.Plotter,
                       pose: Pose,
                       intrinsics: Intrinsics,
                       img_w: Optional[float] = None,
                       img_h: Optional[float] = None,
                       image: Optional[np.ndarray] = None,
                       color='lightgray',
                       size: float = 0.3,
                       label: Optional[Union[str, int]] = None,
                       line_width: float = 1,
                       look_vector_length: float = 0):
    if pose.pose_type == PoseType.WORLD_2_CAM:
        pose = pose.invert()

    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)

    assert pose.pose_type == PoseType.CAM_2_WORLD
    assert pose.camera_coordinate_convention == CameraCoordinateConvention.OPEN_CV

    center = pose.get_translation()

    depth = size * pose.camera_coordinate_convention.forward_direction.sign()

    # if image is not None:
    #     img_h = image.shape[0]
    #     img_w = image.shape[1]

    img_w = 2 * intrinsics.cx if img_w is None else img_w
    img_h = 2 * intrinsics.cy if img_h is None else img_h

    # Assume that x -> right, y -> down
    p_top_left = Vec4(0, 0, depth, 1)
    p_top_right = Vec4(img_w * depth, 0, depth, 1)
    p_bottom_left = Vec4(0, img_h * depth, depth, 1)
    p_bottom_right = Vec4(img_w * depth, img_h * depth, depth, 1)

    # Triangle visualizes -y direction of cam space
    cam_y_sign = pose.camera_coordinate_convention.up_direction.sign()
    p_tri_left = Vec4(0, cam_y_sign * img_h * depth / 5, depth, 1)
    p_tri_right = Vec4(img_w * depth, cam_y_sign * img_h * depth / 5, depth, 1)
    p_tri_top = Vec4(img_w * depth / 2, cam_y_sign * img_h * depth / 2, depth, 1)

    points = np.stack([p_top_left, p_top_right, p_bottom_right, p_bottom_left, p_tri_left, p_tri_right, p_tri_top])

    # TODO: Not sure if intrinsics is used correctly here
    points_world = (pose @ intrinsics.homogenize(invert=True) @ points.T).T
    points_world_frustum = points_world[:4]
    points_world_up_triangle = points_world[4:]

    if image is not None:
        image_rectangle = pv.Rectangle([points_world_frustum[0][:3],
                                        points_world_frustum[1][:3],
                                        points_world_frustum[2][:3],
                                        points_world_frustum[3][:3]])
        # image_file = examples.mapfile
        # tex = pv.read_texture(image_file)
        tex = pv.numpy_to_texture(image)
        image_rectangle.active_t_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
        p.add_mesh(image_rectangle, texture=tex)

    # Draw lines from rectangle corners to camera origin
    for i_point, point_world in enumerate(points_world_frustum):
        next_point_world = points_world_frustum[(i_point + 1) % 4]
        p.add_lines(np.array([center, point_world[:3]]), width=line_width, color=color)
        p.add_lines(np.array([point_world[:3], next_point_world[:3]]), width=line_width, color=color)

    # Draw triangle
    p.add_lines(np.concatenate(
        [points_world_up_triangle[[0], :3],
         points_world_up_triangle[[1], :3],
         points_world_up_triangle[[1], :3],
         points_world_up_triangle[[2], :3],
         points_world_up_triangle[[2], :3],
         points_world_up_triangle[[0], :3]], axis=0),
        width=line_width,
        color=color)

    # Draw look direction line
    if look_vector_length > 0:
        position = pose.get_translation()
        look_direction = pose.get_look_direction()
        p.add_lines(np.stack([position, position + look_vector_length * look_direction]), width=line_width, color=color)

    if label is not None:
        p_center = center
        # (p_top_left + 0.5 * (p_bottom_right - p_top_left))[:3]
        p.add_point_labels([p_center], [label],
                           fill_shape=False, shape=None, show_points=False, text_color=color)


def add_coordinate_system(p: pv.Plotter, cam_to_world: Pose, scale: float = 1, line_width: float = 1):
    axis_color_map = {
        0: 'r',
        1: 'g',
        2: 'b'
    }

    origin = cam_to_world.get_translation()
    px = Vec4(scale, 0, 0, 1)
    py = Vec4(0, scale, 0, 1)
    pz = Vec4(0, 0, scale, 1)

    points = np.stack([px, py, pz])
    points_world = (cam_to_world @ points.T).T

    for i_point, point_world in enumerate(points_world):
        color = axis_color_map[i_point]
        p.add_lines(np.array([origin, point_world[:3]]), width=line_width, color=color)


def set_camera(p: pv.Plotter, cam_to_world: Pose, neg_z_forward: bool = False):
    look_direction = cam_to_world.get_look_direction()
    up_direction = cam_to_world.get_up_direction()
    if neg_z_forward:
        look_direction *= -1
        # up_direction *= -1
    focal_point = cam_to_world.get_translation() + look_direction

    p.camera_set = True
    p.camera_position = cam_to_world.get_translation().tolist()
    p.camera.focal_point = focal_point
    p.camera.up = up_direction  # TODO: Do we need to negate here? OpenCV coordinate y goes down


def render_from_camera(p: pv.Plotter,
                       pose: Pose,
                       intrinsics: Intrinsics) -> np.ndarray:
    """
    Render the given scene from the specified camera.
    The image size will be the size of the pyvista window.
    The pyvista plotter should be initialized like this:
    ```
    p = pv.Plotter(off_screen=True, window_size=[IMG_W, IMG_H])
    ```

    Parameters
    ----------
        p: the pyvista scene to render
        pose: extrinsics camera pose
        intrinsics: camera intrinsics

    Returns
    -------
        A numpy array [IMG_H, IMG_W, 4] containing the rendered scene. The background is transparent and a mask can
        be obtained from the alpha channel
    """

    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    pose = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)

    w = p.window_size[0]
    h = p.window_size[1]

    # ----------------------------------------------------------
    # Intrinsics
    # ----------------------------------------------------------
    cx = intrinsics.cx
    cy = intrinsics.cy
    fx = intrinsics.fx
    fy = intrinsics.fy

    # convert the principal point to window center (normalized coordinate system) and set it
    wcx = -2 * (cx - float(w) / 2) / w
    wcy = 2 * (cy - float(h) / 2) / h
    p.camera.SetWindowCenter(wcx, wcy)

    # convert the focal length to view angle and set it
    view_angle = 180 / np.pi * (2.0 * np.arctan2(h / 2.0, fx))
    p.camera.SetViewAngle(view_angle)

    # ----------------------------------------------------------
    # Extrinsics
    # ----------------------------------------------------------

    # apply the transform to scene objects
    vtk_pose = vtk.vtkMatrix4x4()
    for i in range(pose.shape[0]):
        for j in range(pose.shape[1]):
            vtk_pose.SetElement(i, j, pose[i, j])

    p.camera.SetModelTransformMatrix(vtk_pose)

    # the camera can stay at the origin because we are transforming the scene objects
    p.camera.SetPosition(0, 0, 0)

    # look in the +Z direction of the camera coordinate system
    p.camera.SetFocalPoint(0, 0, 1)

    # the camera Y axis points down
    p.camera.SetViewUp(0, -1, 0)

    # ensure the relevant range of depths are rendered
    p.renderer.ResetCameraClippingRange()

    p.render()  # Important, otherwise view isn't updated when render_from_camera() is called multiple times
    img = np.asarray(p.screenshot(transparent_background=True))

    return img
