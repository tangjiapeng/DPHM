import math
from enum import Enum, auto
import numpy as np
from .vector import Vec3


class AxisDirection(Enum):
    X = (1, 0, 0)
    NEG_X = (-1, 0, 0)

    Y = (0, 1, 0)
    NEG_Y = (0, -1, 0)

    Z = (0, 0, 1)
    NEG_Z = (0, 0, -1)

    def __init__(self, x: int, y: int, z: int):
        assert sum([x == 0, y == 0, z == 0]) == 2, "Exactly two coordinates have to be zero"
        assert sum([abs(x) == 1, abs(y) == 1, abs(z) == 1]) == 1, "Exactly one coordinate has to be 1 or -1"

        self.axis = Vec3(x, y, z)
        self.axis_id = np.abs(np.array([x, y, z])).argmax()


    def sign(self) -> int:
        return self.axis.sum()

    def axis_name(self) -> str:
        if '_' in self.name:
            axis_name = self.name.split('_')[1]
            axis_name = f"-{axis_name.lower()}"
        else:
            axis_name = self.name.lower()

        return axis_name


class SemanticAxisDirection(Enum):
    RIGHT = (AxisDirection.X)
    LEFT = (AxisDirection.NEG_X)
    UP = (AxisDirection.Y)
    DOWN = (AxisDirection.NEG_Y)
    FORWARD = (AxisDirection.Z)
    BACKWARD = (AxisDirection.NEG_Z)

    def __init__(self, canonical_axis_direction: AxisDirection):
        self.canonical_axis_direction = canonical_axis_direction


class Handedness(Enum):
    RIGHT_HANDED = auto()
    LEFT_HANDED = auto()

    @staticmethod
    def from_sign(sign: int) -> 'Handedness':
        if sign == 1:
            return Handedness.LEFT_HANDED
        elif sign == -1:
            return Handedness.RIGHT_HANDED
        else:
            raise ValueError("Handedness sign must be either 1 or -1")


class Space(Enum):
    CAM = auto()
    WORLD = auto()


class PoseType(Enum):
    CAM_2_WORLD = (Space.CAM, Space.WORLD)
    WORLD_2_CAM = (Space.WORLD, Space.CAM)
    CAM_2_CAM = (Space.CAM, Space.CAM)

    def __init__(self, space_from: Space, space_to: Space):
        self.from_space = space_from
        self.to_space = space_to

    def invert(self) -> 'PoseType':
        if self == PoseType.CAM_2_WORLD:
            return PoseType.WORLD_2_CAM
        elif self == PoseType.WORLD_2_CAM:
            return PoseType.CAM_2_WORLD
        else:
            return PoseType.CAM_2_CAM

    def __matmul__(self, other: 'PoseType') -> 'PoseType':
        assert self.to_space == other.from_space, f"Poses cannot be multiplied: {self.name} and {other.name}"

        new_from_space = self.from_space
        new_to_space = other.to_space

        for pose_type in PoseType:
            if pose_type.from_space == new_from_space and pose_type.to_space == new_to_space:
                return pose_type

        raise ValueError(f"Poses cannot be multiplied: {self.name} and {other.name}. "
                         f"Would result in {new_from_space.name} -> {new_to_space.name}")


class CameraCoordinateConvention(Enum):
    OPEN_CV = (SemanticAxisDirection.RIGHT, SemanticAxisDirection.DOWN, SemanticAxisDirection.FORWARD)
    OPEN_GL = (SemanticAxisDirection.RIGHT, SemanticAxisDirection.UP, SemanticAxisDirection.BACKWARD)  # Also Blender
    DIRECT_X = (SemanticAxisDirection.RIGHT, SemanticAxisDirection.UP, SemanticAxisDirection.FORWARD)  # Also Unity
    PYTORCH_3D = (SemanticAxisDirection.LEFT, SemanticAxisDirection.UP, SemanticAxisDirection.FORWARD)

    def __init__(self,
                 x_direction: SemanticAxisDirection,
                 y_direction: SemanticAxisDirection,
                 z_direction: SemanticAxisDirection):
        self.x_direction = x_direction
        self.y_direction = y_direction
        self.z_direction = z_direction

        # For cameras: Either y or -y is the up direction
        self.up_direction = AxisDirection.Y if y_direction == SemanticAxisDirection.UP else AxisDirection.NEG_Y

        # For cameras: Either z or -z is the forward direction
        self.forward_direction = AxisDirection.Z if z_direction == SemanticAxisDirection.FORWARD else AxisDirection.NEG_Z

        coordinate_system_sign = self.x_direction.canonical_axis_direction.sign() \
                                 * self.y_direction.canonical_axis_direction.sign() \
                                 * self.z_direction.canonical_axis_direction.sign()
        self.handedness = Handedness.from_sign(coordinate_system_sign)

    def __str__(self) -> str:
        string = f"{self.name} ({self.handedness.name}):\n" \
                 f"\t x -> {self.x_direction.name}\n" \
                 f"\t y -> {self.y_direction.name}\n" \
                 f"\t z -> {self.z_direction.name}\n" \
                 f"\t --------------\n" \
                 f"\t up -> {self.up_direction.axis_name()}\n" \
                 f"\t forward -> {self.forward_direction.axis_name()}\n"
        return string


def focal_length_to_fov(focal_length: float, image_size: float):
    """
    Parameters
    ----------
        focal_length:
            focal length (x or y) in pixels or physical millimeters
        image_size:
            width (if focal length defines x) or height of the image plane in pixels or physical millimeters.
            focal_length and image_size have to be both pixels or both millimeters
    """

    return 2 * math.atan(image_size / (2 * focal_length))


if __name__ == '__main__':
    print(CameraCoordinateConvention.OPEN_CV)
    print(CameraCoordinateConvention.DIRECT_X)
    print(CameraCoordinateConvention.OPEN_GL)
