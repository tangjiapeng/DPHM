from typing import Union, Optional
import numpy as np


class Intrinsics(np.ndarray):
    def __new__(cls, matrix_or_fx: Union[np.ndarray, float] = np.eye(3),
                fy: Optional[float] = None,
                cx: Optional[float] = None,
                cy: Optional[float] = None,
                s: Optional[float] = None) -> 'Intrinsics':
        intrinsics = super().__new__(cls, (3, 3), dtype=np.float32)
        if not isinstance(matrix_or_fx, np.ndarray) and not isinstance(matrix_or_fx, (float, int)):
            matrix_or_fx = np.asarray(matrix_or_fx)

        if isinstance(matrix_or_fx, np.ndarray) and matrix_or_fx.shape == (3, 3):
            assert fy is None and cx is None and cy is None and s is None, \
                "If a full intrinsics matrix is given, no other parameters should be specified!"
            intrinsics[:] = matrix_or_fx
        elif isinstance(matrix_or_fx, (float, int)):
            assert not (cx is None or cy is None), \
                "If a focal length is given, cx and cy have to be specified!"

            s = 0 if s is None else s
            fy = matrix_or_fx if fy is None else fy

            intrinsics.fill(0)
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
        return self[0, 0].item()

    @property
    def fy(self) -> float:
        return self[1, 1].item()

    @property
    def cx(self) -> float:
        return self[0, 2].item()

    @property
    def cy(self) -> float:
        return self[1, 2].item()

    @property
    def s(self) -> float:
        return self[0, 1].item()

    def rescale(self,
                scale_factor: float,
                scale_factor_y: Optional[float] = None,
                inplace: bool = True) -> 'Intrinsics':
        """
        When images that correspond to this intrinsics matrix are resized, the intrinsics should also be re-scaled
        to account for the image size change.
        This is because the intrinsics effectively just scales
        from the canonical screen space ([-1, 1]^2 with the image center in [0, 0])
        to the image screen space ([0, h] x [0, w] with the image center in [cx, cy]).
        Note: this is an inplace operation.
        See: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix

        Parameters
        ----------
            scale_factor:
                scaling factor for resizing the image screen space
            scale_factor_y:
                if specified, the image screen space will be scaled non-uniformly

        Returns
        -------
            The re-scaled intrinsics matrix
        """
        scale_factor_x = scale_factor
        scale_factor_y = scale_factor if scale_factor_y is None else scale_factor_y

        if inplace:
            intrinsics = self
        else:
            intrinsics = self.copy()

        intrinsics[0, 0] *= scale_factor_x  # fx
        intrinsics[1, 1] *= scale_factor_y  # fy
        intrinsics[0, 2] *= scale_factor_x  # cx
        intrinsics[1, 2] *= scale_factor_y  # cy
        # TODO: What about s?

        return intrinsics

    def crop(self,
             crop_left: int = 0,
             crop_top: int = 0) -> 'Intrinsics':
        """
        When images that correspond to this intrinsics matrix are cropped, the intrinsics should also be adjusted
        to account for the image size change.
        Essentially, it needs to be ensured that the principal point (in image screen space) after cropping still
        refers to the same point in the canonical screen space.
        This is done by simply subtracting the crop anchor point (top-left) from the intrinsics principal point.
        Note: this is an inplace operation.
        See: https://stackoverflow.com/questions/59477398/how-does-cropping-an-image-affect-camera-calibration-intrinsics

        Examples
        --------
            x: [0, 100] -> cx: 50
            y: [0, 200] -> cy: 100

            crop_x: [20, 90]
            crop_y: [20, 120]
            x: [0, 70]  ([20, 90]) -> cx: (50 - 20) = 30
            y: [0, 100] ([20, 120]) -> cy: (100 - 20) = 80

            => crop() only needs to know crop_left and crop_top

        Parameters
        ----------
            crop_left:
                How many pixels are cropped from left
            crop_top:
                How many pixels are cropped from top

        Returns
        -------
            The intrinsics matrix adjusted for the image cropping operation
        """

        # Ensure that the principal point after cropping is still the same point
        # Only cx, cy are affected

        self[0, 2] -= crop_left  # cx
        self[1, 2] -= crop_top  # cy

        return self

    def homogenize(self, invert: bool = False) -> np.ndarray:
        homogenized = np.eye(4)
        homogenized[:3, :3] = self

        if invert:
            homogenized = np.linalg.inv(homogenized)

        return homogenized

    def invert(self) -> np.ndarray:
        return np.linalg.inv(self)

    def __rmatmul__(self, other):
        if isinstance(other, Intrinsics):
            return super(Intrinsics, self).__rmatmul__(other)
        else:
            return other @ np.array(self)

    def __matmul__(self, other):
        # TODO: figure out why numpy operations automatically cast to Pose again
        if isinstance(other, Intrinsics):
            return super(Intrinsics, self).__matmul__(other)
        else:
            return np.array(self) @ other
