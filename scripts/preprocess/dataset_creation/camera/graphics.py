from dataclasses import dataclass
from typing import Union, Tuple, Optional

import numpy as np
from elias.config import Config

from .matrix import Intrinsics, Pose
from .vector.vector_base import unpack_nd_params


@dataclass
class Dimensions(Config, tuple):
    w: int
    h: int

    def __init__(self, w: Union[int, 'Dimensions', Tuple[int, int]], h: Optional[int] = None):
        if isinstance(w, dict):
            # TODO: Is there a better way to match a dictionary against variable names?
            h = w['h']
            w = w['w']

        w, h = unpack_nd_params(2, w, h)
        assert w is not None and h is not None, f"Both w and h have to be given, got: {w}, {h}"

        self.w = w
        self.h = h
        # TODO: Is this a proper dataclass if the super constructor is not called?
        super(Dimensions, self).__init__()  # TODO: Do we need that?

    def __new__(cls, w: Union[int, 'Dimensions', Tuple[int, int]], h: Optional[int] = None):
        if isinstance(w, dict):
            h = w['h']
            w = w['w']

        w, h = unpack_nd_params(2, w, h)
        return tuple.__new__(Dimensions, (w, h))

    # def __getitem__(self, idx: int) -> int:
    #     assert 0 <= idx <= 1, "idx of dimension has to be 0 or 1"
    #
    #     if idx == 0:
    #         return self.w
    #     elif idx == 1:
    #         return self.h

    @property
    def x(self) -> int:
        return self.w

    @property
    def y(self) -> int:
        return self.h
