from .base_filter import BaseFilter
import numpy as np


class HeadSquish1(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Head Squish 1", [0])

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)
        box_size = 100
        src = np.array([
            [lmks['nose'].x, lmks['nose'].y + box_size],
            [lmks['nose'].x + box_size, lmks['nose'].y],
            [lmks['nose'].x, lmks['nose'].y - box_size],
            [lmks['nose'].x - box_size, lmks['nose'].y],
            [lmks['nose'].x, lmks['nose'].y],
        ])
        dst = src.copy()
        dst[-1] = [lmks['nose'].x, lmks['nose'].y - 50]

        return super().add_pins(src, dst)
