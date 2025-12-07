from .base_filter import BaseFilter
import numpy as np


class SmallMouth(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Small Mouth", [9, 10])

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)

        left_mouth_x = lmks['mouth (left)'].x
        left_mouth_y = lmks['mouth (left)'].y

        right_mouth_x = lmks['mouth (right)'].x
        right_mouth_y = lmks['mouth (right)'].y

        # Midpoint
        mid_x = (left_mouth_x + right_mouth_x) / 2
        mid_y = (left_mouth_y + right_mouth_y) / 2

        src = np.array([
            [left_mouth_x,  left_mouth_y],
            [right_mouth_x, right_mouth_y],
            [left_mouth_x,  left_mouth_y - 40],
            [right_mouth_x, right_mouth_y - 40],
            [mid_x,         mid_y],
        ], dtype=np.float32)

        dst = src.copy()

        shrink = 5

        # Left mouth corner
        dst[0] = [
            mid_x + shrink * (left_mouth_x - mid_x),
            mid_y + shrink * (left_mouth_y - mid_y)
        ]

        # Right mouth corner
        dst[1] = [
            mid_x + shrink * (right_mouth_x - mid_x),
            mid_y + shrink * (right_mouth_y - mid_y)
        ]

        return super().add_pins(src, dst)