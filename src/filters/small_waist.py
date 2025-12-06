from .base_filter import BaseFilter
import numpy as np


class SmallWaist(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Small Waist", [23, 24])

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)

        lhx, lhy = lmks["left hip"].x, lmks["left hip"].y
        rhx, rhy = lmks["right hip"].x, lmks["right hip"].y

        waist_in = 120

        src = np.array([
            [lhx, lhy],              # left hip
            [rhx, rhy],              # right hip

            [lhx, lhy - 120],        # left upper waist (above)
            [rhx, rhy - 120],        # right upper waist

            [lhx, lhy + 120],        # left lower waist (below)
            [rhx, rhy + 120],        # right lower waist
        ])

        dst = src.copy()

        # pull hips inward
        dst[0] = [lhx + waist_in, lhy]     
        dst[1] = [rhx - waist_in, rhy]     

        # upper waist slightly pulled inward to keep curve smooth
        dst[2] = [lhx + waist_in * 0.6, lhy - 120]
        dst[3] = [rhx - waist_in * 0.6, rhy - 120]

        # lower waist slightly inward
        dst[4] = [lhx + waist_in * 0.5, lhy + 120]
        dst[5] = [rhx - waist_in * 0.5, rhy + 120]

        return super().add_pins(src, dst)
