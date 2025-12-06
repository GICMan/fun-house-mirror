from .base_filter import BaseFilter
import numpy as np


class NoodleArms1(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Noodle Arms 1", [15, 16, 17, 18, 19, 20])

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)


        # Wrists
        lw_x, lw_y = lmks['left wrist'].x, lmks['left wrist'].y
        rw_x, rw_y = lmks['right wrist'].x, lmks['right wrist'].y

        # Finger roots (good palm references)
        lir_x, lir_y = lmks['left index'].x, lmks['left index'].y
        rir_x, rir_y = lmks['right index'].x, lmks['right index'].y

        lpr_x, lpr_y = lmks['left pinky'].x, lmks['left pinky'].y
        rpr_x, rpr_y = lmks['right pinky'].x, lmks['right pinky'].y

        # Fingertips
        li_x, li_y = lmks['left index'].x, lmks['left index'].y
        ri_x, ri_y = lmks['right index'].x, lmks['right index'].y

        lp_x, lp_y = lmks['left pinky'].x, lmks['left pinky'].y
        rp_x, rp_y = lmks['right pinky'].x, lmks['right pinky'].y

        # palm
        left_palm_x  = (lw_x + lir_x + lpr_x) / 3
        left_palm_y  = (lw_y + lir_y + lpr_y) / 3

        right_palm_x = (rw_x + rir_x + rpr_x) / 3
        right_palm_y = (rw_y + rir_y + rpr_y) / 3


        src = np.array([
            [lw_x, lw_y],       # 0 left wrist anchor
            [left_palm_x, left_palm_y],   # 1 left palm center
            [li_x, li_y],       # 2 left index tip
            [lp_x, lp_y],       # 3 left pinky tip

            [rw_x, rw_y],       # 4 right wrist anchor
            [right_palm_x, right_palm_y], # 5 right palm center
            [ri_x, ri_y],       # 6 right index tip
            [rp_x, rp_y],       # 7 right pinky tip
        ])

        dst = src.copy()


        swell_side = 50      # how wide the palm spreads
        swell_down = 10      # how tall the palm expands downward

   
        dst[1] = [left_palm_x - swell_side, left_palm_y + swell_down]   # palm center
        dst[2] = [li_x - swell_side, li_y + swell_down * 0.3]           # index tip
        dst[3] = [lp_x - swell_side, lp_y + swell_down * 0.3]           # pinky tip


        dst[5] = [right_palm_x + swell_side, right_palm_y + swell_down]
        dst[6] = [ri_x + swell_side, ri_y + swell_down * 0.3]
        dst[7] = [rp_x + swell_side, rp_y + swell_down * 0.3]

        # Wrist anchors (0 and 4) stay fixed

        return super().add_pins(src, dst)