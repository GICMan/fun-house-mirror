from .base_filter import BaseFilter
import numpy as np

class NoodleArms1(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Noodle Arms 1",
                         [11,12,13,14,15,16,17,18,19,20])  # shoulders → wrists+fingers

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)

        # Shoulder → elbow → wrist
        ls = lmks['left shoulder']
        rs = lmks['right shoulder']

        le = lmks['left elbow']
        re = lmks['right elbow']

        lw = lmks['left wrist']
        rw = lmks['right wrist']

        # Finger roots / tips (your existing logic)
        li = lmks['left index']
        ri = lmks['right index']

        lp = lmks['left pinky']
        rp = lmks['right pinky']

        # Midpoints for curve control
        lu_mid = np.array([(ls.x + le.x)/2, (ls.y + le.y)/2])
        ll_mid = np.array([(le.x + lw.x)/2, (le.y + lw.y)/2])

        ru_mid = np.array([(rs.x + re.x)/2, (rs.y + re.y)/2])
        rl_mid = np.array([(re.x + rw.x)/2, (re.y + rw.y)/2])

        # Palm centers
        left_palm_x  = (lw.x + li.x + lp.x) / 3
        left_palm_y  = (lw.y + li.y + lp.y) / 3
        right_palm_x = (rw.x + ri.x + rp.x) / 3
        right_palm_y = (rw.y + ri.y + rp.y) / 3

        # Build src pins
        src = np.array([
            [ls.x, ls.y],       # 0 shoulder
            [le.x, le.y],       # 1 elbow
            [lw.x, lw.y],       # 2 wrist
            lu_mid,             # 3 upper-mid
            ll_mid,             # 4 lower-mid

            [rs.x, rs.y],       # 5 shoulder R
            [re.x, re.y],       # 6 elbow R
            [rw.x, rw.y],       # 7 wrist R
            ru_mid,             # 8 upper-mid R
            rl_mid,             # 9 lower-mid R

            [left_palm_x, left_palm_y],   # 10 palm L
            [li.x, li.y],                 # 11 index L
            [lp.x, lp.y],                 # 12 pinky L

            [right_palm_x, right_palm_y], # 13 palm R
            [ri.x, ri.y],                 # 14 index R
            [rp.x, rp.y],                 # 15 pinky R
        ])

        dst = src.copy()

        # Noodle intensity
        noodle = 40    # outward bend amount

        # --- apply curvature to LEFT arm ---
        # Upper arm vector and perpendicular
        v_up = np.array([le.x - ls.x, le.y - ls.y])
        perp_up = np.array([-v_up[1], v_up[0]])
        perp_up /= (np.linalg.norm(perp_up) + 1e-8)

        # Forearm vector and perpendicular
        v_low = np.array([lw.x - le.x, lw.y - le.y])
        perp_low = np.array([-v_low[1], v_low[0]])
        perp_low /= (np.linalg.norm(perp_low) + 1e-8)

        dst[3] = src[3] + perp_up * noodle
        dst[4] = src[4] + perp_low * noodle

        # --- RIGHT arm ---
        v_up = np.array([re.x - rs.x, re.y - rs.y])
        perp_up = np.array([-v_up[1], v_up[0]])
        perp_up /= (np.linalg.norm(perp_up) + 1e-8)

        v_low = np.array([rw.x - re.x, rw.y - re.y])
        perp_low = np.array([-v_low[1], v_low[0]])
        perp_low /= (np.linalg.norm(perp_low) + 1e-8)

        dst[8] = src[8] + perp_up * noodle
        dst[9] = src[9] + perp_low * noodle

        # Hand swelling (your original effect)
        swell_side = 50
        swell_down = 10

        dst[10] = [left_palm_x - swell_side, left_palm_y + swell_down]
        dst[11] = [li.x - swell_side, li.y + swell_down * 0.3]
        dst[12] = [lp.x - swell_side, lp.y + swell_down * 0.3]

        dst[13] = [right_palm_x + swell_side, right_palm_y + swell_down]
        dst[14] = [ri.x + swell_side, ri.y + swell_down * 0.3]
        dst[15] = [rp.x + swell_side, rp.y + swell_down * 0.3]

        return super().add_pins(src, dst)