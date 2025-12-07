from .base_filter import BaseFilter
import numpy as np


class LongHead(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Long Head", [12, 11, 7, 8])

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)


        # Extract shoulder and ear coordinates
        ls = lmks['left shoulder']
        rs = lmks['right shoulder']
        le = lmks['left ear']
        re = lmks['right ear']

        left_shoulder = np.array([ls.x, ls.y])
        right_shoulder = np.array([rs.x, rs.y])
        left_ear = np.array([le.x, le.y])
        right_ear = np.array([re.x, re.y])

        # Approximate the neck positions (midpoints)
        neck_base = (left_shoulder + right_shoulder) / 2
        neck_top = (left_ear + right_ear) / 2

        # Vertical stretch factor
        stretch_amount = 40   # pixels (adjust)

        # Source points (shoulders + neck line)
        src = np.array([
            left_shoulder,
            right_shoulder,
            neck_base,
            neck_top,
        ])

        # Destination (stretch top-of-neck upward)
        dst = src.copy()
        dst[3] = neck_top - np.array([0, stretch_amount])  # pull up
        dst[2] = neck_base - np.array([0, stretch_amount * 3])  # slight lift

        return super().add_pins(src, dst)