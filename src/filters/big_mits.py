from .base_filter import BaseFilter
import numpy as np


class BigMits1(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Big Mits 1", [12, 11])

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)

        src = np.array([
  
        ])

        dst = src.copy()

        return super().add_pins(src, dst)