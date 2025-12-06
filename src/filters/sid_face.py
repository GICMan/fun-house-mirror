from .base_filter import BaseFilter
import numpy as np

class SidFace(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Sid Face", [0, 2, 5, 7, 8])

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)

        nose_x, nose_y = lmks["nose"].x, lmks["nose"].y
        lx, ly = lmks["left eye"].x, lmks["left eye"].y
        rx, ry = lmks["right eye"].x, lmks["right eye"].y
        lex, ley = lmks["left ear"].x, lmks["left ear"].y
        rex, rey = lmks["right ear"].x, lmks["right ear"].y

        eye_stretch = 25
        ear_stretch = 35

        src = np.array([
            [lx, ly],                # left eye
            [rx, ry],                # right eye
            [lex, ley],              # left ear
            [rex, rey],              # right ear

            [nose_x, nose_y - 70],   # forehead
            [nose_x, nose_y + 70],   # chin
            [nose_x - 100, nose_y],  # left cheek
            [nose_x + 100, nose_y],  # right cheek
        ])

        dst = src.copy()

        # stretch eyes outward
        dst[0] = [lx - eye_stretch, ly]
        dst[1] = [rx + eye_stretch, ry]

        # stretch head outward
        dst[2] = [lex - ear_stretch, ley]
        dst[3] = [rex + ear_stretch, rey]

        # forehead/chin pulled only slightly (stability)
        dst[4] = [nose_x, nose_y - 80]
        dst[5] = [nose_x, nose_y + 90]

        # cheeks push outward a bit (prevents the fold)
        dst[6] = [nose_x - (ear_stretch * 0.9), nose_y]
        dst[7] = [nose_x + (ear_stretch * 0.9), nose_y]

        return super().add_pins(src, dst)