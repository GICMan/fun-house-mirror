from .base_filter import BaseFilter
import numpy as np


class BigTraps1(BaseFilter):
    def __init__(self, frame_size):
        super().__init__(frame_size, "Big Traps 1", [12, 11])

    def filter(self, raw_lmks):
        lmks = super().process_landmarks(raw_lmks)

        left_shoulder_x = lmks['left shoulder'].x
        left_shoulder_y = lmks['left shoulder'].y

        right_shoulder_x = lmks['right shoulder'].x
        right_shoulder_y = lmks['right shoulder'].y

        src = np.array([
            [left_shoulder_x, left_shoulder_y + 50],
            [right_shoulder_x, right_shoulder_y + 50],
            [left_shoulder_x, left_shoulder_y - 100],
            [right_shoulder_x, right_shoulder_y - 100],
            [left_shoulder_x, left_shoulder_y],
            [right_shoulder_x, right_shoulder_y],
        ])

        dst = src.copy()
        dst[-2] = [left_shoulder_x - 150, left_shoulder_y]
        dst[-1] = [right_shoulder_x + 150, right_shoulder_y]

        return super().add_pins(src, dst)

# def effect_2(landmarks):
#     left_shoulder = landmarks[12]
#     right_shoulder = landmarks[11]
#
#     left_shoulder_x = left_shoulder[0] * frame_width
#     left_shoulder_y = left_shoulder[1] * frame_height
#
#     right_shoulder_x = right_shoulder[0] * frame_width
#     right_shoulder_y = right_shoulder[1] * frame_height
#
#     # box_size = 100
#     src = np.array([
#         [0, 0],
#         [0, frame_height - 1],
#         [frame_width - 1, 0],
#         [frame_width - 1, frame_height - 1],
#         [left_shoulder_x, left_shoulder_y + 50],
#         [right_shoulder_x, right_shoulder_y + 50],
#         [left_shoulder_x, left_shoulder_y - 100],
#         [right_shoulder_x, right_shoulder_y - 100],
#         [left_shoulder_x, left_shoulder_y],
#         [right_shoulder_x, right_shoulder_y],
#     ])
#
#     dst = src.copy()
#     dst[-2] = [left_shoulder_x - 150, left_shoulder_y]
#     dst[-1] = [right_shoulder_x + 150, right_shoulder_y]
#
#     return src, dst
