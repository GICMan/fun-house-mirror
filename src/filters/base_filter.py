from collections import namedtuple
import numpy as np

Point = namedtuple('Point', 'x y')

LANDMARK_LABELS = [
    "nose",
    "left eye (inner)",
    "left eye",
    "left eye (outer)",
    "right eye (inner)",
    "right eye",
    "right eye (outer)",
    "left ear",
    "right ear",
    "mouth (left)",
    "mouth (right)",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left pinky",
    "right pinky",
    "left index",
    "right index",
    "left thumb",
    "right thumb",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
    "left heel",
    "right heel",
    "left foot index",
    "right foot index",
]


class BaseFilter:
    def __init__(self, frame_size, name, lmk_deps):
        self.frame_width, self.frame_height = frame_size
        self.name = name
        self.lmk_deps = lmk_deps

    def process_landmarks(self, lmks):
        lmk_out = {}
        for lmk_idx in self.lmk_deps:
            label = LANDMARK_LABELS[lmk_idx]
            lmk = lmks[lmk_idx]
            lmk_out[label] = Point(
                lmk[0] * self.frame_width, lmk[1] * self.frame_height)

        return lmk_out

    def add_pins(self, src, dst):
        pins = [
            [0, 0],
            [0, self.frame_height - 1],
            [0, self.frame_height / 2],
            [self.frame_width - 1, self.frame_height / 2],
            [self.frame_width - 1, 0],
            [self.frame_width / 2, 0],
            [self.frame_width - 1, self.frame_height - 1],
            [self.frame_width / 2, self.frame_height - 1],
        ]
        return np.append(src, pins, axis=0), np.append(dst, pins, axis=0)
