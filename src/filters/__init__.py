# from .head_squish_1 import HeadSquish1
from .big_traps_1 import BigTraps1
# from .long_head import LongHead
from .sid_face import SidFace
from .small_waist import SmallWaist
from .noodle_arms import NoodleArms1
from .small_mouth import SmallMouth


def init_filters(frame_size):
    return [
        # HeadSquish1(frame_size),
        BigTraps1(frame_size),
        # LongHead(frame_size),
        SidFace(frame_size),
        SmallWaist(frame_size),
        NoodleArms1(frame_size),
        SmallMouth(frame_size),
    ]
