from .head_squish_1 import HeadSquish1
from .big_traps_1 import BigTraps1
from .dummy import Dummy1
from .sid_face import HammerHead
from .small_waist import SmallWaist


def init_filters(frame_size):
    return [
        HeadSquish1(frame_size),
        BigTraps1(frame_size),
        Dummy1(frame_size),
        HammerHead(frame_size),
        SmallWaist(frame_size)
    ]
