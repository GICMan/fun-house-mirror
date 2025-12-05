from .head_squish_1 import HeadSquish1
from .big_traps_1 import BigTraps1
from .dummy import Dummy1
from .sid_face import HammerHead
from .small_waist import SmallWaist
from .big_mits import BigMits1

def init_filters(frame_size):
    return [
        HeadSquish1(frame_size),
        BigTraps1(frame_size),
        Dummy1(frame_size),
        HammerHead(frame_size),
        SmallWaist(frame_size),
        BigMits1(frame_size),
    ]
