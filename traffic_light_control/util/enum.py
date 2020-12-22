from enum import Enum, auto


class Movement(Enum):
    # straight
    WE = auto()
    EW = auto()
    NS = auto()
    SN = auto()
    # right
    WS = auto()
    EN = auto()
    NW = auto()
    SE = auto()
    # left
    WN = auto()
    ES = auto()
    SW = auto()
    NE = auto()


class Location(Enum):
    W = auto()
    E = auto()
    N = auto()
    S = auto()


class TrafficStreamDirection(Enum):
    STRAIGHT = auto()
    LEFT = auto()
    RIGHT = auto()


class GraphDirection(Enum):
    IN = auto()
    OUT = auto()
