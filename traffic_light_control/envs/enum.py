from enum import Enum, auto


class Movement(Enum):
    STRAIGHT = "go_straight"
    LEFT = "turn_left"
    RIGHT = "turn_right"


class Stream(Enum):
    IN = auto()
    OUT = auto()
