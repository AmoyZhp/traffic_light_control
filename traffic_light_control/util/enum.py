from enum import Enum, auto


class TrafficStreamDirection(Enum):
    STRAIGHT = "go_straight"
    LEFT = "turn_left"
    RIGHT = "turn_right"


class GraphDirection(Enum):
    IN = auto()
    OUT = auto()
