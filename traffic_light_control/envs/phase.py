from enum import Enum, auto
from typing import List


class Movement(Enum):
    WE = auto()
    EW = auto()
    NS = auto()
    SN = auto()


class Direction(Enum):
    W = auto()
    E = auto()
    N = auto()
    S = auto()
    STRAIGHT = auto()
    LEFT = auto()
    RIGHT = auto()


class Phase():
    """Phase
        表示十字路口灯的状态
    """

    def __init__(self, movements: List[Movement]):
        """初始化函数

        Args:
            movements (list): 是枚举类型 Movement 的列表，比如可能是 [WE, EW] 表示其实是一个横向通行的路口
        """
        self.movements = movements

    def get_movements(self) -> List[Movement]:
        return self.movements

    def add_movements(self, movement: Movement):
        self.movements.append(movement)

    def equal(self, target) -> bool:
        if len(self.movements) != len(target.movements):
            return False
        for (s, t) in zip(self.movements, target.movements):
            if s != t:
                return False
        return True
