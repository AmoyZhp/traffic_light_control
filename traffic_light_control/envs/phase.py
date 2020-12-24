from enum import Enum, auto
from typing import List
import numpy as np

from util.enum import Movement


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

    def to_tensor(self) -> np.ndarray:
        tensor = np.zeros(12)
        for mov in self.movements:
            if mov == Movement.WE:
                tensor[0] = 1
            elif mov == Movement.WS:
                tensor[1] = 1
            elif mov == Movement.WN:
                tensor[2] = 1
            elif mov == Movement.EW:
                tensor[3] = 1
            elif mov == Movement.EN:
                tensor[4] = 1
            elif mov == Movement.ES:
                tensor[5] = 1
            elif mov == Movement.NS:
                tensor[6] = 1
            elif mov == Movement.NW:
                tensor[7] = 1
            elif mov == Movement.NE:
                tensor[8] = 1
            elif mov == Movement.SN:
                tensor[9] = 1
            elif mov == Movement.SE:
                tensor[10] = 1
            elif mov == Movement.SW:
                tensor[11] = 1
        return tensor
