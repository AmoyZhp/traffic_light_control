from envs.intersection import Intersection
from envs.phase import Phase


class State():
    def __init__(self, intersection: Intersection):
        self.intersection = intersection

    def to_tensor(self):
        return self.intersection.to_tensor()
