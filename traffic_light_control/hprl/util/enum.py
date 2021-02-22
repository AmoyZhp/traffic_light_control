from enum import Enum, auto


class TrainnerTypes(Enum):
    IQL = "IQL"


class ReplayBufferTypes(Enum):
    Common = "Common"


class PolicyTypes(Enum):
    DQN = "DQN"
