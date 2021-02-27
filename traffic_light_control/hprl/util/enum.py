from enum import Enum, auto


class TrainnerTypes(Enum):
    IQL = "IQL"
    PPO = "PPO"
    IAC = "IAC"


class ReplayBufferTypes(Enum):
    Common = "Common"


class PolicyTypes(Enum):
    DQN = "DQN"


class AdvantageTypes(Enum):
    RewardToGO = auto()
    QMinusV = auto()
