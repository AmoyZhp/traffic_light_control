from enum import Enum, auto


class TrainnerTypes(Enum):
    IQL = "IQL"
    PPO = "PPO"
    IAC = "IAC"
    VDN = "VDN"
    COMA = "COMA"
    IQL_PS = "IQL_PS"
    PPO_PS = "PPO_PS"


class ReplayBufferTypes(Enum):
    Common = "Common"


class AdvantageTypes(Enum):
    RewardToGO = auto()
    QMinusV = auto()
