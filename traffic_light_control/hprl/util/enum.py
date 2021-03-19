from enum import Enum, auto
from hprl.replaybuffer.prioritized_replay_buffer import PrioritizedReplayBuffer


class TrainnerTypes(Enum):
    IQL = "IQL"
    PPO = "PPO"
    IAC = "IAC"
    VDN = "VDN"
    COMA = "COMA"
    QMIX = "QMIX"


class ReplayBufferTypes(Enum):
    Common = "Common"
    Prioritized = "PER"


class AdvantageTypes(Enum):
    RewardToGO = auto()
    QMinusV = auto()
