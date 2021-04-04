from typing import Dict
from hprl.replaybuffer.replay_buffer import ReplayBufferTypes
from hprl.replaybuffer.common_buffer import CommonBuffer, MultiAgentCommonBuffer
from hprl.replaybuffer.prioritized_replay_buffer import PrioritizedReplayBuffer, MultiAgentPER


def build(config: Dict):
    buffer_type = config["type"]
    capacity = config["capacity"]
    if buffer_type == ReplayBufferTypes.Prioritized:
        alpha = config["alpha"]
        buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=alpha,
        )
    elif buffer_type == ReplayBufferTypes.Common:
        buffer = CommonBuffer(capacity)
    return buffer


def build_multi(config: Dict):
    buffer_type = config["type"]
    capacity = config["capacity"]
    if buffer_type == ReplayBufferTypes.Prioritized:
        alpha = config["alpha"]
        buffer = MultiAgentPER(
            capacity=capacity,
            alpha=alpha,
        )
    elif buffer_type == ReplayBufferTypes.Common:
        buffer = MultiAgentCommonBuffer(capacity)
    return buffer