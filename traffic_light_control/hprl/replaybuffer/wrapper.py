from typing import Dict, List

from hprl.replaybuffer.replay_buffer import (MAgentReplayBuffer, ReplayBuffer,
                                             ReplayBufferTypes)
from hprl.util.typing import SampleBatch, Transition, TransitionTuple


class IndependentWrapper(MAgentReplayBuffer):
    def __init__(
        self,
        type: ReplayBufferTypes,
        buffers: Dict[str, ReplayBuffer],
    ) -> None:
        self._type = type
        self._buffers = buffers

    def store(self, data: Transition, priorities: Dict[str, float] = {}):
        for id, buffer in self._buffers.items():
            trans = TransitionTuple(
                state=data.state.local[id],
                action=data.action.local[id],
                reward=data.reward.local[id],
                next_state=data.next_state.local[id],
                terminal=data.terminal.local[id],
            )
            p = None
            if priorities:
                p = priorities[id]
            buffer.store(trans, p)

    def sample(self, batch_size: int, beta: float) -> Dict[str, SampleBatch]:
        samples = {}
        for id, buffer in self._buffers.items():
            samples[id] = buffer.sample(batch_size, beta)
        return samples

    def update_priorities(
        self,
        idxes: Dict[str, List[int]],
        priorities: Dict[str, List[float]],
    ):
        for id, buffer in self._buffers.items():
            buffer.update_priorities(
                idxes=idxes[id],
                priorities=priorities[id],
            )

    def clear(self):
        for buffer in self._buffers.values():
            buffer.clear()

    def get_weight(self):
        weight = {}
        for id, buffer in self._buffers.items():
            weight[id] = buffer.get_weight()
        return weight

    def get_config(self):
        config = list(self._buffers.values())[0].get_config()
        return config

    def set_weight(self, weight):
        for id, buffer in self._buffers.items():
            buffer.set_weight(weight[id])

    @property
    def type(self):
        return self._type
