
import torch.nn as nn

from hprl.trainer.core import Train_Fn_Type
from hprl.trainer.common_trainer import CommonTrainer
from hprl.env import MultiAgentEnv
from hprl.policy import Policy
from hprl.replaybuffer import ReplayBuffer


class QLearningTranier(CommonTrainer):
    def __init__(self,
                 config,
                 train_fn: Train_Fn_Type,
                 env: MultiAgentEnv,
                 policy: Policy,
                 replay_buffer: ReplayBuffer) -> None:
        super().__init__(config, train_fn, env, policy, replay_buffer)

    def eval(self, episode: int):
        records = {}
        for ep in episode:
            state = self.env.reset()
            while True:
                raw_q_policy = self.policy.unwrapped()
                action = raw_q_policy.compute_action(state)
                ns, r, done, _ = self.env.step(action)
                state = ns[0]
                if done:
                    break
        return records
