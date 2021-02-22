from typing import List

from hprl.util.typing import Reward, TrainingRecord
from hprl.trainer.core import Train_Fn_Type, Trainer
from hprl.env import MultiAgentEnv
from hprl.policy import Policy
from hprl.replaybuffer import ReplayBuffer


class CommonTrainer(Trainer):
    def __init__(self,
                 config,
                 train_fn: Train_Fn_Type,
                 env: MultiAgentEnv,
                 policy: Policy,
                 replay_buffer: ReplayBuffer) -> None:
        super().__init__()

        self.config = config
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.train_fn = train_fn

    def train(self, episode: int):
        record = TrainingRecord({})
        for ep in range(episode):
            rewards = self.train_fn(
                self.env, self.policy,
                self.replay_buffer, self.config)

            unwarp_r = self.__unwrap_reward(rewards)

            print(unwarp_r)
            record.rewards[ep] = unwarp_r

        return record

    def eval(self, episode: int):
        record = TrainingRecord({})
        for ep in episode:
            state = self.env.reset()
            rewards = []

            while True:
                action = self.policy.compute_action(state)
                r = self.env.step(action)
                state = r.state
                rewards.append(r.reward)
                if r.terminal:
                    break

            record.rewards[ep] = self.__unwrap_reward(rewards)
        return record

    def save_checkpoint(self, checkpoint_dir: str):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_file: str):
        raise NotImplementedError

    def log_result(self, log_dir: str):
        raise NotImplementedError

    def __unwrap_reward(self, rewards: List[Reward]):
        length = len(rewards)
        ret = Reward(central=0.0, local={})
        if length == 0:
            return ret
        agents_id = self.env.get_agents_id()
        for k in agents_id:
            ret.local[k] = 0.0

        for r in rewards:
            ret.central += r.central
            for k, v in r.local.items():
                ret.local[k] += v

        ret.central /= length
        for k in r.local.keys():
            r.local[k] /= length
        return ret
