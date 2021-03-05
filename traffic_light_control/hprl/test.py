from hprl.util.enum import TrainnerTypes
from hprl.env import GymWrapper
from hprl.policy.single import ppo, dqn, actor_critic
from hprl.trainer.factory import create_trainer
import gym


def test(
    gym_env_id: str = "CartPole-v1",
    trainer: TrainnerTypes = TrainnerTypes.IQL,
    episodes: int = 100,
):
    env = GymWrapper(gym.make(gym_env_id))
    agents_id = env.get_agents_id()
    if trainer == TrainnerTypes.PPO:
        config, model = ppo.get_ppo_default_config()
    elif trainer == TrainnerTypes.IQL:
        config, model = dqn.get_default_config()
    elif trainer == TrainnerTypes.IAC:
        config, model = actor_critic.get_ac_default_config()
    models = {agents_id[0]: model}
    trainer = create_trainer(
        config=config,
        env=env,
        models=models,
    )
    trainer.train(episodes)
    trainer.eval(10)
