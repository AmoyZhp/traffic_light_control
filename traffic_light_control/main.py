import gym
from policy import net
import hprl
import hprl.policy.dqn as dqn
import trainer
import cityflow


def static_test():
    path = "config/hangzhou_1x1_bc-tyc_18041607_1h/config.json"
    max_time = 360
    eng = cityflow.Engine(path, thread_num=1)
    for t in range(max_time):
        eng.next_step()


def rl_train():
    id_ = "independent"
    tr = trainer.get_trainer(id_, {})
    tr.run()


def new_run():
    env = hprl.GymWrapper(gym.make("CartPole-v1"))
    local_ids = env.get_agents_id()
    config, model = dqn.get_default_config()
    trainer = hprl.create_trainer(
        config=config,
        env=env,
        models={
            local_ids[0]: model
        }
    )
    episode = 1000
    train_records = trainer.train(episode)
    print(train_records)


if __name__ == "__main__":
    new_run()
