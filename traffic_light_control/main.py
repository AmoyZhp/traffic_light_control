import gym
from policy import net
import hprl
import hprl.policy.dqn as dqn
import hprl.policy.actor_critic as ac
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
    config, model = ac.get_ac_default_config()
    models = {
        local_ids[0]: model
    }
    # trainer = hprl.load_trainer(
    #     env=env,
    #     models=models,
    #     checkpoint_dir="records",
    #     checkpoint_file="ckpt_100.pth",
    # )
    trainer = hprl.create_trainer(
        config=config,
        env=env,
        models=models,
    )
    episode = 500
    train_records = trainer.train(episode)
    trainer.eval(10)
    train_records = trainer.train(episode)
    trainer.eval(10)
    train_records = trainer.train(episode)
    trainer.eval(10)
    trainer.log_result("records")


if __name__ == "__main__":
    new_run()
