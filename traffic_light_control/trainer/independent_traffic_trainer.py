import argparse
from typing import Any, Dict, List
import buffer
from envs.intersection import Intersection
import envs
import torch
import net
import numpy as np
from policy.dqn_n import DQNNew
from util.type import Transition


CITYFLOW_CONFIG_PATH = "config/config.json"
STATIC_CONFIG = "config/static_config.json"
MODEL_ROOT_DIR = "records/"

# env setting
MAX_TIME = 300
INTERVAL = 5

# agent setting
CAPACITY = 100000
LERNING_RATE = 1e-4
BATCH_SIZE = 2
DISCOUNT_FACTOR = 0.99
EPS_INIT = 1.0
EPS_MIN = 0.01
EPS_FRAME = 200000
UPDATE_PERIOD = 1000
STATE_SPACE = 6*4 + 12*2
ACTION_SPACE = 2

# exec setting
DATA_SAVE_PERIOD = 500


class IndependentTrainer():
    def __init__(self) -> None:
        super().__init__()

    def run(self):
        args = self.__parase_args()
        mode = args.mode
        model_file = args.model_file
        thread_num = args.thread_num
        episodes = args.episodes
        save = False if args.save == 0 else True
        resume = True if args.resume == 1 else False
        record_dir = args.record_dir

        print("mode : {}, model file :  {}, ep : {}, thread : {} ".format(
            mode, model_file, episodes, thread_num
        ) + "save : {}".format(save))
        if mode == "train":
            if resume and (model_file == "" or record_dir == ""):
                print("please input model file and record dir if want resume")
            env_config = {
                "max_time": MAX_TIME, "interval": INTERVAL,
                "thread_num": thread_num,
                "save_replay": False}
            buffer_config = {"capacity": CAPACITY}
            dqn_config = {
                "learning_rate": LERNING_RATE,
                "discount_factor": DISCOUNT_FACTOR,
                "eps_init": EPS_INIT,
                "eps_min": EPS_MIN,
                "eps_frame": EPS_FRAME,
                "update_period": UPDATE_PERIOD,
                "input_space": STATE_SPACE,
                "output_space": ACTION_SPACE,
                "device": torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")}
            train_config = {
                "env": env_config,
                "buffer": buffer_config,
                "policy": dqn_config,
                "num_episodes": episodes,
                "batch_size": BATCH_SIZE,
            }
            self.train(train_config)
        elif mode == "test":
            if resume and (model_file == "" or record_dir == ""):
                print("please input model file and record dir if want resume")
            env_config = {
                "max_time": MAX_TIME, "interval": INTERVAL,
                "thread_num": thread_num,
                "save_replay": True}
            buffer_config = {"capacity": CAPACITY}
            dqn_config = {
                "learning_rate": LERNING_RATE,
                "discount_factor": DISCOUNT_FACTOR,
                "eps_init": EPS_INIT,
                "eps_min": EPS_MIN,
                "eps_frame": EPS_FRAME,
                "update_period": UPDATE_PERIOD,
                "input_space": STATE_SPACE,
                "output_space": ACTION_SPACE,
                "device": torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")}
            train_config = {
                "env": env_config,
                "buffer": buffer_config,
                "policy": dqn_config,
                "num_episodes": episodes,
                "batch_size": BATCH_SIZE,
                "params_path": record_dir,
            }
            self.test()

    def train(self, train_config):

        env_config = train_config["env"]
        buffer_config = train_config["buffer"]
        dqn_config = train_config["policy"]
        num_episodes = train_config["num_episodes"]
        batch_size = train_config["batch_size"]

        id_ = "single_agent_simplest"
        # id_ = "single_agent_complete"
        env = envs.make(id_, env_config)

        policies = {}
        buffers = {}
        ids = env.intersection_ids()
        for id_ in ids:
            policies[id_] = self.__init_policy(dqn_config)
            buffers[id_] = self.__init_buffer(buffer_config)

        for episode in range(num_episodes):
            states = env.reset()
            cumulative_reward = 0.0
            while True:
                actions = {}
                for id_ in ids:
                    obs = states[id_]
                    policy = policies[id_]
                    if obs is None or policy is None:
                        print("intersection id is not exit {}".format(id_))
                    action = policy.compute_single_action(obs, True)
                    actions[id_] = action
                next_states, rewards, done, info = env.step(actions)

                for id_ in ids:
                    s = states[id_]
                    a = np.reshape(actions[id_], (1, 1))
                    r = np.reshape(rewards[id_], (1, 1))
                    ns = next_states[id_]
                    terminal = np.array([[0 if done else 1]])
                    buff = buffers[id_]
                    buff.store(Transition(s, a, r, ns, terminal))
                for id_ in ids:
                    buff = buffers[id_]
                    batch_data = buff.sample(batch_size)
                    policy = policies[id_]
                    policy.learn_on_batch(batch_data)

                for r in rewards.values():
                    cumulative_reward += r
                if done:
                    print(" episode : {}, reward : {}".format(
                        episode, cumulative_reward))
                    break

    def test(self, config):
        pass

    def eval(self, policies, env, num_episodes):
        states = env.reset()
        cumulative_reward = 0.0
        ids = env.intersection_ids()
        reward_history = []
        while True:
            actions = {}
            for id_ in ids:
                obs = states[id_]
                policy = policies[id_]
                if obs is None or policy is None:
                    print("intersection id is not exit {}".format(id_))
                action = policy.compute_single_action(obs, False)
                actions[id_] = action
            _, rewards, done, _ = env.step(actions)
            for r in rewards.values():
                cumulative_reward += r
            if done:
                reward_history.append(cumulative_reward)
                break
        return reward_history

    def __init_policy(self, dqn_config: Dict) -> DQNNew:
        net_id = "single_intersection"
        acting_net = net.get_net(net_id, dqn_config)
        target_net = net.get_net(net_id, dqn_config)
        return DQNNew(
            acting_net=acting_net,
            target_net=target_net,
            learning_rate=dqn_config["learning_rate"],
            discount_factor=dqn_config["discount_factor"],
            eps_init=dqn_config["eps_init"],
            eps_min=dqn_config["eps_min"],
            eps_frame=dqn_config["eps_frame"],
            update_period=dqn_config["update_period"],
            device=dqn_config["device"],
            action_space=dqn_config["output_space"],
        )

    def __init_buffer(self, buffer_config: Dict) -> buffer.ReplayBuffer:
        return buffer.ReplayBuffer(buffer_config["capacity"])

    def __parase_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m", "--mode", type=str, default="", required=True,
            help="mode of exec, include [train, test, static]")
        parser.add_argument(
            "-e", "--episodes", type=int, default=1,
            help="episode of exectue time"
        )
        parser.add_argument(
            "-mf", "--model_file", type=str,
            help="the path of model parameter file"
        )
        parser.add_argument(
            "-th", "--thread_num", type=int, default=1,
            help="thread number of simulator"
        )
        parser.add_argument(
            "-s", "--save", type=int, default=1,
            help="save params or not"
        )
        parser.add_argument(
            "-r", "--resume", type=int, default=0,
            help="resume training or not"
        )
        parser.add_argument(
            "-rd", "--record_dir", type=str, default="",
            help="resume dir if set resume with true"
        )

        return parser.parse_args()
