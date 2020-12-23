import argparse
import datetime
import os
import cityflow
from typing import Any, Dict, List

from numpy.core.records import record
from agent import dqn
from basis.action import Action
import buffer
from envs.intersection import Intersection
import envs
import torch
import net
import numpy as np
from policy.dqn_n import DQNNew
import util
from util.type import Transition


CITYFLOW_CONFIG_PATH = "config/config.json"
STATIC_CONFIG = "config/static_config.json"
RECORDS_ROOT_DIR = "records/"
CITYFLOW_CONFIG_ROOT_DIR = "config/"

# env setting
MAX_TIME = 360
INTERVAL = 5

# agent setting
CAPACITY = 200000
LERNING_RATE = 1e-4
BATCH_SIZE = 256
DISCOUNT_FACTOR = 0.99
EPS_INIT = 1.0
EPS_MIN = 0.01
EPS_FRAME = 300000
UPDATE_PERIOD = 1000
STATE_SPACE = 6*4 + 12*2
ACTION_SPACE = 2


# exec setting
DATA_SAVE_PERIOD = 20
EVAL_NUM_EPISODE = 50
SAVED_THRESHOLD = -100.0


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

        env_id = "single_complete_1x1"
        cityflow_config_dir = CITYFLOW_CONFIG_ROOT_DIR + env_id + "/"

        print(" mode : {}, model file :  {}, ep : {}, thread : {} ".format(
            mode, model_file, episodes, thread_num
        ) + "save : {}".format(save))

        if mode == "train":
            if resume and (model_file == "" or record_dir == ""):
                print("please input model file and record dir if want resume")

            env_config = {
                "id": env_id,
                "cityflow_config_dir": cityflow_config_dir,
                "max_time": MAX_TIME,
                "interval": INTERVAL,
                "thread_num": thread_num,
                "save_replay": False,
            }

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
                "saved": save,
                "batch_size": BATCH_SIZE,
                "data_saved_period": DATA_SAVE_PERIOD,
                "eval_num_episodes": EVAL_NUM_EPISODE,
            }
            self.train(train_config)
        elif mode == "test":
            if model_file == "" or record_dir == "":
                print("please input model file and record dir if want resume")

            test_config = self.__load_test_config(
                model_file,
                RECORDS_ROOT_DIR + record_dir + "/")
            self.test(test_config)
        elif mode == "static":
            env_config = {
                "id": env_id,
                "cityflow_config_dir": cityflow_config_dir,
                "max_time": MAX_TIME,
                "interval": INTERVAL,
                "thread_num": thread_num,
                "save_replay": True,
            }
            self.static_test(env_config)
        else:
            print(" invalid mode {} !!!".format(mode))

    def train(self, train_config):

        env_config = train_config["env"]
        buffer_config = train_config["buffer"]
        dqn_config = train_config["policy"]
        num_episodes = train_config["num_episodes"]
        eval_num_episodes = train_config["eval_num_episodes"]
        batch_size = train_config["batch_size"]
        saved = train_config["saved"]
        data_saved_period = train_config["data_saved_period"]

        # 初始化环境
        env = envs.make(env_config)

        # 初始化策略
        policies = {}
        buffers = {}
        local_loss = {}
        ids = env.intersection_ids()
        for id_ in ids:
            policies[id_] = self.__init_policy(dqn_config)
            buffers[id_] = self.__init_buffer(buffer_config)
            local_loss[id_] = 0.0

        # 创建和本次训练相应的保存目录
        record_dir = self.__create_record_dir(RECORDS_ROOT_DIR)

        # 保存每轮 episode 完成后的 reward 奖励
        central_reward_record = {}

        # 保存每次评估时的评估奖励，和评估奖励的平均值
        eval_reward_recrod = {
            "all": {},
            "mean": {}
        }

        ep_begin = 1

        for episode in range(ep_begin, num_episodes + ep_begin):
            states = env.reset()
            central_cumulative_reward = 0.0
            for k in local_loss.keys():
                local_loss[k] = 0.0
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
                    local_loss[id_] += policy.learn_on_batch(
                        batch_data)
                states = next_states
                for r in rewards.values():
                    central_cumulative_reward += r
                if done:
                    print(" ====== episode {} ======".format(episode))
                    print(" central reward : {:.3f}".format(
                        central_cumulative_reward))
                    print(" loss :")
                    for id_, loss in local_loss.items():
                        print("      id {}, loss {:.3f}".format(
                            id_, loss / len(local_loss)
                        ))
                    print("=========================")
                    central_reward_record[episode] = central_cumulative_reward
                    break
            if (episode % data_saved_period == 0):
                eval_rewards = self.eval_(
                    policies=policies,
                    env=env,
                    num_episodes=eval_num_episodes,
                )
                eval_reward_mean = eval_rewards["mean"]
                print(" episode : {}, eval mean reward is {:.3f}".format(
                    episode, eval_reward_mean
                ))
                if saved and eval_reward_mean > SAVED_THRESHOLD:
                    eval_reward_recrod["all"][episode] = eval_rewards["all"]
                    eval_reward_recrod["mean"][episode] = eval_rewards["mean"]
                    param_file_name = "params_{}.pth".format(episode)
                    param_file = record_dir + "params/" + param_file_name
                    exec_params = {
                        "batch_size": batch_size,
                        "episode": episode,
                    }
                    self.__snapshot_params(
                        env, policies, buffers, exec_params,
                        param_file
                    )
                    self.__snapshot_training(
                        record_dir, central_reward_record, eval_reward_recrod)
        if saved:
            param_file_name = "params_final.pth"
            param_file = record_dir + param_file_name
            exec_params = {
                "batch_size": batch_size,
                "episode": num_episodes,
            }
            self.__snapshot_params(
                env, policies, buffers,
                exec_params,
                param_file
            )
            self.__snapshot_training(
                record_dir, central_reward_record, eval_reward_recrod)

            test_config = self.__load_test_config(
                param_file_name,
                record_dir)
            self.test(test_config)
            os.system("cp ../replay/replay_roadnet.json {}".format(
                record_dir
            ))
            os.system("cp ../replay/replay.txt {}".format(
                record_dir
            ))

    def eval_(self, policies, env, num_episodes):

        ids = env.intersection_ids()
        reward_history = []
        for _ in range(num_episodes):
            states = env.reset()
            cumulative_reward = 0.0
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
        reward_record = {}
        reward_record["all"] = reward_history
        reward_record["mean"] = sum(reward_history) / len(reward_history)
        return reward_record

    def test(self, test_config):

        env_config = test_config["env"]
        policy_params = test_config["policy"]
        num_episodes = test_config["num_episodes"]

        # 初始化环境
        env = envs.make(env_config)

        # 初始化策略
        policies = {}
        ids = env.intersection_ids()
        for id_ in ids:
            policy_config = policy_params[id_]["config"]
            policy_weight = policy_params[id_]["weight"]
            policy_config["device"] = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            policies[id_] = self.__init_policy(policy_config)
            policies[id_].set_weight(policy_weight)

        reward_history = {}
        ids = env.intersection_ids()

        for eps in range(num_episodes):
            states = env.reset()
            cumulative_reward = 0.0
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
                    print("In test mode, episodes {}, reward is {:.3f}".format(
                        eps, cumulative_reward))
                    reward_history[eps] = cumulative_reward
                    break

    def static_test(self, env_config):
        # 初始化环境
        env = envs.make(env_config)
        cumulative_reward = 0.0
        while True:
            actions = {}
            _, rewards, done, _ = env.step(actions)
            for r in rewards.values():
                cumulative_reward += r
            if done:
                break
        print("total reward is {}".format(cumulative_reward))

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
            state_space=dqn_config["input_space"]
        )

    def __init_buffer(self, buffer_config: Dict) -> buffer.ReplayBuffer:
        return buffer.ReplayBuffer(buffer_config["capacity"])

    def __snapshot_params(self, env, polices, buffer,
                          exec_params, params_file):
        env_params = {
            "max_time": env.max_time,
            "interval": env.interval,
            "id": env.id_
        }

        polices_params = {}
        for id_, p in polices.items():
            polices_params[id_] = {
                "weight": p.get_weight(),
                "config": p.get_config()
            }

        buffer_params = {}
        for id_, b in buffer.items():
            buffer_params[id_] = {
                "weight": b.get_weight(),
                "config": b.get_config(),
            }

        params = {
            "env": env_params,
            "policy": polices_params,
            "buffer": buffer_params,
            "exec": exec_params,
        }

        torch.save(params, params_file)

    def __snapshot_training(self, record_dir,
                            reward_record, eval_reward_record):
        eval_mean_recrod = eval_reward_record["mean"]
        eval_reward_record = eval_reward_record["all"]
        saved_data = {
            "reward": reward_record,
            "eval_reward_mean": eval_mean_recrod,
            "eval_reward_all": eval_reward_record,
        }
        saved_data_file_name = "exp_result.txt"
        result_file = record_dir + saved_data_file_name
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(str(saved_data))
        self.__store_record_img(record_dir, result_file)

    def __create_record_dir(self, root_record, last_record="") -> str:
        # 创建的目录
        date = datetime.datetime.now()
        sub_dir = "record_{}_{}_{}_{}_{}_{}/".format(
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second
        )
        path = root_record + sub_dir
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print("create record folder error , path exist : ", path)
        param_path = path + "params/"
        if not os.path.exists(param_path):
            os.mkdir(param_path)
        else:
            print("create record folder error , path exist : ", param_path)
        return path

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

    def __store_record_img(self, record_dir, data_file):
        data = {}
        with open(data_file, "r", encoding="utf-8") as f:
            data = eval(f.read())
        episodes = []
        rewards = []
        for ep, r in data["reward"].items():
            episodes.append(int(ep))
            rewards.append(float(r))
        util.savefig(
            episodes, rewards, x_lable="episodes",
            y_label="reward", title="rewards",
            img=record_dir+"reward.png")

        episodes = []
        mean_eval_reward = []

        for ep, r in data["eval_reward_mean"].items():
            episodes.append(int(ep))
            mean_eval_reward.append(r)
        util.savefig(
            episodes, mean_eval_reward, x_lable="episodes",
            y_label="reward", title="rewards",
            img=record_dir+"eval_reward_mean.png")
        """
        episodes = []
        loss = []
        for ep, r in data["loss"].items():
            episodes.append(int(ep))
            loss.append(float(r))
        util.savefig(
            episodes, loss, x_lable="episodes",
            y_label="loss", title="loss",
            img=record_dir+"loss.png")
        """

    def __load_test_config(self, model_file, record_dir):
        params = torch.load(record_dir + model_file)
        env_params = params["env"]
        cityflow_config_dir = CITYFLOW_CONFIG_ROOT_DIR + env_params["id"] + "/"
        env_config = {
            "id": env_params["id"],
            "cityflow_config_dir": cityflow_config_dir,
            "max_time": env_params["max_time"],
            "interval": env_params["interval"],
            "thread_num": 1,
            "save_replay": True
        }

        policy_params = params["policy"]
        test_config = {
            "env": env_config,
            "policy": policy_params,
            "record_dir": record_dir,
            "num_episodes": 1,
        }

        return test_config
