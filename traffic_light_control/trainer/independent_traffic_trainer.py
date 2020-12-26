import argparse
import datetime
import os
import time
import json
import buffer
import envs
import torch
import numpy as np
import policy
import util
from util.type import Transition


RECORDS_ROOT_DIR = "records/"
PARAMS_DIR = "params/"
FIGS_DIR = "figs/"
DATA_DIR = "data/"

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
SAVED_THRESHOLD = -30.0

ENV_ID = "hangzhou_1x1_bc-tyc_18041607_1h"


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

        cityflow_config_dir = CITYFLOW_CONFIG_ROOT_DIR + ENV_ID + "/"

        print(" mode : {}, model file :  {}, ep : {}, thread : {} ".format(
            mode, model_file, episodes, thread_num
        ) + "save : {}".format(save))

        if mode == "train":
            if resume and (model_file == "" or record_dir == ""):
                print("please input model file and record dir if want resume")

            env_config = {
                "id": ENV_ID,
                "cityflow_config_dir": cityflow_config_dir,
                "max_time": MAX_TIME,
                "interval": INTERVAL,
                "thread_num": thread_num,
                "save_replay": False,
            }

            buffer_config = {
                "id": "basis",
                "capacity": CAPACITY}

            policy_config = {
                "id": "DQN",
                "learning_rate": LERNING_RATE,
                "discount_factor": DISCOUNT_FACTOR,
                "net_id": "single_intersection",
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
                "policy": policy_config,
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
                "id": ENV_ID,
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
        policy_config = train_config["policy"]
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
        local_reward = {}
        local_record = {}
        ids = env.intersection_ids()
        for id_ in ids:
            policies[id_] = policy.get_policy(
                policy_config["id"], policy_config)
            buffers[id_] = buffer.get_buffer(
                buffer_config["id"],  buffer_config)
            local_loss[id_] = 0.0
            local_reward[id_] = {}
            local_record[id_] = {
                "loss": {},
                "reward": {},
            }

        # 创建和本次训练相应的保存目录
        record_dir = self.__create_record_dir(RECORDS_ROOT_DIR)
        data_dir = record_dir + "data/"
        params_dir = record_dir + "params/"
        self.__record_init_config(data_dir, train_config)

        # 保存每轮 episode 完成后的 reward 奖励
        central_reward_record = {}
        central_record = {}

        # 保存每次评估时的评估奖励，和评估奖励的平均值
        eval_reward_recrod = {
            "all": {},
            "mean": {}
        }

        ep_begin = 1

        for episode in range(ep_begin, num_episodes + ep_begin):
            ep_begin_time = time.time()
            states = env.reset()
            central_cumulative_reward = 0.0
            sim_time_cost = 0.0
            learn_time_cost = 0.0
            for k in ids:
                local_loss[k] = 0.0
                local_reward[k] = 0.0
            while True:
                actions = {}
                for id_ in ids:
                    obs = states[id_]
                    policy_ = policies[id_]
                    if obs is None or policy_ is None:
                        print("intersection id is not exit {}".format(id_))
                    action = policy_.compute_single_action(obs, True)
                    actions[id_] = action
                sim_begin_time = time.time()
                next_states, rewards, done, info = env.step(actions)
                sim_time_cost += time.time() - sim_begin_time

                for id_ in ids:
                    s = states[id_]
                    a = np.reshape(actions[id_], (1, 1))
                    r = np.reshape(rewards[id_], (1, 1))
                    ns = next_states[id_]
                    terminal = np.array([[0 if done else 1]])
                    buff = buffers[id_]
                    buff.store(Transition(s, a, r, ns, terminal))

                learn_begin_time = time.time()
                for id_ in ids:
                    buff = buffers[id_]
                    batch_data = buff.sample(batch_size)
                    policy_ = policies[id_]
                    local_loss[id_] += policy_.learn_on_batch(
                        batch_data)
                learn_time_cost += time.time() - learn_begin_time
                states = next_states
                for id_, r in rewards.items():
                    central_cumulative_reward += r
                    local_reward[id_] += r
                if done:
                    ep_end_time = time.time()
                    print(" ====== episode {} ======".format(episode))
                    print(" total time cost {:.3f}s".format(
                        ep_end_time - ep_begin_time))
                    print(" simulation time cost {:.3f}s".format(
                        sim_time_cost))
                    print(" learning time cost {:.3f}s".format(
                        learn_time_cost))
                    print(" central reward : {:.3f}".format(
                        central_cumulative_reward))
                    print(" loss :")
                    for id_, loss in local_loss.items():
                        print("      id {}, loss {:.3f}".format(
                            id_, loss / len(local_loss)
                        ))

                    print("=========================")
                    central_reward_record[episode] = central_cumulative_reward
                    for i in ids:
                        local_record[i]["loss"][episode] = local_loss[i]
                        local_record[i]["reward"][episode] = local_reward[i]
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
                eval_reward_recrod["all"][episode] = eval_rewards["all"]
                eval_reward_recrod["mean"][episode] = eval_rewards["mean"]
                central_record = {
                    "reward": central_reward_record,
                    "eval_reward": eval_reward_recrod,
                }
                self.__snapshot_training(
                    record_dir, central_record, local_record)
                if saved:
                    exec_params = {
                        "episode": episode,
                    }
                    param_file_name = "params_latest.pth"
                    param_file = params_dir + param_file_name
                    self.__snapshot_params(
                        env, policies, buffers, exec_params, train_config,
                        param_file
                    )
                    if eval_reward_mean > SAVED_THRESHOLD:
                        # 如果当前模型效果达到期望的阈值，就保存模型
                        param_file_name = "params_{}.pth".format(episode)
                        param_file = params_dir + param_file_name
                        self.__snapshot_params(
                            env, policies, buffers, exec_params, train_config,
                            param_file
                        )

        if saved:
            param_file_name = "params_latest.pth"
            param_file = params_dir + param_file_name
            exec_params = {
                "batch_size": batch_size,
                "episode": num_episodes,
            }
            self.__snapshot_params(
                env, policies, buffers,
                exec_params,
                train_config,
                param_file
            )
            central_record = {
                "reward": central_reward_record,
                "eval_reward": eval_reward_recrod,
            }
            self.__snapshot_training(
                record_dir, central_record, local_record)

            test_config = self.__load_test_config(
                param_file_name,
                params_dir)
            self.test(test_config)

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
                    policy_ = policies[id_]
                    if obs is None or policy_ is None:
                        print("intersection id is not exit {}".format(id_))
                    action = policy_.compute_single_action(obs, False)
                    actions[id_] = action
                states, rewards, done, _ = env.step(actions)
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
        policy_config = test_config["policy"]
        policy_config["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        weight = test_config["weight"]
        num_episodes = test_config["num_episodes"]
        record_dir = test_config["record_dir"]

        # 初始化环境
        env = envs.make(env_config)

        # 初始化策略
        policies = {}
        ids = env.intersection_ids()
        for id_ in ids:
            policy_weight = weight["policy"][id_]
            policies[id_] = policy.get_policy(
                policy_config["id"], policy_config)
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
                    policy_ = policies[id_]
                    if obs is None or policy_ is None:
                        print("intersection id is not exit {}".format(id_))
                    action = policy_.compute_single_action(obs, False)
                    actions[id_] = action
                states, rewards, done, _ = env.step(actions)
                for r in rewards.values():
                    cumulative_reward += r
                if done:
                    print("In test mode, episodes {}, reward is {:.3f}".format(
                        eps, cumulative_reward))
                    reward_history[eps] = cumulative_reward
                    break
        os.system("cp ../replay/replay_roadnet.json {}".format(
            record_dir
        ))
        os.system("cp ../replay/replay.txt {}".format(
            record_dir
        ))

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

    def __snapshot_params(self, env, polices, buffer, exec_params,
                          train_config, params_file):

        polices_weight = {}
        for id_, p in polices.items():
            polices_weight[id_] = p.get_weight()

        buffer_weight = {}
        for id_, b in buffer.items():
            buffer_weight[id_] = b.get_weight()

        weight = {
            "policy": polices_weight,
            "buffer": buffer_weight,
        }

        params = {
            "config": train_config,
            "weight": weight,
            "exec": exec_params,
        }

        torch.save(params, params_file)

    def __snapshot_training(self, record_dir,
                            central_record, local_record):
        saved_data = {
            "central": central_record,
            "local": local_record,
        }
        saved_data_file_name = "exp_result.txt"
        result_file = record_dir + "data/" + saved_data_file_name
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(str(saved_data))
        self.__store_record_img(record_dir + "figs/", result_file)

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
        record_dir = root_record + sub_dir
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)
        else:
            print("create record folder error , path exist : ", record_dir)

        param_path = record_dir + "params/"
        if not os.path.exists(param_path):
            os.mkdir(param_path)
        else:
            print("create record folder error , path exist : ", param_path)

        fig_path = record_dir + "figs/"
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        else:
            print("create record folder error , path exist : ", fig_path)

        data_path = record_dir + "data/"
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        else:
            print("create record folder error , path exist : ", data_path)

        return record_dir

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

        central_data = data["central"]
        local_data = data["local"]

        episodes = []
        rewards = []
        for ep, r in central_data["reward"].items():
            episodes.append(int(ep))
            rewards.append(float(r))
        util.savefig(
            episodes, rewards, x_lable="episodes",
            y_label="reward", title="rewards",
            img=record_dir+"central_reward.png")

        episodes = []
        mean_eval_reward = []
        eval_reward_mean = central_data["eval_reward"]["mean"]
        for ep, r in eval_reward_mean.items():
            episodes.append(int(ep))
            mean_eval_reward.append(r)
        util.savefig(
            episodes, mean_eval_reward, x_lable="episodes",
            y_label="reward", title="rewards",
            img=record_dir+"eval_reward_mean.png")

        for id_, val in local_data.items():

            episodes = []
            rewards = []
            for ep, r in val["reward"].items():
                episodes.append(int(ep))
                rewards.append(float(r))
            util.savefig(
                episodes, rewards, x_lable="episodes",
                y_label="reward", title="rewards",
                img=record_dir+"local_reward_{}.png".format(id_))

            episodes = []
            loss = []
            for ep, r in val["loss"].items():
                episodes.append(int(ep))
                loss.append(float(r))
            util.savefig(
                episodes, loss, x_lable="episodes",
                y_label="loss", title="loss",
                img=record_dir+"local_loss_{}.png".format(id_))

    def __load_test_config(self, model_file, record_dir):
        params = torch.load(record_dir + model_file)
        saved_config = params["config"]

        env_config = saved_config["env"]
        env_config["save_replay"] = True
        env_config["thread_num"] = 1

        policy_config = saved_config["policy"]
        test_config = {
            "env": env_config,
            "policy": policy_config,
            "weight": params["weight"],
            "record_dir": record_dir,
            "num_episodes": 1,
        }

        return test_config

    def __record_init_config(self, record_dir, config):
        config["policy"].pop("device")
        params_path = record_dir + "init_params.json"
        with open(params_path, "w") as f:
            json.dump(config, f)
