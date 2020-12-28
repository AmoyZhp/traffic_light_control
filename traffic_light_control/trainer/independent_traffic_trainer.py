import os
import time
import json

from numpy.lib.utils import info
import buffer
import envs
import torch
import numpy as np
import policy
from policy import policy_wrapper
import util
from util.type import Transition


RECORDS_ROOT_DIR = "records/"
PARAMS_DIR = "params/"
FIGS_DIR = "figs/"
DATA_DIR = "data/"


# env setting
MAX_TIME = 3600
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
EVAL_NUM_EPISODE = 50
INTERATION_UPPER_BOUND = 1000000


class IndependentTrainer():

    def __init__(self) -> None:
        super().__init__()

    def run(self):

        args = util.parase_args()
        print(args)
        mode = args.mode
        model_file = args.model_file
        thread_num = args.thread_num
        episodes = args.episodes
        save = False if args.save == 0 else True
        resume = True if args.resume == 1 else False
        saved_peroid = args.data_saved_period
        saved_threshold = args.saved_threshold
        record_dir = args.record_dir
        env_id = args.environment

        if mode == "train":
            train_config = {}
            if resume:
                if (model_file == "" or record_dir == ""):
                    print("please input model file and record dir if resume")
                    return
                train_config = self.__resume_train_config(
                    record_dir, model_file)
            else:
                train_config = self.__get_default_config()

            train_config["env"]["thread_num"] = thread_num
            train_config["env"]["save_replay"] = False
            train_config["env"]["id"] = env_id
            train_config["policy"]["device"] = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            train_config["exec"]["num_episodes"] = episodes
            train_config["exec"]["saved"] = save
            train_config["exec"]["resume"] = resume
            train_config["exec"]["data_saved_period"] = saved_peroid
            train_config["exec"]["saved_threshold"] = saved_threshold

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
        exec_config = train_config["exec"]

        num_episodes = exec_config["num_episodes"]
        eval_num_episodes = exec_config["eval_num_episodes"]
        saved = exec_config["saved"]
        data_saved_period = exec_config["data_saved_period"]
        saved_threshold = exec_config["saved_threshold"]

        # 初始化环境
        env = envs.make(env_config)
        ids = env.intersection_ids()

        # 初始化策略
        p_wrapper = policy.PolicyWrapper(ids, {
            "buffer": buffer_config,
            "policy": policy_config,
        })

        # 创建和本次训练相应的保存目录
        record_dir = util.create_record_dir(RECORDS_ROOT_DIR)
        data_dir = record_dir + "data/"
        params_dir = record_dir + "params/"
        self.__record_init_config(data_dir, train_config)

        # 保存每轮 episode 完成后的 reward 奖励
        central_record = {
            "reward": {},
            "eval_reward": {
                "all": {},
                "mean": {},
                "average_travel_time": {},
            },
            "average_travel_time": {}
        }
        local_record = {}
        for id_ in ids:
            local_record[id_] = {
                "loss": {},
                "reward": {},
            }
        train_info = {}

        ep_begin = 1
        if exec_config["resume"]:
            ep_begin = exec_config["ep_begin"]

        train_begin_time = time.time()
        for episode in range(ep_begin, num_episodes + ep_begin):
            ep_begin_time = time.time()
            states = env.reset()
            central_cumulative_reward = 0.0
            sim_time_cost = 0.0
            learn_time_cost = 0.0
            local_loss = {}
            local_reward = {}
            for k in ids:
                local_loss[k] = 0.0
                local_reward[k] = 0.0
            for cnt in range(INTERATION_UPPER_BOUND):

                actions = p_wrapper.compute_action(states, True)

                sim_begin_time = time.time()
                next_states, rewards, done, info = env.step(actions)
                sim_time_cost += time.time() - sim_begin_time

                p_wrapper.record_transition(
                    states, actions, rewards, next_states, done, info)

                states = next_states

                learn_begin_time = time.time()
                local_losses = p_wrapper.update_policy()
                learn_time_cost += time.time() - learn_begin_time

                for id_, r in rewards.items():
                    central_cumulative_reward += r
                    local_reward[id_] += r
                    local_loss[id_] += local_losses[id_]

                if done:
                    ep_end_time = time.time()
                    print(" ====== episode {} ======".format(episode))
                    print(" total time cost {:.3f}s".format(
                        ep_end_time - ep_begin_time))
                    print(" simulation time cost {:.3f}s".format(
                        sim_time_cost))
                    print(" learning time cost {:.3f}s".format(
                        learn_time_cost))
                    print(" average travel time : {:.3f}".format(
                        info["average_travel_time"]))
                    print(" central reward : {:.3f}".format(
                        central_cumulative_reward))
                    print(" local info : ")
                    for id_ in ids:
                        print("  id {}, reward {:.3f}, avg loss {:.3f}".format(
                            id_, local_reward[id_], local_loss[id_] / cnt
                        ))
                    print(" training time pass {:.1f} min ".format(
                        (time.time() - train_begin_time) / 60
                    ))
                    print("=========================")
                    central_record["reward"][
                        episode] = central_cumulative_reward
                    central_record["average_travel_time"][
                        episode] = info["average_travel_time"]
                    for i in ids:
                        local_record[i]["loss"][episode] = local_loss[i]
                        local_record[i]["reward"][episode] = local_reward[i]
                    break
            if (episode % data_saved_period == 0):
                eval_rewards = self.eval_(
                    policy=policy_wrapper,
                    env=env,
                    num_episodes=eval_num_episodes,
                )
                eval_reward_mean = eval_rewards["mean"]
                travel_time = eval_rewards["average_travel_time"]

                print(" episode : {},".format(episode) +
                      "eval mean reward is {:.3f}, travel time {:.3f} ".format(
                    episode, eval_rewards["mean"], travel_time,
                ))
                eval_reward_recrod = central_record["eval_reward"]
                eval_reward_recrod["all"][episode] = eval_rewards["all"]
                eval_reward_recrod["mean"][episode] = eval_rewards["mean"]
                eval_reward_recrod["average_travel_time"][
                    episode] = travel_time
                train_info["training_time"] = time.time() - train_begin_time
                util.snapshot_exp_result(
                    record_dir, central_record, local_record, train_info)
                if saved:
                    exec_params = {
                        "episode": episode,
                    }
                    param_file_name = "params_latest.pth"
                    param_file = params_dir + param_file_name
                    util.snapshot_params(
                        config=train_config,
                        weight=p_wrapper.get_weight(),
                        exec_params=exec_params,
                        params_file=param_file
                    )
                    if eval_reward_mean < saved_threshold:
                        # 如果当前模型效果达到期望的阈值，就保存模型
                        param_file_name = "params_{}.pth".format(episode)
                        param_file = params_dir + param_file_name
                        util.snapshot_params(
                            config=train_config,
                            weight=p_wrapper.get_weight(),
                            exec_params=exec_params,
                            params_file=param_file
                        )

        if saved:
            param_file_name = "params_latest.pth"
            param_file = params_dir + param_file_name
            exec_params = {
                "episode": num_episodes,
            }
            util.snapshot_params(
                config=train_config,
                weight=p_wrapper.get_weight(),
                exec_params=exec_params,
                params_file=param_file
            )

            train_info["training_time"] = time.time() - train_begin_time
            util.snapshot_exp_result(
                record_dir, central_record, local_record, train_info)

            test_config = self.__load_test_config(
                "params/" + param_file_name,
                record_dir)
            self.test(test_config)

    def eval_(self, policy, env, num_episodes):

        reward_history = []
        travel_time_history = []
        for _ in range(num_episodes):
            states = env.reset()
            cumulative_reward = 0.0
            while True:
                actions = policy.compute_action(states)
                states, rewards, done, info = env.step(actions)
                for r in rewards.values():
                    cumulative_reward += r
                if done:
                    reward_history.append(cumulative_reward)
                    travel_time_history.append(info["average_travel_time"])
                    break
        reward_record = {}
        reward_record["all"] = reward_history
        reward_record["mean"] = sum(reward_history) / len(reward_history)
        reward_record["average_travel_time"] = sum(
            travel_time_history) / len(travel_time_history)
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
        ids = env.intersection_ids()

        # 初始化策略
        p_wrapper = policy_wrapper.PolicyWrapper(
            ids, {
                "policy": policy_config,
            },
            mode="test"
        )
        p_wrapper.set_weight(weight)
        reward_history = {}

        for eps in range(num_episodes):
            states = env.reset()
            cumulative_reward = 0.0
            while True:
                actions = p_wrapper.compute_action(states, False)
                states, rewards, done, info = env.step(actions)
                for r in rewards.values():
                    cumulative_reward += r
                if done:
                    print("In test mode, episodes {},".format(eps) +
                          "reward is {:.3f}, travel time {:.3f}".format(
                        cumulative_reward, info["average_travel_time"]))
                    reward_history[eps] = cumulative_reward
                    test_result = {
                        "travel_time": info["average_travel_time"],
                        "central_reward": cumulative_reward,
                    }
                    test_result_file = record_dir + "data/" + \
                        "test_result_{}.json".format(eps)
                    with open(test_result_file, "w", encoding="utf-8") as f:
                        json.dump(test_result, f)
                    break

        os.system("cp ../replay/replay_roadnet.json {}".format(
            record_dir + "data/"
        ))
        os.system("cp ../replay/replay.txt {}".format(
            record_dir + "data/"
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

    def __get_default_config(self):

        env_config = {
            "max_time": MAX_TIME,
            "interval": INTERVAL,
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
            "batch_size": BATCH_SIZE,
        }

        exec_config = {
            "eval_num_episodes": EVAL_NUM_EPISODE,
        }

        train_config = {
            "env": env_config,
            "buffer": buffer_config,
            "policy": policy_config,
            "exec": exec_config,
        }

        return train_config

    def __resume_train_config(self, record_dir, model_file):
        pass
