from typing import Dict

from numpy.lib.type_check import imag
from basis.action import Action
from envs.tl_env import TlEnv
from agent.dqn import DQNAgent
from policy.dqn import DQN, DQNConfig
from policy.buffer.replay_memory import Transition
import os
import torch
import time
import sys
import getopt
import json
import util.plot as plot
import datetime

CONFIG_PATH = "./config/config.json"
STATIC_CONFIG = "./config/static_config.json"
MAX_TIME = 300
INTERVAL = 5
DATA_SAVE_PERIOD = 500
CAPACITY = 80000
LERNING_RATE = 1e-3
BATCH_SIZE = 512
DISCOUNT_FACTOR = 0.99
EPS_INIT = 1.0
EPS_MIN = 0.01
EPS_FRAME = 200000
UPDATE_COUNT = 500
STATE_SPACE = 6*4 + 2*2
ACTION_SPACE = 2
MODEL_ROOT_DIR = "./records/"


class Exectutor():
    def __init__(self):
        pass

    def run(self):
        argv = sys.argv[1:]
        mode = ""
        model_file = ""
        thread = 1
        episode = 1
        opts = []
        try:
            opts, args = getopt.getopt(
                argv, shortopts="hm:e:mf:th:",
                longopts=["mode=",
                          "episode=", "model_file=", "thread="])

        except getopt.GetoptError:
            print("input error, template : python test.py --mode=train"
                  + " --episode=1 --path=model.pth --thread=1")
            return

        for opt, arg in opts:
            if opt in ("-m", "--mode"):
                mode = arg
            elif opt in ("-mf", "--model_file"):
                model_file = arg
            elif opt in ("-e", "--episode"):
                episode = int(arg)
            elif opt in ("-th", "--thread"):
                thread = int(arg)
        print("mode : {}, model file :  {}, ep : {}, thread : {}".format(
            mode, model_file, episode, thread
        ))
        if mode == "test":
            self.test(model_file, num_episodes=episode)
        elif mode == "train":
            self.train(num_episodes=episode, thread_num=thread)
        elif mode == "static":
            self.static_run()
        else:
            print("please input mode")

    def train(self, num_episodes, thread_num=1):
        config_path = CONFIG_PATH
        intersection_id = "intersection_mid"
        env = TlEnv(config_path, max_time=MAX_TIME, thread_num=thread_num)
        data_save_period = DATA_SAVE_PERIOD

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() is False:
            print(" cuda is not available")
        config = DQNConfig(
            learning_rate=LERNING_RATE, batch_size=BATCH_SIZE,
            capacity=CAPACITY, discount_factor=DISCOUNT_FACTOR,
            eps_init=EPS_INIT, eps_min=EPS_MIN, eps_frame=EPS_FRAME,
            update_count=UPDATE_COUNT, state_space=STATE_SPACE,
            action_space=ACTION_SPACE, device=device)
        agent = DQNAgent(intersection_id, config)
        init_params = {
            "learning_rate": LERNING_RATE,
            "batch_size": BATCH_SIZE,
            "capacity": CAPACITY,
            "discount_facotr": DISCOUNT_FACTOR,
            "eps_init": EPS_INIT,
            "eps_min": EPS_MIN,
            "eps_frame": EPS_FRAME,
            "update_count": UPDATE_COUNT,
        }

        # 创建的目录
        date = datetime.datetime.now()
        sub_dir = "record_{}_{}_{}_{}_{}_{}".format(
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second
        )
        path = MODEL_ROOT_DIR + sub_dir
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print("create record folder error , path exist : ", path)
        path += "/"
        self.__write_json(
            init_params, "{}init_params.json".format(path))
        reward_history = {}
        loss_history = {}
        for episode in range(num_episodes):
            total_reward = 0.0
            total_loss = 0.0
            total_sim_time = 0.0
            total_net_time = 0.0
            state = env.reset()
            begin = time.time()
            for t in range(MAX_TIME + 1):
                if t % INTERVAL != 0:
                    action = Action(agent.intersection_id, True)
                    env.step(action)
                    continue
                step_time = time.time()
                action = agent.act(state)
                total_net_time += (time.time() - step_time)

                step_time = time.time()
                next_state, reward, done, _ = env.step(action)
                total_sim_time += (time.time() - step_time)

                t_state = torch.tensor(state.to_tensor()).float().unsqueeze(0)
                t_action = torch.tensor(action.to_tensor()).long().view(1, 1)
                t_reward = torch.tensor(reward).float().view(1, 1)
                if not done:
                    t_next_state = torch.tensor(
                        next_state.to_tensor()).float().unsqueeze(0)
                else:
                    t_next_state = None
                transition = Transition(
                    t_state, t_action, t_reward, t_next_state)
                agent.store(transition)
                state = next_state

                step_time = time.time()
                loss = agent.update_policy()
                total_net_time += (time.time() - step_time)

                total_reward += reward
                total_loss += loss
                if done:
                    end = time.time()
                    reward_history[episode + 1] = total_reward
                    loss_history[episode + 1] = total_loss / (t / INTERVAL)
                    print("episodes: {}, eps: {:.3f}, time: {:.3f}s,".format(
                        episode, agent.policy.eps, end - begin)
                        + " sim time : {:.3f}s, net time : {:.3f},".format(
                            total_sim_time, total_net_time)
                        + " total reward : {:.3f}, avg loss : {:.3f} ".format(
                            total_reward, total_loss / (t / INTERVAL)))
                    break
            if ((episode + 1) % data_save_period == 0
                    or episode == num_episodes - 1):
                full_path = path + "model.pth"
                agent.save_model(path=full_path)
                eval_reward_history = self.eval(
                    agent=agent, env=env,
                    num_episodes=10)
                save_data = {"reward": reward_history,
                             "loss": loss_history,
                             "eval_reward": eval_reward_history}
                self.__save_dict(save_data, "{}obs.txt".format(path))

        full_path = path + "model.pth"
        agent.save_model(full_path)
        self.test(full_path)
        self.__plot(path)

    def eval(self, agent: DQNAgent, env: TlEnv,
             num_episodes: int):
        reward_history = {}
        for episode in range(num_episodes):
            total_reward = 0.0
            state = env.reset()
            for t in range(MAX_TIME+1):
                if t % INTERVAL != 0:
                    action = Action(agent.intersection_id, True)
                    env.step(action)
                    continue
                action = agent.eval_act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    print("eval mode ! episodes {}, reward is {:.3f}".format(
                        episode, total_reward))
                    reward_history[episode] = total_reward
                    break
        return reward_history

    def test(self, model_path, num_episodes=1):
        config_path = "./config/test_config.json"
        intersection_id = "intersection_mid"
        env = TlEnv(config_path, max_time=MAX_TIME, thread_num=1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() is False:
            print(" cuda is not available")
        config = DQNConfig(
            learning_rate=LERNING_RATE, batch_size=BATCH_SIZE,
            capacity=CAPACITY, discount_factor=DISCOUNT_FACTOR,
            eps_init=EPS_INIT, eps_min=EPS_MIN, eps_frame=EPS_FRAME,
            update_count=UPDATE_COUNT, state_space=STATE_SPACE,
            action_space=ACTION_SPACE, device=device)

        agent = DQNAgent(intersection_id, config)
        # 读取网络参数
        agent.load_model(model_path)

        for i_ep in range(num_episodes):
            total_reward = 0.0
            state = env.reset()
            for t in range(MAX_TIME):
                if t % INTERVAL != 0:
                    action = Action(agent.intersection_id, True)
                    env.step(action)
                    continue
                action = agent.eval_act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done:

                    break
            print("episodes {}, reward is {:.3f}".format(
                i_ep, total_reward))

    def static_run(self):
        env = TlEnv(STATIC_CONFIG, MAX_TIME)
        total_reward = 0.0
        for _ in range(MAX_TIME):
            _, reward, done, _ = env.step(Action("", True))
            total_reward += reward
        print("static setting total reward is : {}".format(total_reward))

    def __save_dict(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))

    def __write_json(self, data, path):
        with open(path, "w") as f:
            json.dump(data, f)

    def __plot(self, record_dir):
        data = {}
        with open(record_dir + "obs.txt", "r", encoding="utf-8") as f:
            data = eval(f.read())
        episodes = []
        rewards = []
        for k, v in data["reward"].items():
            episodes.append(int(k))
            rewards.append(int(v))
        plot.plot(
            episodes, rewards, x_lable="episodes",
            y_label="reward", title="rewards",
            img=record_dir+"reward.png")
        episodes = []
        loss = []
        for k, v in data["loss"].items():
            episodes.append(int(k))
            loss.append(int(v))
        plot.plot(
            episodes, loss, x_lable="episodes",
            y_label="loss", title="loss",
            img=record_dir+"loss.png")
