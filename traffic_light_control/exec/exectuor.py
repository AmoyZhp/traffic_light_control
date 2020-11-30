from os import EX_CANTCREAT
from pickle import DICT
from basis.action import Action
from envs.tl_env import TlEnv
from agent.dqn import DQNAgent
from policy.dqn import DQN, DQNConfig
from policy.buffer.replay_memory import Transition
from collections import namedtuple
import os
import torch
import time
import json
import util.plot as plot
import datetime
import argparse

CITYFLOW_CONFIG_PATH = "config/config.json"
STATIC_CONFIG = "config/static_config.json"
MODEL_ROOT_DIR = "records/"

# env setting
MAX_TIME = 300
INTERVAL = 5

# agent setting
CAPACITY = 100000
LERNING_RATE = 1e-4
BATCH_SIZE = 256
DISCOUNT_FACTOR = 0.99
EPS_INIT = 1.0
EPS_MIN = 0.01
EPS_FRAME = 200000
UPDATE_PERIOD = 1000
STATE_SPACE = 6*4 + 2*2
ACTION_SPACE = 2

# exec setting
DATA_SAVE_PERIOD = 500
SAVED_THRESHOLD = -30.0


class Exectutor():
    def __init__(self):
        pass

    def run(self):
        args = self.__parase_args()
        mode = args.mode
        model_file = args.model_file
        thread_num = args.thread_num
        episodes = args.episodes
        save = False if args.save == 0 else True

        print("mode : {}, model file :  {}, ep : {}, thread : {} ".format(
            mode, model_file, episodes, thread_num
        ) + "save : {}".format(save))
        if mode == "test":
            self.test(model_file, num_episodes=episodes)
        elif mode == "train":
            self.train(num_episodes=episodes, thread_num=thread_num, save=save)
        elif mode == "static":
            self.static_run()
        else:
            print("mode is invalid : {}".format(mode))

    def train(self, num_episodes, thread_num=1, save=True):
        env = self.__init_env(CITYFLOW_CONFIG_PATH,
                              MAX_TIME, INTERVAL, thread_num)
        agent = self.__init_agent()

        record_dir = ""
        if save:
            record_dir = self.__init_record()

        reward_history = {}
        eval_reward_history = {}
        loss_history = {}
        for episode in range(num_episodes):
            state = env.reset()

            total_reward = 0.0
            total_loss = 0.0
            total_sim_time = 0.0
            total_net_time = 0.0
            begin = time.time()
            while True:
                # 手动控制决策的时间间隔
                if env.time % env.interval != 0:
                    action = Action(agent.intersection_id, True)
                    step_time = time.time()
                    env.step(action)
                    total_sim_time += (time.time() - step_time)
                    continue

                # agent 选择行动
                step_time = time.time()
                action = agent.act(state)
                total_net_time += (time.time() - step_time)

                # 仿真器执行
                step_time = time.time()
                next_state, reward, done, _ = env.step(action)
                total_sim_time += (time.time() - step_time)

                # 将数据转化为 pytorch 的 tensor
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

                # 将数据存进 replay buffer
                agent.store(transition)

                state = next_state

                # 更新参数
                step_time = time.time()
                loss = agent.update_policy()
                total_net_time += (time.time() - step_time)

                total_reward += reward
                total_loss += loss
                if done:
                    end = time.time()
                    reward_history[episode + 1] = total_reward
                    loss_history[episode + 1] = total_loss / \
                        (env.max_time / env.interval)
                    print("episodes: {}, eps: {:.3f}, time: {:.3f}s,".format(
                        episode, agent.policy.eps, end - begin)
                        + " sim time : {:.3f}s, net time : {:.3f},".format(
                            total_sim_time, total_net_time)
                        + " total reward : {:.3f}, avg loss : {:.3f} ".format(
                            total_reward,
                            total_loss / (env.max_time / env.interval)))
                    break
            if (save
                and ((episode + 1) % DATA_SAVE_PERIOD == 0
                     or episode + 1 == num_episodes)):
                # 评估当前的效果
                eval_rewards, mean_reward = self.eval(
                    agent=agent, env=env,
                    num_episodes=50)
                eval_reward_history[episode] = eval_rewards

                # 记录参数
                saved_data = {"reward": reward_history,
                              "loss": loss_history,
                              "eval_reward": eval_reward_history}
                result_file = record_dir + "exp_result.txt"
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(str(saved_data))
                self.__plot(record_dir, result_file)

                model_path = record_dir + "model.pth"
                replay_path = ""
                if mean_reward > SAVED_THRESHOLD:
                    # 如果当前的评估达到了预期，则将它保存到 well_model 文件夹下
                    well_record_path = record_dir + \
                        "episode_{}/".format(episode)
                    os.mkdir(well_record_path)
                    model_path = well_record_path + "model.pth"
                    replay_path = "../" + well_record_path + "replay.txt"

                agent.save_model(path=model_path)
                self.test(model_path, replay_path)

                print("episode {}, mean eval reward is {:.3f}".format(
                    episode, mean_reward))

    def eval(self, agent: DQNAgent, env: TlEnv,
             num_episodes: int):
        reward_history = []
        for _ in range(num_episodes):
            total_reward = 0.0
            state = env.reset()
            while True:
                if env.time % env.interval != 0:
                    action = Action(agent.intersection_id, True)
                    env.step(action)
                    continue
                action = agent.eval_act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state

                if done:
                    reward_history.append(total_reward)
                    break
        mean_reward = sum(reward_history) / num_episodes
        return reward_history, mean_reward

    def test(self, model_path, replay_path="", num_episodes=1):

        config_path = "./config/test_config.json"
        env = self.__init_env(config_path,
                              MAX_TIME, INTERVAL)
        if replay_path != "":
            print(replay_path)
            env.set_replay_file(replay_path)
        agent = self.__init_agent()

        # 读取网络参数
        agent.load_model(model_path, True)

        for episode in range(num_episodes):
            total_reward = 0.0
            state = env.reset()
            while True:
                if env.time % env.interval != 0:
                    action = Action(agent.intersection_id, True)
                    env.step(action)
                    continue
                action = agent.eval_act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state

                if done:
                    break
            print("In test mode, episodes {}, reward is {:.3f}".format(
                episode, total_reward))

    def static_run(self):
        env = TlEnv(STATIC_CONFIG, MAX_TIME)
        total_reward = 0.0
        while True:
            _, reward, done, _ = env.step(Action("", True))
            if env.time % env.interval == 0:
                total_reward += reward
            if done:
                break
        print("static setting total reward is : {}".format(total_reward))

    def __parase_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m", "--mode", type=str, default="", required=True,
            help="mode of exec, include [train, test, static]")
        parser.add_argument(
            "-e", "--episodes", type=int, default=1,
            help="episode of exec time"
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
            help="save paramse or not"
        )
        return parser.parse_args()

    def __init_env(self, env_config_path: str,
                   max_time: int, interval: int,
                   thread_num: int = 1) -> TlEnv:
        env = TlEnv(
            config_path=env_config_path, max_time=max_time, interval=interval,
            thread_num=thread_num)
        return env

    def __init_agent(self) -> DQNAgent:

        intersection_id = "intersection_mid"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() is False:
            print(" cuda is not available")
        config = DQNConfig(
            learning_rate=LERNING_RATE, batch_size=BATCH_SIZE,
            capacity=CAPACITY, discount_factor=DISCOUNT_FACTOR,
            eps_init=EPS_INIT, eps_min=EPS_MIN, eps_frame=EPS_FRAME,
            update_period=UPDATE_PERIOD, state_space=STATE_SPACE,
            action_space=ACTION_SPACE, device=device)
        agent = DQNAgent(intersection_id, config)
        return agent

    def __init_record(self) -> str:
        init_params = {
            "learning_rate": LERNING_RATE,
            "batch_size": BATCH_SIZE,
            "capacity": CAPACITY,
            "discount_facotr": DISCOUNT_FACTOR,
            "eps_init": EPS_INIT,
            "eps_min": EPS_MIN,
            "eps_frame": EPS_FRAME,
            "update_period": UPDATE_PERIOD,
        }
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
        path = MODEL_ROOT_DIR + sub_dir
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print("create record folder error , path exist : ", path)
        params_path = path + "init_params.json"
        with open(params_path, "w") as f:
            json.dump(init_params, f)

        return path

    def __save_params(self, saved_path: str,
                      agent: DQNAgent, env: TlEnv):

        agent_config = {
            "learning_rate": agent.policy.learning_rate,
            "batch_size": agent.policy.batch_size,
            "capacity": agent.policy.memory.capacity,
            "discount_factor": agent.policy.discount_factor,
            "eps_init": agent.policy.eps_init,
            "eps_min": agent.policy.eps_min,
            "eps_frame": agent.policy.eps_frame,
            "step": agent.policy.step
        }

        agent_params = {
            "net": agent.policy.acting_net.state_dict(),
            "optimizer": agent.policy.optimizer.state_dict(),
            "memory": agent.policy.memory.memory,
            "config": agent_config
        }

        env_params = {
            "max_time": env.max_time,
            "interval": env.interval,
        }

        params = {
            "agent": agent_params,
            "env": env_params,
        }
        torch.save(params, saved_path)

    def __load_params(self, saved_path: str):
        data = torch.load(saved_path)
        agent_params = data["agent"]
        env_params = data["env"]
        env = TlEnv(CITYFLOW_CONFIG_PATH,
                    env_params["max_time"], env_params["interval"])
        intersection_id = "intersection_mid"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() is False:
            print(" cuda is not available")
        config_json = agent_params["config"]
        config = DQNConfig(
            learning_rate=config_json["learning_rate"],
            batch_size=config_json["batch_size"],
            capacity=config_json["capacity"],
            discount_factor=config_json["discount_factor"],
            eps_init=config_json["eps_init"],
            eps_min=config_json["eps_min"],
            eps_frame=config_json["eps_frame"],
            update_period=UPDATE_PERIOD,
            state_space=STATE_SPACE,
            action_space=ACTION_SPACE,
            device=device)
        agent = DQNAgent(intersection_id, config)
        config = DQNConfig()

    def __plot(self, record_dir, data_file):
        data = {}
        with open(data_file, "r", encoding="utf-8") as f:
            data = eval(f.read())
        episodes = []
        rewards = []
        for k, v in data["reward"].items():
            episodes.append(int(k))
            rewards.append(float(v))
        plot.plot(
            episodes, rewards, x_lable="episodes",
            y_label="reward", title="rewards",
            img=record_dir+"reward.png")

        episodes = []
        mean_eval_reward = []
        for k, v in data["eval_reward"].items():
            episodes.append(int(k))
            eval_rewards = v
            num = 0.0
            for reward in eval_rewards:
                num += float(reward)
            mean_eval_reward.append(int(num / len(eval_rewards)))
        plot.plot(
            episodes, mean_eval_reward, x_lable="episodes",
            y_label="reward", title="rewards",
            img=record_dir+"eval_reward.png")
        episodes = []
        loss = []
        for k, v in data["loss"].items():
            episodes.append(int(k))
            loss.append(float(v))
        plot.plot(
            episodes, loss, x_lable="episodes",
            y_label="loss", title="loss",
            img=record_dir+"loss.png")
