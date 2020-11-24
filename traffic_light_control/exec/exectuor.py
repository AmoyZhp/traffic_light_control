from typing import Dict
from basis.action import Action
from envs.tl_env import TlEnv
from agent.dqn import DQNAgent
from policy.dqn import DQN, DQNConfig
from policy.buffer.replay_memory import Transition
import torch
import time
import sys
import getopt

CONFIG_PATH = "./config/config.json"


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
                argv, shortopts=["m:e:mf:th"],
                longopts=["mode=",
                          "episode=", "model_file=", "thread="])

        except getopt.GetoptError:
            print("python test.py --mode=train"
                  + " --episode=1 --path=model.pth --thread=1")

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
            self.test(model_file, episode=episode)
        elif mode == "train":
            self.train(episode=episode, thread_num=thread)
        else:
            print("please input mode")

    def train(self, episode, thread_num=1):
        config_path = CONFIG_PATH
        thread_num = thread_num
        num_episodes = episode
        max_time = 300
        intersection_id = "intersection_mid"
        env = TlEnv(config_path, max_time=max_time, thread_num=thread_num)

        data_save_period = 10
        capacity = 200000
        learning_rate = 5e-4
        batch_size = 512
        discount_factor = 0.99
        eps_init = 0.99
        eps_min = 0.01
        eps_frame = 100000
        update_count = 500
        state_space = 6*4 + 2*2
        action_space = 2
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() is False:
            print(" cuda is not available")
        config = DQNConfig(learning_rate=learning_rate, batch_size=batch_size,
                           capacity=capacity,
                           discount_factor=discount_factor, eps_init=eps_init,
                           eps_min=eps_min, eps_frame=eps_frame,
                           update_count=update_count, state_space=state_space,
                           action_space=action_space, device=device)
        agent = DQNAgent(intersection_id, config)
        reward_history = {}
        loss_history = {}
        for episode in range(num_episodes):
            total_reward = 0.0
            total_loss = 0.0
            state = env.reset()
            begin = time.time()
            for t in range(max_time):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
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
                loss = agent.update_policy()
                total_reward += reward
                total_loss += loss
                if done:
                    end = time.time()
                    reward_history[episode + 1] = total_reward
                    loss_history[episode + 1] = total_loss
                    print("episodes : {}, eps : {}, time cost : {}s".format(
                        episode, agent.policy.eps, end - begin)
                        + " total reward : {} , avg loss : {:.4f} ".format(
                            total_reward, total_loss / t))
                    break
            if (episode + 1) % data_save_period == 0:
                dir_path = "./params/"
                full_path = dir_path + "model_{}.pth".format(episode)
                agent.save_model(path=full_path)
                eval_reward_history = self.eval(
                    agent=agent, env=env,
                    num_episodes=50, max_time=max_time)
                save_data = {"reward": reward_history,
                             "loss": loss_history,
                             "eval_reward": eval_reward_history}
                self.__save_dict(save_data, "./params/obs.txt")

    def eval(self, agent: DQNAgent, env: TlEnv,
             num_episodes: int,  max_time: int):
        reward_history = {}
        for episode in range(num_episodes):
            total_reward = 0.0
            state = env.reset()
            for t in range(max_time):
                action = agent.eval_act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    print("eval mode ! episodes {}, reward is {}".format(
                        episode, total_reward))
                    reward_history[episode] = total_reward
                    break
        return reward_history

    def test(self, model_file, num_episodes=1):
        config_path = CONFIG_PATH
        thread_num = 1
        max_time = 300
        intersection_id = "intersection_mid"
        env = TlEnv(config_path, max_time=max_time, thread_num=thread_num)
        capacity = 100000
        learning_rate = 1e-2
        batch_size = 256
        discount_factor = 0.99
        eps_init = 0.99
        eps_min = 0.01
        eps_frame = 0.999
        update_count = 100
        state_space = 6*4 + 2*2
        action_space = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = DQNConfig(learning_rate=learning_rate, batch_size=batch_size,
                           capacity=capacity,
                           discount_factor=discount_factor, eps_init=eps_init,
                           eps_min=eps_min, eps_frame=eps_frame,
                           update_count=update_count, state_space=state_space,
                           action_space=action_space, device=device)
        agent = DQNAgent(intersection_id, config)
        dir_path = "./params/"
        full_path = dir_path + model_file
        agent.load_model(full_path)
        for i_ep in range(num_episodes):
            total_reward = 0.0
            state = env.reset()
            for t in range(max_time):
                action = agent.eval_act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done:

                    break
            print("episodes {}, reward is {}".format(i_ep,
                                                     total_reward))

    def __save_dict(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))
