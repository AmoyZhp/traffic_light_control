from basis.action import Action
from envs.tl_env import TlEnv
from agent.dqn import DQNAgent
from policy.dqn import DQN, DQNConfig
from policy.buffer.replay_memory import Transition
import torch
import time

CONFIG_PATH = "./config/config.json"


class Exectutor():
    def __init__(self):
        pass

    def train(self):
        config_path = CONFIG_PATH
        thread_num = 1
        num_episodes = 2000
        max_time = 300
        intersection_id = "intersection_mid"
        env = TlEnv(config_path, max_time=max_time, thread_num=thread_num)

        capacity = 50000
        learning_rate = 1e-4
        batch_size = 512
        discount_factor = 0.99
        eps_max = 0.99
        eps_min = 0.01
        eps_decay = 0.999
        update_count = 100
        state_space = 6*4 + 2*2
        action_space = 2
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() is False:
            print( " cuda is not available")
        config = DQNConfig(learning_rate=learning_rate, batch_size=batch_size,
                           capacity=capacity,
                           discount_factor=discount_factor, eps_max=eps_max,
                           eps_min=eps_min, eps_decay=eps_decay,
                           update_count=update_count, state_space=state_space,
                           action_space=action_space, device=device)
        agent = DQNAgent(intersection_id, config)
        for i_ep in range(num_episodes):
            total_reward = 0.0
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
                total_reward += reward
                transition = Transition(
                    t_state, t_action, t_reward, t_next_state)
                agent.store(transition)
                state = next_state
                agent.update_policy()
            end = time.time()
            print("episodes {}, reward is {}, time cost {}s".format(i_ep,
                                                     total_reward, end - begin))
            if (i_ep + 1) % 500 == 0:
                path = "model_{}.pth".format(i_ep)
                agent.save_model(path=path)

    def eval(self, agent: DQNAgent, env: TlEnv):
        num_episodes = 10
        max_time = 500
        for i_ep in range(num_episodes):
            total_reward = 0.0
            state = env.reset()
            for t in range(max_time):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                t_state = torch.tensor(state.to_tensor()).float().unsqueeze(0)
                t_action = torch.tensor(action.to_tensor()).long().view(1, 1)
                t_reward = torch.tensor(reward).float().view(1, 1)
                if not done:
                    t_next_state = torch.tensor(
                        next_state.to_tensor()).float().unsqueeze(0)
                else:
                    t_next_state = None
                total_reward += reward
                transition = Transition(
                    t_state, t_action, t_reward, t_next_state)
                agent.store(transition)
                state = next_state
                agent.update_policy()
                if done:
                    break
            print("episodes {}, reward is {}".format(i_ep,
                                                     total_reward))

    def test(self):
        config_path = CONFIG_PATH
        thread_num = 1
        num_episodes = 50
        max_time = 300
        intersection_id = "intersection_mid"
        env = TlEnv(config_path, max_time=max_time, thread_num=thread_num)
        capacity = 100000
        learning_rate = 1e-2
        batch_size = 256
        discount_factor = 0.99
        eps_max = 0.99
        eps_min = 0.01
        eps_decay = 0.999
        update_count = 100
        state_space = 6*4 + 2*2
        action_space = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = DQNConfig(learning_rate=learning_rate, batch_size=batch_size,
                           capacity=capacity,
                           discount_factor=discount_factor, eps_max=eps_max,
                           eps_min=eps_min, eps_decay=eps_decay,
                           update_count=update_count, state_space=state_space,
                           action_space=action_space, device=device)
        agent = DQNAgent(intersection_id, config)
        agent.load_model("./model_4.pth")
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
