from agent.static_agent import StaticAgent
from envs.tl_env import TlEnv
from agent.agent import Agent

CONFIG_PATH = "./config/config.json"


class Exectutor():
    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        config_path = CONFIG_PATH
        thread_num = 1
        episode = 1
        max_time = 100
        env = TlEnv(config_path, thread_num)
        agent = Agent("intersection_mid")

        for e in range(episode):
            state = env.reset()
            for t in range(max_time):
                action = agent.act(state)
                state, reward, done, info = env.step(action)
                print("state tensor : ", state.to_tensor())
                print("reward : {} ".format(reward))

    def eval(self):
        pass
