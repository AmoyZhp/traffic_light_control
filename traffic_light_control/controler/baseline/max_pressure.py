import envs
import hprl


def eval(env_config):
    env = envs.make_mp_cityflow(env_config)
    state = env.reset()
    sum_reward = 0.0
    avg_travel_time = 0
    while True:
        actions = {}
        for id, local_s in state.local.items():
            action = 0
            max_pressure = local_s[0]
            for i in range(len(local_s)):
                if local_s[i] > max_pressure:
                    action = i
                    max_pressure = local_s[i]
            actions[id] = action
        state, reward, done, info = env.step(hprl.Action(local=actions))
        sum_reward += reward.central
        if done.central:
            avg_travel_time = info["avg_travel_time"]
            break
    result = {
        "reward": sum_reward,
        "avg_travel_time": avg_travel_time,
    }
    return result
