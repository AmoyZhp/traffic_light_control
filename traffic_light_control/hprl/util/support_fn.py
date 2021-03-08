import logging
import time
from typing import Dict, List

from hprl.util.typing import Reward, Terminal, TrainingRecord, Trajectory, Transition
from hprl.policy import Policy
from hprl.env import MultiAgentEnv
from hprl.replaybuffer import ReplayBuffer


def off_policy_train_fn(
    env: MultiAgentEnv,
    policy: Policy,
    replay_buffer: ReplayBuffer,
    config: Dict,
    logger: logging.Logger,
) -> List[Reward]:

    batch_size = config["batch_size"]

    state = env.reset()
    reward_record = []
    info_record = []
    sim_time_cost = 0.0
    learn_time_cost = 0.0
    while True:
        action = policy.compute_action(state)

        sim_begin = time.time()
        ns, r, done, info = env.step(action)
        sim_end = time.time()
        sim_time_cost += (sim_end - sim_begin)

        reward_record.append(r)
        info_record.append(info)

        trans = Transition(
            state=state,
            action=action,
            reward=r,
            next_state=ns,
            terminal=done,
        )
        replay_buffer.store(trans)
        state = ns

        learn_begin = time.time()
        batch_data = replay_buffer.sample(batch_size)
        policy.learn_on_batch(batch_data)
        learn_end = time.time()
        learn_time_cost += (learn_end - learn_begin)

        if done.central:
            logger.info("simulation time cost : {:.3f}s".format(sim_time_cost))
            logger.info("learning time cost : {:.3f}s".format(learn_time_cost))
            break
    return reward_record, info_record


def on_policy_train_fn(
    env: MultiAgentEnv,
    policy: Policy,
    replay_buffer: ReplayBuffer,
    config: Dict,
    logger: logging.Logger,
):

    batch_size = config["batch_size"]

    reward_records = []
    info_records = []
    sim_time_cost = 0.0
    learn_time_cost = 0.0
    batch_data: List[Trajectory] = []

    for _ in range(batch_size):

        reward_record = []
        info_record = []
        states = []
        actions = []
        rewards = []

        state = env.reset()
        while True:
            states.append(state)
            action = policy.compute_action(state)

            sim_begin = time.time()
            ns, r, done, info = env.step(action)
            sim_end = time.time()
            sim_time_cost += (sim_end - sim_begin)
            state = ns

            actions.append(action)
            rewards.append(r)

            reward_record.append(r)
            info_record.append(info)

            if done.central:
                trajecotry = Trajectory(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    terminal=done,
                )
                batch_data.append(trajecotry)
                reward_records.append(reward_record)
                info_records.append(info_record)
                break
    learn_begin = time.time()
    policy.learn_on_batch(batch_data)
    learn_end = time.time()
    learn_time_cost += (learn_end - learn_begin)

    logger.info("simulation time cost : {:.3f}s".format(sim_time_cost))
    logger.info("learning time cost : {:.3f}s".format(learn_time_cost))
    return reward_records[0], info_records[0]


def default_log_record_fn(record: TrainingRecord, logger: logging.Logger):

    avg_reward = cal_avg_reward(record.rewards)
    logger.info("avg reward : ")
    logger.info("    central {:.3f}".format(avg_reward.central))
    for k, v in avg_reward.local.items():
        logger.info("    agent {} reward is {:.3f} ".format(k, v))

    cumulative_reward = cal_cumulative_reward(record.rewards)
    logger.info("cumulative reward : ")
    logger.info("    central {:.3f}".format(cumulative_reward.central))
    for k, v in cumulative_reward.local.items():
        logger.info("    agent {} reward is {:.3f} ".format(k, v))


def cal_cumulative_reward(rewards: List[Reward]) -> Reward:
    length = len(rewards)
    ret = Reward(central=0.0, local={})
    if length == 0:
        return ret
    agents_id = rewards[0].local.keys()
    for k in agents_id:
        ret.local[k] = 0.0

    for r in rewards:
        ret.central += r.central
        for k, v in r.local.items():
            ret.local[k] += v

    return ret


def cal_avg_reward(rewards: List[Reward]) -> Reward:
    length = len(rewards)
    ret = Reward(central=0.0, local={})
    if length == 0:
        return ret
    agents_id = rewards[0].local.keys()
    for k in agents_id:
        ret.local[k] = 0.0

    for r in rewards:
        ret.central += r.central
        for k, v in r.local.items():
            ret.local[k] += v
    ret.central /= length

    for k in agents_id:
        ret.local[k] /= length
    return ret
