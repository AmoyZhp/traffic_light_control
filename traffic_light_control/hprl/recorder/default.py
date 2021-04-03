from typing import Dict, List
from hprl.recorder.recorder import Recorder
from hprl.util.typing import Reward, TrainingRecord

import logging
import json


def comput_summation_reward(rewards: List[Reward]) -> Reward:
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


def compute_avg_reward(rewards: List[Reward]) -> Reward:
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


def unwrap_records(records: List[TrainingRecord]) -> Dict:
    sum_rewards: List[Reward] = []
    avg_rewards: List[Reward] = []
    episodes = []
    for rd in records:
        sum_rewards.append(comput_summation_reward(rd.rewards))
        avg_rewards.append(compute_avg_reward(rd.rewards))
        episodes.append(rd.episode)
    rewards = {}
    sum_central_reward, sum_local_reward = unwrap_rewards(sum_rewards)
    rewards["sum"] = {
        "central": sum_central_reward,
        "local": sum_local_reward,
    }
    avg_central_reward, avg_local_reward = unwrap_rewards(avg_rewards)
    rewards["avg"] = {
        "central": avg_central_reward,
        "local": avg_local_reward,
    }
    records = {
        "rewards": rewards,
        "episodes": episodes,
    }
    return records


def unwrap_rewards(rewards: List[Reward]):
    central_reward = []
    local_reward = {}
    for reward in rewards:
        central_reward.append(reward.central)
        for id, v in reward.local.items():
            if not id in local_reward:
                local_reward[id] = []
            local_reward[id].append(v)
    return central_reward, local_reward


class DefaultRecorder(Recorder):
    def __init__(self) -> None:
        super().__init__()

    def log_record(self, record: TrainingRecord, logger: logging.Logger):
        avg_reward = compute_avg_reward(record.rewards)
        logger.info("avg reward : ")
        logger.info("\tcentral {:.3f}".format(avg_reward.central))
        for k, v in avg_reward.local.items():
            logger.info("\tagent {} reward : {:.3f} ".format(k, v))

        cumulative_reward = comput_summation_reward(record.rewards)
        logger.info("summation reward : ")
        logger.info("\tcentral {:.3f}".format(cumulative_reward.central))
        for k, v in cumulative_reward.local.items():
            logger.info("\tagent {} reward : {:.3f} ".format(k, v))

    def write_records(self, records: List[TrainingRecord], path: str):
        suffix = path.split(".")[-1]
        if suffix != "json":
            raise ValueError(
                "default recorder only accept json format records for write")
        records = unwrap_records(records)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f)

    def read_records(self, path: str) -> List[TrainingRecord]:
        suffix = path.split(".")[-1]
        if suffix != "json":
            raise ValueError(
                "default recorder only accept json format records for read")
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(path)
        return records