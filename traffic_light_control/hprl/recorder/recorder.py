import abc
import json
import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from hprl.util.typing import Reward, TrainingRecord
import os


def plot_summation_rewards(
    records: List[TrainingRecord],
    save_fig=False,
    fig_path="",
):
    summation_rewards: List[Reward] = []
    episodes = []
    for record in records:
        summation_rewards.append(comput_summation_reward(record.rewards))
        episodes.append(record.episode)

    central_reward, local_reward = unwrap_rewards(summation_rewards)

    central_r_fig_path = ""
    if save_fig:
        central_r_fig_path = "train_sum_central_reward.jpg"
        if fig_path:
            if not os.path.isdir(fig_path):
                raise ValueError(
                    "fig path {} should be a directory".format(fig_path))
            central_r_fig_path = f"{fig_path}/{central_r_fig_path}"

    plot_fig(
        x=episodes,
        y=central_reward,
        x_lable="episodes",
        y_label="rewards",
        title="summation rewards",
        fig_path=central_r_fig_path,
    )

    for id, rewards in local_reward.items():
        local_r_fig_path = ""
        if save_fig:
            local_r_fig_path = "train_sum_{}_local_reward.jpg".format(id)
            if fig_path:
                local_r_fig_path = f"{fig_path}/{local_r_fig_path}"
        plot_fig(
            x=episodes,
            y=rewards,
            x_lable="episodes",
            y_label="rewards",
            title=f"agent {id} summation rewards",
            fig_path=local_r_fig_path,
        )


def plot_avg_rewards(
    records: List[TrainingRecord],
    save_fig=False,
    fig_path="",
):
    avg_reward: List[Reward] = []
    episodes = []
    for record in records:
        avg_reward.append(compute_avg_reward(record.rewards))
        episodes.append(record.episode)

    central_reward, local_reward = unwrap_rewards(avg_reward)

    central_r_fig_path = ""
    if save_fig:
        img_name = "train_avg_central_reward.jpg"
        if not fig_path:
            central_r_fig_path = img_name
        else:
            if not os.path.isdir(fig_path):
                raise ValueError(
                    "fig path {} should be a directory".format(fig_path))
            central_r_fig_path = f"{fig_path}/{img_name}"

    plot_fig(
        x=episodes,
        y=central_reward,
        x_lable="episodes",
        y_label="rewards",
        title="average rewards",
        fig_path=central_r_fig_path,
    )

    for id, rewards in local_reward.items():
        local_r_fig_path = ""
        if save_fig:

            img_name = "train_avg_{}_local_reward.jpg".format(id)
            if not fig_path:
                local_r_fig_path = img_name
            else:
                local_r_fig_path = f"{fig_path}/{img_name}"

        plot_fig(
            x=episodes,
            y=rewards,
            x_lable="episodes",
            y_label="rewards",
            title=f"agent {id} average rewards",
            fig_path=local_r_fig_path,
        )


def plot_fig(x: List, y: List, x_lable, y_label, title, fig_path=""):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='linear')
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if fig_path:
        fig.savefig(fig_path)
    plt.clf()
    plt.close(fig)


def write_ckpt(ckpt: Dict, path: str):
    torch.save(ckpt, path)


def read_ckpt(path: str):
    return torch.load(path)


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


def log_record(record: TrainingRecord, logger: logging.Logger):
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


def write_records(records: List[TrainingRecord], path: str):
    suffix = path.split(".")[-1]
    if suffix != "json":
        raise ValueError(
            "default recorder only accept json format records for write")
    records = unwrap_records(records)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)


def read_records(path: str) -> List[TrainingRecord]:
    suffix = path.split(".")[-1]
    if suffix != "json":
        raise ValueError(
            "default recorder only accept json format records for read")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(path)
    return records


class Recorder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log_record(
        self,
        record: TrainingRecord,
        logger: logging.Logger,
    ):
        ...

    @abc.abstractmethod
    def write_records(self, records: List[TrainingRecord], path: str):
        ...

    @abc.abstractmethod
    def read_records(self, path: str) -> List[TrainingRecord]:
        ...
