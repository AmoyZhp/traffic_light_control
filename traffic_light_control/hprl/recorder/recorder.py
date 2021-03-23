import logging
from hprl.util.typing import Reward, TrainingRecord
from typing import List
import torch
import matplotlib.pyplot as plt
import abc


class Recorder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_record(self, record: TrainingRecord):
        ...

    @abc.abstractmethod
    def add_records(self, records: List[TrainingRecord]):
        ...

    @abc.abstractmethod
    def print_record(
        self,
        record: TrainingRecord,
        logger: logging.Logger,
        fig: bool,
    ):
        ...

    @abc.abstractmethod
    def write_records(self, dir: str, filename: str):
        ...

    @abc.abstractmethod
    def read_records(self, dir: str, filename: str):
        ...

    @abc.abstractmethod
    def write_ckpt(self, ckpt, dir: str, filename: str):
        ...

    @abc.abstractmethod
    def read_ckpt(self, dir: str, filename: str):
        ...

    @abc.abstractmethod
    def write_config(self, config, dir: str):
        ...

    @abc.abstractmethod
    def read_config(self, dir: str):
        ...

    @abc.abstractmethod
    def get_records(self) -> List[TrainingRecord]:
        ...


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


def draw_avg_travel_time(records: List[TrainingRecord], log_dir: str):
    avg_travel_times = []
    episodes = []
    for r in records:
        avg_travel_times.append(r.infos[-1]["avg_travel_time"])
        episodes.append(r.episode)

    img_name = "train_avg_travel_time"
    save_fig(
        y=avg_travel_times,
        x=episodes,
        x_lable="average travel time",
        y_label="episodes",
        title=img_name,
        dir=log_dir,
        img_name=img_name,
    )


def draw_train_culumative_rewards(records: List[TrainingRecord], log_dir: str):
    culumative_rewards: List[Reward] = []
    episodes = []
    for reward in records:
        culumative_rewards.append(cal_cumulative_reward(reward.rewards))
        episodes.append(reward.episode)

    central_reward, local_reward = unwrap_rewards(culumative_rewards)

    img_name = "train_culumative_central_reward"
    save_rewards_fig(episodes, central_reward, img_name, log_dir)
    for id, rewards in local_reward.items():
        img_name = "tran_culumative_local_{}_reward".format(id)
        save_rewards_fig(episodes, rewards, img_name, log_dir)


def draw_train_avg_rewards(records: List[TrainingRecord], log_dir: str):
    avg_reward: List[Reward] = []
    episodes = []
    for reward in records:
        avg_reward.append(cal_cumulative_reward(reward.rewards))
        episodes.append(reward.episode)

    central_reward, local_reward = unwrap_rewards(avg_reward)

    img_name = "train_avg_central_reward"
    save_rewards_fig(episodes, central_reward, img_name, log_dir)
    for id, rewards in local_reward.items():
        img_name = "tran_avg_local_{}_reward".format(id)
        save_rewards_fig(episodes, rewards, img_name, log_dir)


def save_rewards_fig(episodes, rewards, image_name: str, log_dir: str):
    save_fig(
        x=episodes,
        y=rewards,
        x_lable="episodes",
        y_label="reward",
        title=image_name,
        dir=log_dir,
        img_name=image_name,
    )


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


def save_fig(x: List, y: List, x_lable, y_label, title, dir, img_name=""):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='linear')
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if img_name:
        fig.savefig("{}/{}".format(dir, img_name))
    plt.clf()
    plt.close(fig)
