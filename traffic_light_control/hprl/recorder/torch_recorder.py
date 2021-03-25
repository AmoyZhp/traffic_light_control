import os
import logging
from enum import Enum
import json
from os.path import join
import torch
from hprl.util.typing import Reward, TrainingRecord
from hprl.recorder.recorder import Recorder, cal_avg_reward, cal_cumulative_reward, draw_avg_travel_time, draw_train_avg_rewards, draw_train_culumative_rewards, unwrap_rewards
from typing import List

local_logger = logging.getLogger(__package__)

CHEKCPOINT_DIR_SUFFIX = "checkpoints"
CONFIG_DIR_SUFFIX = "configs"
LOG_DIR_SUFFIX = "log"


class TorchRecorder(Recorder):
    def __init__(self, base_dir: str) -> None:
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        self.base_dir = base_dir

        checkpoint_path = f"{base_dir}/{CHEKCPOINT_DIR_SUFFIX}"
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        record_path = f"{base_dir}/{LOG_DIR_SUFFIX}"
        if not os.path.exists(record_path):
            os.mkdir(record_path)

        config_path = f"{base_dir}/{CONFIG_DIR_SUFFIX}"
        if not os.path.exists(config_path):
            os.mkdir(config_path)

        self.ckpt_dir = checkpoint_path
        self.record_dir = record_path
        self.config_dir = config_path
        self.records: List[TrainingRecord] = []

    def add_record(self, record: TrainingRecord):
        self.records.append(record)

    def add_records(self, records: List):
        self.records.extend(records)

    def print_record(
        self,
        record: TrainingRecord,
        logger: logging.Logger,
        fig: bool = True,
    ):
        if logger is None:
            logger = local_logger
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
        # this two logger should be detele later
        # only for quick test traffic env porpose
        logger.info("avg travel time : %f",
                    record.infos[-1]["avg_travel_time"])
        if fig:
            draw_train_culumative_rewards(self.records, self.record_dir)
            draw_train_avg_rewards(self.records, self.record_dir)
            draw_avg_travel_time(self.records, self.record_dir)

    def write_records(self, dir="", filename=""):
        if not dir:
            dir = self.record_dir
        if not filename:
            filename = "records.json"
        record_file = f"{dir}/{filename}"
        records = self._unwrap_record(self.get_records())
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(records, f)
        draw_train_culumative_rewards(self.records, dir)
        draw_train_avg_rewards(self.records, dir)
        draw_avg_travel_time(self.records, dir)

    def _unwrap_record(self, records: List[TrainingRecord]):
        sum_rewards: List[Reward] = []
        avg_rewards: List[Reward] = []
        avg_travel_time = []
        episodes = []
        for rd in records:
            sum_rewards.append(cal_cumulative_reward(rd.rewards))
            avg_rewards.append(cal_avg_reward(rd.rewards))
            avg_travel_time.append(rd.infos[-1]["avg_travel_time"])
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
            "travel_time": avg_travel_time,
            "episodes": episodes,
        }
        return records

    def read_records(self, dir: str, filename: str):
        record_file = f"{dir}/{filename}"
        with open(record_file, "w", encoding="utf-8") as f:
            records = json.load(record_file)
        return records

    def write_ckpt(self, ckpt, dir="", filename=""):
        if not dir:
            dir = self.ckpt_dir
        if not filename:
            filename = "ckpt.pth"
        file = f"{dir}/{filename}"
        torch.save(ckpt, file)

    def read_ckpt(self, dir="", filename=""):
        if not dir:
            dir = self.ckpt_dir
        if not filename:
            filename = "ckpt.pth"
        file = f"{dir}/{filename}"
        torch.load(file)

    def write_config(self, config, dir="", filename=""):
        if not dir:
            dir = self.config_dir
        if not filename:
            filename = "init_config.json"
        config_path = f"{dir}/{filename}"

        suffix = filename.split(".")[-1]
        if suffix == "json":

            class EnumEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Enum):
                        return obj.name
                    return json.JSONEncoder.default(self, obj)

            with open(config_path, "w") as f:
                json.dump(config, f, cls=EnumEncoder)
        else:
            with open(config_path, "w") as f:
                f.write(config)

    def read_config(self, dir: str = "", filename=""):
        raise NotImplementedError

    def get_records(self):
        return self.records
