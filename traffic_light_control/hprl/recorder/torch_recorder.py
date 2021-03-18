import os
import logging
from enum import Enum
import json
import torch
from hprl.util.typing import Reward, TrainingRecord
from hprl.recorder.recorder import Recorder, cal_avg_reward, cal_cumulative_reward, draw_train_avg_rewards, draw_train_culumative_rewards
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
        self.records = []

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
        if fig:
            draw_train_culumative_rewards(self.records, self.record_dir)
            draw_train_avg_rewards(self.records, self.record_dir)

    def write_records(self, dir="", filename=""):
        if not dir:
            dir = self.record_dir
        if not filename:
            filename = "records.txt"
        record_file = f"{dir}/{filename}"
        with open(record_file, "w", encoding="utf-8") as f:
            f.write(str(self.get_records()))
        draw_train_culumative_rewards(self.records, dir)
        draw_train_avg_rewards(self.records, dir)

    def read_records(self, dir: str, filename: str):
        raise NotImplementedError

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

    def write_config(self, config, dir=""):
        if not dir:
            dir = self.config_dir
        filename = "init_config.json"
        config_path = f"{dir}/{filename}"

        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.name
                return json.JSONEncoder.default(self, obj)

        with open(config_path, "w") as f:
            json.dump(config, f, cls=EnumEncoder)

    def read_config(self, dir: str):
        raise NotImplementedError

    def get_records(self):
        return self.records
