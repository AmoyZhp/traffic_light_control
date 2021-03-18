import logging
from hprl.util.typing import TrainingRecord
from hprl.recorder.recorder import Recorder

from hprl.recorder.recorder import Recorder, cal_avg_reward, cal_cumulative_reward, draw_train_avg_rewards, draw_train_culumative_rewards
local_logger = logging.getLogger(__package__)


class Printer(Recorder):
    def add_record(self, record=None):
        ...

    def add_records(self, records=None):
        ...

    def print_record(
        self,
        record: TrainingRecord,
        logger: logging.Logger,
        fig=False,
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

    def write_records(self, dir="", filename=""):
        ...

    def read_records(self, dir="", filename=""):
        ...

    def write_ckpt(self, ckpt, dir="", filename=""):
        ...

    def read_ckpt(self, dir="", filename=""):
        ...

    def write_config(self, config={}, dir=""):
        ...

    def read_config(self, dir=""):
        ...

    def get_records(self):
        ...