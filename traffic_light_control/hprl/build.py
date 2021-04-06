import collections.abc
import logging
import os
from typing import Dict

import gym

from hprl.env.gym_wrapper import GymWrapper
from hprl.recorder import read_ckpt
from hprl.trainer.builder.iql import build_iql_trainer
from hprl.trainer.trainer import Trainer
from hprl.typing import PolicyTypes, ReplayBufferTypes

logger = logging.getLogger(__name__)


def load_trainer(
    ckpt_path: str,
    override_config: Dict = {},
    recording: bool = False,
):
    ckpt = read_ckpt(ckpt_path)
    config: Dict = ckpt["config"]
    trainer_conf = config["trainer"]
    policy_type = trainer_conf["type"]
    if recording:
        old_output_dir: str = trainer_conf["output_dir"]
        split_str = old_output_dir.split("_")
        cnt = 0
        if split_str[-2] == "cnt":
            cnt = int(split_str[-1])
            preffix = old_output_dir.split("cnt")[0]
            output_dir = f"{preffix}cnt_{cnt+1}"
        else:
            output_dir = f"{old_output_dir}_cnt_0"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        log_to_file(output_dir)
    else:
        output_dir = ""

    if "trainer" not in override_config:
        override_config["trainer"] = {}
    override_config["trainer"]["output_dir"] = output_dir
    if policy_type == PolicyTypes.IQL:
        dict_update(config, override_config)
        trainer = build_iql_trainer(config)
        trainer.set_weight(ckpt["weight"])
        trainer.set_records(ckpt["records"])
    else:
        raise ValueError(
            "policy {} loading function not implement".format(policy_type))
    return trainer


def log_to_file(path: str = ""):
    if not path:
        path = "hprl.log"
    elif os.path.isdir(path):
        path = f"{path}/hprl.log"
    filehander = logging.FileHandler(path, "a")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", )
    filehander.setFormatter(formatter)
    hprl_logger = logging.getLogger("hprl")
    hprl_logger.addHandler(filehander)


def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def build_gym_trainer(
    trainer_type: PolicyTypes,
    buffer_type: ReplayBufferTypes = None,
    batch_size: int = 0,
) -> Trainer:
    raise NotImplementedError
