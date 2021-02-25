import logging

from hprl.trainer.core import Trainer
from hprl.trainer.common_trainer import CommonTrainer
from hprl.trainer.factory import create_trainer, load_trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch.setFormatter(formatter)

logger.addHandler(ch)

__all__ = [
    "Trainer",
    "CommonTrainer",
    "create_trainer",
    "load_trainer"
]
