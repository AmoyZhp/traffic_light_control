import logging

from hprl.trainer.core import Trainer
from hprl.trainer.common_trainer import CommonTrainer
from hprl.trainer.factory import create_trainer, load_trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)

logger.addHandler(handler)

__all__ = [
    "Trainer",
    "CommonTrainer",
    "create_trainer",
    "load_trainer"
]
