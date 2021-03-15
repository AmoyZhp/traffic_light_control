import logging

from hprl.trainer.trainer import Trainer
from hprl.trainer.common_trainer import CommonTrainer
from hprl.trainer.independent_learner_trainer import IndependentLearnerTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)

logger.addHandler(handler)

__all__ = ["Trainer", "CommonTrainer", "IndependentLearnerTrainer"]
