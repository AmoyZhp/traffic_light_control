import abc
import logging
from typing import Callable, Dict, List

from hprl.util.typing import TrainingRecord

LogRecordFn = Callable[[TrainingRecord, logging.Logger], None]

logger = logging.getLogger(__name__)


class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(
        self,
        episodes: int,
        ckpt_frequency: int,
        log_record_fn: LogRecordFn,
    ) -> List[TrainingRecord]:
        ...

    @abc.abstractmethod
    def eval(
        self,
        episodes: int,
        log_record_fn: LogRecordFn,
    ) -> List[TrainingRecord]:
        ...

    @abc.abstractmethod
    def save_checkpoint(self, path: str):
        ...

    @abc.abstractmethod
    def load_checkpoint(self, path: str):
        ...

    @abc.abstractmethod
    def get_checkpoint(self) -> Dict:
        ...

    @abc.abstractmethod
    def set_weight(self, weight: Dict):
        ...

    @abc.abstractmethod
    def get_weight(self):
        ...

    @abc.abstractmethod
    def get_config(self):
        ...

    @abc.abstractmethod
    def set_records(self, records: List[TrainingRecord]):
        ...

    @abc.abstractmethod
    def get_records(self):
        ...

    @property
    @abc.abstractmethod
    def output_dir(self):
        ...
