import hprl.recorder.recorder as recorder
import logging
from hprl.util.typing import TrainingRecord
from typing import List


class DefaultRecorder(recorder.Recorder):
    def __init__(self) -> None:
        super().__init__()

    def log_record(self, record: TrainingRecord, logger: logging.Logger):
        recorder.log_record(record=record, logger=logger)

    def write_records(self, records: List[TrainingRecord], path: str):
        recorder.write_records(records=records, path=path)

    def read_records(self, path: str) -> List[TrainingRecord]:
        return recorder.read_records(path)