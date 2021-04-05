import logging
from hprl.util.typing import TrainingRecord
from typing import List
from hprl.recorder.recorder import log_record as _log_record
from hprl.recorder.recorder import write_records as _write_records
from hprl.recorder.recorder import read_records as _read_records
from hprl.recorder.recorder import Recorder


class DefaultRecorder(Recorder):
    def __init__(self) -> None:
        super().__init__()

    def log_record(self, record: TrainingRecord, logger: logging.Logger):
        _log_record(record=record, logger=logger)

    def write_records(self, records: List[TrainingRecord], path: str):
        _write_records(records=records, path=path)

    def read_records(self, path: str) -> List[TrainingRecord]:
        return _read_records(path)