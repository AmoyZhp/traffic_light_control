from typing import Dict, List
import hprl
import logging
import json


def unwrap_records(records: List[hprl.TrainingRecord]):

    records_dict: Dict = hprl.recorder.unwrap_records(records)
    avg_travel_time = []
    for rd in records:
        avg_travel_time.append(rd.infos[-1]["avg_travel_time"])
    records_dict["avg_travel_time"] = avg_travel_time
    return records_dict


class TrafficRecorder(hprl.recorder.DefaultRecorder):
    def __init__(self) -> None:
        super().__init__()

    def log_record(self, record: hprl.TrainingRecord, logger: logging.Logger):
        super().log_record(record, logger)
        logger.info(
            "avg travel time : %f",
            record.infos[-1]["avg_travel_time"],
        )

    def write_records(self, records: List[hprl.TrainingRecord], path: str):
        suffix = path.split(".")[-1]
        if suffix != "json":
            raise ValueError(
                "default recorder only accept json format records for write")
        records = unwrap_records(records)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f)

    def read_records(self, path: str) -> List[hprl.TrainingRecord]:
        suffix = path.split(".")[-1]
        if suffix != "json":
            raise ValueError(
                "default recorder only accept json format records for read")
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(path)
        return records