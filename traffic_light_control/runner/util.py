import logging
import os
from typing import Dict, List

import hprl


def plot_avg_travel_time(
    records: List[hprl.TrainingRecord],
    save_fig=False,
    fig_path="",
):
    avg_travel_times = []
    episodes = []
    for record in records:
        avg_travel_times.append(record.infos[-1]["avg_travel_time"])
        episodes.append(record.episode)

    if save_fig:
        img_name = "train_avg_travel_time.jpg"
        if fig_path:
            if not os.path.isdir(fig_path):
                raise ValueError(
                    "fig path {} should be a directory".format(fig_path))
            fig_path = f"{fig_path}/{img_name}"
        else:
            fig_path = img_name
    else:
        fig_path = ""
    hprl.recorder.plot_fig(
        x=episodes,
        y=avg_travel_times,
        x_lable="episodes",
        y_label="average travel time",
        title="average travel time",
        fig_path=fig_path,
    )


def unwrap_records(records: List[hprl.TrainingRecord]):

    records_dict: Dict = hprl.recorder.unwrap_records(records)
    avg_travel_time = []
    for rd in records:
        avg_travel_time.append(rd.infos[-1]["avg_travel_time"])
    records_dict["avg_travel_time"] = avg_travel_time
    return records_dict


def log_record(record: hprl.TrainingRecord, logger: logging.Logger):
    hprl.recorder.log_record(record, logger)
    logger.info(
        "avg travel time : %f",
        record.infos[-1]["avg_travel_time"],
    )
