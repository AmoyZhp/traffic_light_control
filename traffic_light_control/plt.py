from typing import List
import matplotlib.pyplot as plt
import json


def plot(filenames: List[str]):
    records = {}
    for fname in filenames:
        fname = f"records/{fname}/log/records.json"
        with open(fname, "r") as f:
            records[fname] = json.load(f)
    travel_time_fig, travel_time_ax = plt.subplots()
    central_sum_r_fig, central_sum_r_ax = plt.subplots()
    for name, rd in records.items():
        travel_time = rd["travel_time"]
        central_sum_r = rd["rewards"]["sum"]["central"]
        episodes = rd["episodes"]
        travel_time_ax.plot(episodes, travel_time, label=name)
        central_sum_r_ax.plot(episodes, central_sum_r, label=name)
    travel_time_ax.set_xlabel("episodes")
    travel_time_ax.set_ylabel("travel time")
    travel_time_ax.set_title("travel time")
    travel_time_ax.legend()
    travel_time_fig.savefig("{}/{}".format("./", "test_time.jpg"))

    central_sum_r_ax.set_xlabel("episodes")
    central_sum_r_ax.set_ylabel("rewards")
    central_sum_r_ax.set_title("rewards")
    central_sum_r_ax.legend()
    central_sum_r_fig.savefig("{}/{}".format("./", "test_r.jpg"))
    plt.clf()
    plt.close(travel_time_fig)
    plt.close(central_sum_r_fig)


if __name__ == "__main__":
    names = [
        "1x3_IQL_2021_3_24_14_25_1",
        "1x3_QMIX_2021_3_24_23_10_16",
        "1x3_VDN_2021_3_24_14_24_20",
    ]
    plot(names)
