from typing import List
import matplotlib.pyplot as plt
import json


def plot(filenames: List[str], baseline: str):
    records = {}
    for fname in filenames:
        fname = f"records/{fname}/log/records.json"
        with open(fname, "r") as f:
            records[fname] = json.load(f)
    travel_time_fig, travel_time_ax = plt.subplots()
    central_sum_r_fig, central_sum_r_ax = plt.subplots()
    episodes = []
    for name, rd in records.items():
        travel_time = rd["travel_time"]
        central_sum_r = rd["rewards"]["sum"]["central"]
        episodes = rd["episodes"]
        travel_time_ax.plot(episodes, travel_time, label=name)
        central_sum_r_ax.plot(episodes, central_sum_r, label=name)

    baseline_name = f"records/{baseline}/result.json"
    with open(baseline_name, "r") as f:
        result = json.load(f)
    avg_travel_time = result["avg_travel_time"]
    reward = result["reward"]
    times = []
    rewards = []
    for i in range(len(episodes)):
        times.append(avg_travel_time)
        rewards.append(reward)
    travel_time_ax.plot(episodes, times, label="mp")
    central_sum_r_ax.plot(episodes, rewards, label="mp")

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
        "archived/1x1_IQL_2021_3_25_15_43_50",
    ]
    baseline = "1x1_MP_2021_3_26_19_19_54"

    plot(names, baseline)
