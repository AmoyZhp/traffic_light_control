from os import name
from typing import Dict, List
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


def smooth(scalar, weight=0.85):
    smoothed = []
    last = scalar[0]
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_avg_time(
    algs: Dict[str, str],
    baselines: Dict[str, str],
    path,
    img_name,
):
    travel_time_fig, travel_time_ax = plt.subplots()
    max_episodes = 0
    for name, file_name in algs.items():
        fname = f"records/{file_name}/log/records.json"
        with open(fname, "r") as f:
            record = json.load(f)
            travel_time = record["travel_time"]
            travel_time = smooth(travel_time)
            episodes = record["episodes"]
            max_episodes = max(max_episodes, len(episodes))
            travel_time_ax.plot(episodes, travel_time, label=name)

    for name, file_name in baselines.items():
        baseline_name = f"records/{file_name}/result.json"
        with open(baseline_name, "r") as f:
            result = json.load(f)
        avg_travel_time = result["avg_travel_time"]
        times = []
        episodes = []
        for i in range(max_episodes):
            times.append(avg_travel_time)
            episodes.append(i)
        travel_time_ax.plot(episodes, times, label=name)

    travel_time_ax.set_xlabel("episodes")
    travel_time_ax.set_ylabel("travel time")
    travel_time_ax.legend(loc="best")
    travel_time_fig.savefig("{}/{}".format(path, f"{img_name}.jpg"))

    plt.clf()
    plt.close(travel_time_fig)


def plot_reward_and_avg_time_replation(algs: Dict[str, str], path, img_name):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for name, file_name in algs.items():
        fname = f"records/{file_name}/log/records.json"
        with open(fname, "r") as f:
            record = json.load(f)
            travel_time = record["travel_time"]
            rewards = record["rewards"]["sum"]["central"]
            episodes = record["episodes"]
            travel_time = smooth(travel_time)
            rewards = smooth(rewards)
            rewards = list(map(lambda x: -x, rewards))
            ax.plot(episodes, travel_time, label="travel time")
            ax2.plot(rewards, 'r', label="speed rate")

    ax.set_xlabel("episodes")
    ax.set_ylabel("average travel time")
    ax.legend(loc="upper left")
    ax2.set_xlabel("episodes")
    ax2.set_ylabel("average speed rate")
    ax2.legend(loc="upper right")
    fig.savefig("{}/{}".format(path, f"{img_name}.jpg"))

    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    # img_name = "1x3_VDN_compaerd"
    IQL_1x3 = {"IQL": "1x3_IQL_2021_3_29_12_12_25"}
    VDN_1x3_speed = {"VDN Speed Rate": "1x3_VDN_2021_3_29_12_13_50"}
    VDN_1x3_pressure = {"VDN Pressure": "1x3_VDN_2021_3_29_12_14_36"}
    IQL_1x5 = {"IQL": "1x5_IQL_2021_3_29_12_12_50_cnt_2"}
    VDN_1x5_speed = {"VDN Speed Rate": "1x5_VDN_2021_3_29_12_13_8_cnt_4"}
    VDN_1x5_pressure = {"VDN Pressure": "1x5_VDN_2021_3_29_13_16_51_cnt_4"}
    VDN_1x4_speed = {"VDN Speed Rate": "1x4_VDN_2021_3_29_20_15_24_cnt_0"}
    VDN_1x4_pressure = {"VDN Pressure": "1x4_VDN_2021_3_29_20_17_34_cnt_0"}
    IQL_1x4 = {"IQL": "1x4_IQL_2021_3_29_12_12_38_cnt_0"}
    VDN_2x2_speed = {"VDN Speed Rate": "2x2_VDN_2021_3_30_0_36_20"}
    VDN_2x2_pressure = {"VDN Pressure": "2x2_VDN_2021_3_30_0_38_18"}
    IQL_2x2 = {"IQL": "2x2_IQL_2021_3_30_0_37_2"}

    MP_1x3 = {"Max Pressure": "1x3_MP_2021_3_29_21_58_19"}
    MP_1x4 = {"Max Pressure": "1x4_MP_2021_3_29_21_58_32"}
    MP_1x5 = {"Max Pressure": "1x5_MP_2021_3_29_21_58_38"}
    MP_2x2 = {"Max Pressure": "2x2_MP_2021_3_30_16_41_53"}

    names = {}
    baseline = {}

    # names.update(IQL_1x3)
    # names.update(VDN_1x3_speed)
    # names.update(VDN_1x3_pressure)
    # baseline.update(MP_1x3)
    # img_name = "1x3_avg_time"
    # img_name = "1x3_speed_pressure"
    # img_name = "1x3_speed_time_relation"

    # names.update(IQL_1x4)
    # names.update(VDN_1x4_speed)
    # names.update(VDN_1x4_pressure)
    # baseline.update(MP_1x4)
    # img_name = "1x4_avg_time"
    # img_name = "1x4_speed_pressure"
    # img_name = "1x4_speed_time_relation"

    # names.update(IQL_1x5)
    # names.update(VDN_1x5_speed)
    # names.update(VDN_1x5_pressure)
    # baseline.update(MP_1x5)
    # img_name = "1x5_avg_time"
    # img_name = "1x5_speed_pressure"
    # img_name = "1x5_speed_time_relation"

    # names.update(IQL_2x2)
    names.update(VDN_2x2_speed)
    # names.update(VDN_2x2_pressure)
    # baseline.update(MP_2x2)
    # img_name = "2x2_avg_time"
    # img_name = "2x2_speed_pressure"
    img_name = "2x2_speed_time_relation"

    # plot_avg_time(
    #     algs=names,
    #     baselines=baseline,
    #     path="records/plt_result",
    #     img_name=img_name,
    # )
    plot_reward_and_avg_time_replation(
        algs=names,
        path="records/plt_result",
        img_name=img_name,
    )
