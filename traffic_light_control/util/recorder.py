import datetime
import json
import os
import torch
from typing import List
import matplotlib.pyplot as plt


def create_record_dir(root_record, info, last_record="") -> str:
    # 创建的目录
    date = datetime.datetime.now()
    sub_dir = "{}_{}_{}_{}_{}_{}_{}_{}/".format(
        info["env_id"],
        info["alg_id"],
        date.year,
        date.month,
        date.day,
        date.hour,
        date.minute,
        date.second,
    )
    record_dir = root_record + sub_dir
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    else:
        print("create record folder error , path exist : ", record_dir)

    param_path = record_dir + "params/"
    if not os.path.exists(param_path):
        os.mkdir(param_path)
    else:
        print("create record folder error , path exist : ", param_path)

    fig_path = record_dir + "figs/"
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    else:
        print("create record folder error , path exist : ", fig_path)

    data_path = record_dir + "data/"
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        print("create record folder error , path exist : ", data_path)

    return record_dir


def snapshot_params(config, weight, record, params_file):
    params = {
        "config": config,
        "weight": weight,
        "record": record,
    }
    torch.save(params, params_file)


def snapshot_exp_result(record_dir,
                        central_record, local_record,
                        train_info):
    saved_data = {
        "central": central_record,
        "local": local_record,
    }
    saved_data_file_name = "exp_result.txt"
    result_file = record_dir + "data/" + saved_data_file_name
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(str(saved_data))
    save_exp_result(record_dir + "figs/", result_file)

    train_info_file = record_dir + "data/" + "train_info.json"
    with open(train_info_file, "w", encoding="utf-8") as f:
        json.dump(train_info, f)


def save_exp_result(record_dir, data_file):
    data = {}
    with open(data_file, "r", encoding="utf-8") as f:
        data = eval(f.read())
    central_data = data["central"]
    local_data = data["local"]

    episodes = []
    rewards = []
    for ep, r in central_data["reward"].items():
        episodes.append(int(ep))
        rewards.append(float(r))
    savefig(
        episodes, rewards, x_lable="episodes",
        y_label="reward", title="central rewards",
        img=record_dir+"central_reward.png")

    episodes = []
    loss = []
    for ep, l in central_data["loss"].items():
        episodes.append(int(ep))
        loss.append(float(l))
    savefig(
        episodes, rewards, x_lable="episodes",
        y_label="reward", title="central loss",
        img=record_dir+"central_loss.png")

    episodes = []
    mean_eval_reward = []
    eval_reward_mean = central_data["eval_reward"]["mean"]
    for ep, r in eval_reward_mean.items():
        episodes.append(int(ep))
        mean_eval_reward.append(r)
    savefig(
        episodes, mean_eval_reward, x_lable="episodes",
        y_label="reward", title="eval rewards",
        img=record_dir+"eval_reward_mean.png")

    episodes = []
    average_travel_time = []
    for ep, t in central_data["average_travel_time"].items():
        episodes.append(ep)
        average_travel_time.append(t)
    savefig(episodes, average_travel_time,
            x_lable="episodes", y_label="average travel time",
            title="average travel time",
            img=record_dir+"average_travel_time.png")

    episodes = []
    average_travel_time = []
    for ep, t in central_data["eval_reward"]["average_travel_time"].items():
        episodes.append(ep)
        average_travel_time.append(t)
    savefig(episodes, average_travel_time,
            x_lable="episodes", y_label="average travel time",
            title="average travel time",
            img=record_dir+"eval_average_travel_time.png")

    for id_, val in local_data.items():

        episodes = []
        rewards = []
        for ep, r in val["reward"].items():
            episodes.append(int(ep))
            rewards.append(float(r))
        savefig(
            episodes, rewards, x_lable="episodes",
            y_label="reward", title="rewards",
            img=record_dir+"local_reward_{}.png".format(id_))

        episodes = []
        loss = []
        for ep, r in val["loss"].items():
            episodes.append(int(ep))
            loss.append(float(r))
        savefig(
            episodes, loss, x_lable="episodes",
            y_label="loss", title="loss",
            img=record_dir+"local_loss_{}.png".format(id_))


def savefig(x: List, y: List, x_lable, y_label, title, img=""):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='linear')
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if img != "":
        fig.savefig(img)
    plt.clf()
    plt.close(fig)
