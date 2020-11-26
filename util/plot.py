from os import lseek
from typing import List
import matplotlib.pyplot as plt


def get_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        dict = eval(f.read())
        return dict


def plot(ax, x: List, y: List, x_lable, y_label, title):
    ax.plot(x, y, label='linear')
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    out = ax.set_title(title)
    return out


if __name__ == "__main__":
    data_path = "./obs.txt"
    data = get_data(data_path)
    episodes = []
    rewards = []
    for k, v in data["reward"].items():
        episodes.append(int(k))
        rewards.append(int(v))
    fig, ax = plt.subplots()
    plot(ax, episodes, rewards, x_lable="episodes",
         y_label="reward", title="rewards")
    ax
    fig, ax2 = plt.subplots()
    episodes = []
    loss = []
    for k, v in data["loss"].items():
        episodes.append(int(k))
        loss.append(int(v))
    plot(ax2, episodes, loss, x_lable="episodes",
         y_label="loss", title="loss")
    plt.show()
