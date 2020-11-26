from os import lseek
from typing import List
import matplotlib.pyplot as plt


def get_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        dict = eval(f.read())
        return dict


def plot(x: List, y: List, x_lable, y_label, title, img=""):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='linear')
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if img != "":
        fig.savefig(img)
