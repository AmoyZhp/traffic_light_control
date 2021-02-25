from typing import List


import matplotlib.pyplot as plt


def save_fig(x: List, y: List, x_lable, y_label, title, dir, img_name=""):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='linear')
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if img_name:
        fig.savefig("{}/{}".format(dir, img_name))
    plt.clf()
    plt.close(fig)
