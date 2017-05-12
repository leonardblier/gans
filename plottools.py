import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_metrics(d, namefile="metrics.png"):
    """
    Plot a dictionary of metrics
    """
    fig, axes = plt.subplots(nrows=len(d), ncols=1)
    ax_list = axes.flatten()

    for metric, ax in zip(d, ax_list):
        ax.set_title(metric)
        ax.plot(d[metric])

    fig.tight_layout()
    plt.savefig(namefile)
    plt.clf()
    plt.close()


def plot_batch(batch, namefile=None):
    fig, axes = plt.subplots(4, 8,
                             subplot_kw={'xticks': [], 'yticks': []})
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in zip(range(32), axes.flat):
        ax.imshow(((batch[i]+1)*(255./2)).astype(np.uint8),
                  interpolation=None,
                  cmap="gray")
    fig.tight_layout()
    if namefile is None:
        plt.show()
    else:
        plt.savefig(namefile)
    plt.clf()
    plt.close()
