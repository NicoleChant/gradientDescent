import numpy as np
import matplotlib.pyplot as plt
from utils import set_axis_attrs
from descent import loss
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_contour(X, y) -> tuple:
    x_axis = np.linspace(-1, 1, 50)
    y_axis = np.linspace(-1, 1, 50)
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax = np.array([ax1, ax2, ax3])

    fig.suptitle("Gradient Descent Simulation", fontsize=16)
    Z = np.array([[loss(X, y, i, j) for i in x_axis] for j in y_axis])

    im = ax[0].contourf(x_axis, y_axis, Z, cmap=plt.cm.bone)
    set_axis_attrs(ax[0],
                   xlabel_size="14",
                   ylabel_size="14",
                   title="Gradient Descent")
    ax[0].contour(im, levels=im.levels[::2], colors='r')

    # create colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax)

    ax[1].plot([], label="loss")
    set_axis_attrs(ax[1],
                   title="Loss History",
                   xlabel="Epochs",
                   ylabel="Loss Val.",
                   xlabel_size="14",
                   ylabel_size="14")
    ax[1].legend(loc=0, )

    ax[2].scatter(X, y, color='orange', alpha=0.6)
    set_axis_attrs(ax[2],
                   title="Linear Regression",
                   xlabel="x",
                   ylabel="y",
                   xlabel_size="14",
                   ylabel_size="14",
                   title_size="16")

    # disable if using constrainted layout
    # plt.tight_layout()
    return fig, ax
