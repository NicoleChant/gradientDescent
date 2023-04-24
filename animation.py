########################################
### Gradient Descent Live Animation  ###
### ~ Made by Channi <3              ###
########################################

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from termcolor import colored
from utils import set_axis_attrs
from descent import solvers, h
from plot import plot_contour
from sklearn.linear_model import LinearRegression

def train_model(X_trans, y):
    model = LinearRegression()
    model.fit(X_trans, y)
    return model

def get_X_y() -> tuple[pd.Series, pd.Series]:
    data = pd.read_csv("data.csv")
    X = data["zinc"]
    y = data["phosphorus"]
    return X, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="minibatch",
                        type=str,
                        choices=["normal", "minibatch"])
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--z0", default=-1.0, type=float)
    parser.add_argument("--w0", default=-1.0, type=float)
    parser.add_argument("--scaler", default=None, type=str, choices=["minMax", "standard", "robust"])
    parser.add_argument("--interval", default=50.0, type=float)

    args = parser.parse_args()
    mode = args.mode
    scaler = args.scaler
    interval = args.interval
    kwargs = vars(args)
    print(colored(f"Chosen arguments: {kwargs}", "green"))
    to_drop: list[str] = ["scaler", "mode", "interval"]
    for key in to_drop:
        kwargs.pop(key)

    chosen_solver = solvers[mode]

    scalers: dict[str, Callable] = {"minMax": MinMaxScaler(),
                                    "robust": RobustScaler(),
                                    "standard": StandardScaler()}
    chosen_scaler = scalers.get(scaler)

    plt.style.use("dark_background")

    a_history: list[float] = []
    b_history: list[float] = []
    loss_history: list[float] = []

    X, y = get_X_y()

    assert X.shape[0] > kwargs["batch_size"], "Error: Batch size must be smaller than number of rows!"

    if chosen_scaler:
        X_trans = chosen_scaler.fit_transform(X)
    else:
        X_trans = X

    # reshape data from feeding scikit-learning estimator
    if isinstance(X, np.ndarray):
        trained_model = train_model(X.reshape(-1, 1), y)
    elif isinstance(X, pd.Series):
        trained_model = train_model(pd.DataFrame(X), y)
    else:
        trained_model = train_model(X, y)

    gen = chosen_solver(X_trans, y, **kwargs)
    fig, ax = plot_contour(X, y)
    best_point = type("Best", (), {"z": kwargs["z0"],
                                   "w": kwargs["w0"],
                                   "done": False})

    def anime(i):
        try:
            z, w, cur_loss = next(gen)
            best_point.z = z
            best_point.w = w
            flag = False
        except StopIteration as err:
            if not best_point.done:
                print(colored(f"Stopping {err}...", "red"))
            flag = True


        # update history
        if not flag:
            a_history.append(z)
            b_history.append(w)
            loss_history.append(cur_loss)

        # update plot
        ax[0].scatter(a_history,
                   b_history,
                   alpha=0.3,
                   color='yellow')
        set_axis_attrs(ax[0], title="Gradient Descent",
                xlabel="x",
                ylabel="y",
                xlabel_size="14",
                ylabel_size="14")
        if not best_point.done and flag:
            best_point.done = True
            ax[0].scatter(best_point.z, best_point.w,
                          label=f"Best: ({best_point.z:.2f}, {best_point.w:.2f})",
                          marker='x',
                          color='red')
            ax[0].legend(loc=0)
        elif best_point.done:
            ax[0].scatter(best_point.z,
                          best_point.w,
                          label=f"({best_point.z}, {best_point.w})",
                          marker='x',
                          color='red')

        ax[1].clear()
        ax[1].plot(loss_history[-50:],
                   label="loss",
                   color='pink',
                   alpha=1.0,
                   lw=1.2,
                   linestyle='--')
        ax[1].grid(lw=0.4, alpha=0.6)

        set_axis_attrs(ax[1], title="Loss History",
                xlabel="Epochs",
                ylabel="Loss Val.",
                xlabel_size="14",
                ylabel_size="14")
        ax[1].legend(loc=0,)

        ax[2].clear()
        ax[2].scatter(X, y, color='orange', alpha=0.6)
        x_axis = np.linspace(X.min() - 0.1, X.max() + 0.1, 200)
        ax[2].plot(x_axis, h(x_axis, best_point.z, best_point.w),
                   label="OLS Line",
                   alpha=0.7,
                   color='lightblue')
        ax[2].plot(x_axis,
                x_axis * trained_model.coef_[0] + trained_model.intercept_,
                label="Scikit-learn OLS",
                color='pink',
                alpha=0.6,
                linestyle='--')
        set_axis_attrs(ax[2], title="Linear Regression",
                    xlabel="x",
                    ylabel="y",
                    xlabel_size="14",
                    ylabel_size="14",
                    title_size="16")
        ax[2].legend(loc=0)

        # do not used tight layout if using constrainted layout
        # plt.tight_layout()
        return ax

    ani = FuncAnimation(fig, anime, interval=interval,)
    plt.show()
