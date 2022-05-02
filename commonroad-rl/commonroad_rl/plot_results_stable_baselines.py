"""
Module for plotting learning curves
"""
import os

os.environ["KMP_WARNINGS"] = "off"
import logging
import matplotlib

matplotlib.use("TkAgg")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
import argparse

from stable_baselines.results_plotter import load_results, ts2xy

__author__ = "Peter Kocsis, Xiao Wang"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"


def argsparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--log-folder", help="Log folder", type=str, default="log"
    )
    parser.add_argument(
        "--model_path",
        "-model",
        type=str,
        nargs="+",
        default=(),
        help="(tuple) Relative path of the to be plotted model from the log folder",
    )
    parser.add_argument(
        "--legend_name",
        "-legend",
        type=str,
        nargs="+",
        default=(),
        help="(tuple) Legend informations in the same order as the model_name",
    )
    parser.add_argument(
        "--no_render", "-nr", action="store_true", help="Whether to render images"
    )
    parser.add_argument(
        "-t", "--title", help="Figure title", type=str, default="result"
    )
    return parser.parse_args()


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def ts2violationoroffroad(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = []
    for off, fric in zip(
        timesteps.is_off_road.values, timesteps.is_friction_violation.values
    ):
        y_var.append(off or fric)
    y_var = np.array(y_var)

    return x_var, y_var


def ts2timeoutoroffroad(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = []
    for off, fric in zip(timesteps.is_off_road.values, timesteps.is_time_out.values):
        y_var.append(off or fric)
    y_var = np.array(y_var)

    return x_var, y_var


def ts2goal(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = timesteps.is_goal_reached.values

    return x_var, y_var


def ts2offroad(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = timesteps.is_off_road.values

    return x_var, y_var


def ts2collision(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = timesteps.is_collision.values

    return x_var, y_var

def ts2rule_violation(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(timesteps.l.values)
    y_var = timesteps.num_traffic_rule_violation.values

    return x_var, y_var

def ts2friction_violation(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = timesteps.is_friction_violation.values

    return x_var, y_var


def ts2ep_rew_mean(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = timesteps.r.values / timesteps.l.values

    return x_var, y_var


def ts2max_time(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = timesteps.is_time_out.values

    return x_var, y_var


def ts2off_road(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(timesteps.l.values)
    y_var = timesteps.is_off_road.values

    return x_var, y_var


PLOT_DICT = {
    "Mean Reward": ts2ep_rew_mean,
    "Total Reward": ts2xy,
    "Collision": ts2collision,
    # "Friction violation or offroad": ts2violationoroffroad,
    "Goal": ts2goal,
    # "Max time reached": ts2max_time,
    # "Traffic rule violation": ts2rule_violation,
    "Friction violation": ts2friction_violation,
    "Off road": ts2off_road,
}


def plot_results(log_folder: str, title: str = "Learning Curve", legend=None):
    """
    Plot the results

    :param log_folder: The save location of the results to plot
    :param title: The title of the task to plot
    :param legend: Legend label of the data
    """
    results = load_results(log_folder)
    num_of_columns = 2
    num_of_rows = math.ceil(len(PLOT_DICT) / num_of_columns)

    for i, (k, v) in enumerate(PLOT_DICT.items()):
        plt.subplot(num_of_rows, num_of_columns, i + 1)
        x, y = v(results, "timesteps")
        y = moving_average(y, window=10)
        # Truncate x
        x = x[len(x) - len(y) :] * 1e-3
        plt.plot(x, y, label=legend)
        plt.xlabel("Number of Timesteps * 1e3")
        plt.ylabel(k)
        plt.title(f"{k}")
        plt.legend()

    plt.tight_layout()
    plt.suptitle(title)


def main():
    args = argsparser()

    log_dir = args.log_folder
    model_paths = tuple(args.model_path)
    legend_names = tuple(args.legend_name)

    # plot_results(result_dir, result_dir, title="train")

    # plot_results(os.path.join(result_dir, "test"), result_dir, title="test")
    # models = ["sparse_reward", "dense_reward"]
    # for m in models:
    #     plot_results(os.path.join(log_dir, m, args.algo, model_name, "test"), result_dir, legend=m)

    for idx, model in enumerate(model_paths):
        for figure in ["test", "train"]:
            fig = plt.figure(figsize=(16, 9), dpi=100)
            if figure == "test":
                model_log_path = os.path.join(log_dir, model, "test")
                if not os.path.isdir(model_log_path):
                    continue
            else:
                model_log_path = os.path.join(log_dir, model)
            if len(legend_names) > idx:
                legend_name = legend_names[idx] + "-" + figure
            else:
                legend_name = model + "-" + figure
            plot_results(model_log_path, title=figure, legend=legend_name)

            fig.savefig(os.path.join(log_dir, model, f"{figure}.png"))
            LOGGER.info("Saved %s.png to %s", figure, os.path.join(log_dir, model))
            plt.show()


if __name__ == "__main__":
    main()
