"""Utilities related to graphing."""

import matplotlib.pyplot as plt
import numpy as np


def style_axis2d(ax, xlabel: str, ylabel: str):
    """Styles a 2D graph.
    """
    very_light_gray = "#eeeeee"
    light_gray = "#999999"
    dark_gray = "#444444"

    ax.set_xlabel(xlabel, {"color": dark_gray})
    ax.set_ylabel(ylabel, {"color": dark_gray})
    ax.set_title(ax.get_title(), {"color": dark_gray})
    ax.tick_params(axis="x", colors=light_gray)
    ax.tick_params(axis="y", colors=light_gray)
    ax.set_facecolor(very_light_gray)
    for spine in ax.spines.values():
        spine.set_edgecolor(light_gray)


def graph_results(u: np.ndarray, u_approximation: np.ndarray) -> None:
    """Graphs two 2D trajectories side-by-side.
    """
    orange = "#EF6C00"

    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)
    ax1.plot(u[:, 0], u[:, 1], color=orange, linewidth=0.4)
    ax1.set_title("Original trajectory")
    style_axis2d(ax1, "x", "y")
    ax2.plot(u_approximation[:, 0],
             u_approximation[:, 1],
             color=orange,
             linewidth=0.4)
    ax2.set_title("SINDy approximation")
    style_axis2d(ax2, "x", "y")

    plt.show()


def graph_result_t(u: np.ndarray, u_approximation: np.ndarray,
                   t: np.ndarray) -> None:
    """Graphs two 2D trajectories side-by-side, showing y(t) and t
    (ignoring x).
    """
    orange = "#EF6C00"

    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)
    ax1.plot(t[:], u[:, 1], color=orange, linewidth=0.4)
    ax1.set_title("Original trajectory")
    style_axis2d(ax1, "t", "y")
    ax2.plot(t[:], u_approximation[:, 1], color=orange, linewidth=0.4)
    ax2.set_title("SINDy approximation")
    style_axis2d(ax2, "t", "y")

    plt.show()
