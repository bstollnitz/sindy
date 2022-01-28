"""Utilities related to graphing."""

import matplotlib.pyplot as plt
import numpy as np


def style_axis2d(ax, xlabel: str, ylabel: str):
    """Styles a 2D graph.
    """
    very_light_gray = "#efefef"
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


def graph_result(u: np.ndarray, u_approximation_x: np.ndarray,
                 u_approximation_y: np.ndarray, t: np.ndarray) -> None:
    """Graphs y(t).
    """
    orange = "#EF6C00"
    teal = "#007b96"

    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)

    ax1.plot(t[:], u[:, 0], color=orange, linewidth=0.4)
    ax1.plot(t[:], u_approximation_x[:, 0], color=teal, linewidth=0.4)
    ax1.set_title("x(t)")
    style_axis2d(ax1, "t", "x")
    ax1.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)

    ax2.plot(t[:], u[:, 1], color=orange, linewidth=0.4)
    ax2.plot(t[:], u_approximation_y[:, 0], color=teal, linewidth=0.4)
    ax2.set_title("y(t)")
    style_axis2d(ax2, "t", "y")
    ax2.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)

    plt.show()
