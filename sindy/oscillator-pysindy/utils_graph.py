"""Utilities related to graphing."""

import matplotlib.pyplot as plt
import numpy as np


def style_axis2d(axis):
    """Styles a 2D graph."""
    very_light_gray = "#eeeeee"
    light_gray = "#999999"
    dark_gray = "#444444"

    axis.set_xlabel("x", {"color": dark_gray})
    axis.set_ylabel("y", {"color": dark_gray})
    axis.set_title(axis.get_title(), {"color": dark_gray})
    axis.tick_params(axis="x", colors=light_gray)
    axis.tick_params(axis="y", colors=light_gray)
    axis.set_facecolor(very_light_gray)
    for spine in axis.spines.values():
        spine.set_edgecolor(light_gray)


def graph_results(u: np.ndarray, u_approximation: np.ndarray) -> None:
    """Graphs two 3D trajectories side-by-side."""
    sample_count = 400
    orange = "#EF6C00"

    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)
    ax1.plot(u[0:sample_count, 0],
             u[0:sample_count, 1],
             color=orange,
             linewidth=0.4)
    ax1.set_title("Original trajectory")
    style_axis2d(ax1)
    ax2.plot(u_approximation[0:sample_count, 0],
             u_approximation[0:sample_count, 1],
             color=orange,
             linewidth=0.4)
    ax2.set_title("SINDy approximation")
    style_axis2d(ax2)

    plt.show()
