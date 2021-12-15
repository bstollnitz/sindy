"""Predicts a trajectory given the output from fitting (which is xi), and
an initial condition."""

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from common import (DATA_DIR, OUTPUT_DIR, POLYNOMIAL_ORDER, USE_TRIG,
                    create_library, get_absolute_dir)


def lorenz_approximation(_: float, u: np.ndarray, xi: np.ndarray,
                         polynomial_order: int, use_trig: bool) -> np.ndarray:
    """For each 1 x 3 u vector, this function calculates the corresponding
    1 x n row of the theta matrix, and multiples that theta row by the n x 3 xi.
    The result is the corresponding 1 x 3 du/dt vector."""
    theta = create_library(u.reshape((1, 3)), polynomial_order, use_trig)
    return theta @ xi


def compute_trajectory(u0: np.ndarray, xi: np.ndarray, polynomial_order: int,
                       use_trig: bool) -> np.ndarray:
    """Calculates the trajectory of the model discovered by SINDy.

    Given an initial value for u, we call the lorenz_approximation function
    to calculate the corresponding du/dt. Then, using a
    numerical method, solve_ivp uses that derivative to calculate the next
    value of u, using the xi matrix discovered by SINDy.
    This process repeats, until we have all values of the u
    trajectory.
    """
    t0 = 0.001
    dt = 0.001
    tmax = 100
    n = int(tmax / dt + 1)

    t_eval = np.linspace(start=t0, stop=tmax, num=n)
    result = solve_ivp(fun=lorenz_approximation,
                       t_span=(t0, tmax),
                       y0=u0,
                       t_eval=t_eval,
                       args=(xi, polynomial_order, use_trig))
    u = result.y.T

    return u


def style_axis(axis):
    """Styles a graph's x, y, or z axis."""
    # pylint: disable=protected-access
    axis._axinfo["grid"]["color"] = "#dddddd"
    axis._axinfo["grid"]["linewidth"] = 0.4
    axis._axinfo["tick"]["linewidth"][True] = 0.4
    axis._axinfo["tick"]["linewidth"][False] = 0.4
    axis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    axis.line.set_color("#bbbbbb")
    axis.label.set_color("#333333")
    pass


def style_axis3d(axis3d):
    """Styles a 3D graph."""
    axis3d.set_xlabel("x")
    axis3d.set_ylabel("y")
    axis3d.set_zlabel("z")
    axis3d.tick_params(axis="x", colors="#666666")
    axis3d.tick_params(axis="y", colors="#666666")
    axis3d.tick_params(axis="z", colors="#666666")
    style_axis(axis3d.w_xaxis)
    style_axis(axis3d.w_yaxis)
    style_axis(axis3d.w_zaxis)
    axis3d.set_title(axis3d.get_title(), fontdict={"color": "#333333"})


def graph_results(u: np.ndarray, u_approximation: np.ndarray) -> None:
    """Graphs the original trajectory, and the one computed from the model
    discovered by SINDy, side-by-side."""
    sample_count = 20000
    orange = "#EF6C00"

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Graph trajectory from the true model.
    axis3d = fig.add_subplot(1, 2, 1, projection="3d")
    x = u[0:sample_count, 0]
    y = u[0:sample_count, 1]
    z = u[0:sample_count, 2]
    axis3d.plot3D(x, y, z, orange, linewidth=0.4)
    axis3d.set_title("Original Lorenz trajectory")
    style_axis3d(axis3d)

    # Graph trajectory computed from model discovered by SINDy.
    axis3d = fig.add_subplot(1, 2, 2, projection="3d")
    x = u_approximation[0:sample_count, 0]
    y = u_approximation[0:sample_count, 1]
    z = u_approximation[0:sample_count, 2]
    axis3d.plot3D(x, y, z, orange, linewidth=0.4)
    axis3d.set_title("SINDy approximation")
    style_axis3d(axis3d)

    plt.show()


def main() -> None:
    logging.info("Predicting.")
    parser = argparse.ArgumentParser()

    default_data_dir = get_absolute_dir(DATA_DIR)
    parser.add_argument("--data_dir", dest="data_dir", default=default_data_dir)

    default_output_dir = get_absolute_dir(OUTPUT_DIR)
    parser.add_argument("--output_dir",
                        dest="output_dir",
                        default=default_output_dir)

    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        u = np.array(file_read.get("u"))

    output_file_dir = Path(output_dir, "output.hdf5")
    with h5py.File(output_file_dir, "r") as file_read:
        xi = np.array(file_read.get("xi"))

    u0 = np.array([-8, 8, 27])
    u_approximation = compute_trajectory(u0, xi, POLYNOMIAL_ORDER, USE_TRIG)
    graph_results(u, u_approximation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
