"""Predicts a trajectory given the output from fitting (which is xi), and
an initial condition."""

import argparse
import logging
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps

from common import DATA_DIR, OUTPUT_DIR, get_absolute_dir


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


def compute_trajectory(u0: np.ndarray, model: ps.SINDy) -> np.ndarray:
    """Calculates the trajectory of the model discovered by SINDy.
    """
    t0 = 0.001
    dt = 0.001
    tmax = 100
    n = int(tmax / dt + 1)
    t_eval = np.linspace(start=t0, stop=tmax, num=n)

    u_approximation = model.simulate(u0, t_eval)

    return u_approximation


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

    output_file_dir = Path(output_dir, "output.obj")
    with open(output_file_dir, "rb") as file_read:
        model = pickle.load(file_read)

    u0 = np.array([-8, 8, 27])
    u_approximation = compute_trajectory(u0, model)
    graph_results(u, u_approximation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
