"""Predicts a trajectory given the output from fitting (which is xi), and
an initial condition."""

import argparse
import logging
import sys
from pathlib import Path
from IPython import get_ipython

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from common import (DATA_DIR, OUTPUT_DIR, POLYNOMIAL_ORDER, USE_TRIG,
                    create_library)
from utils_graph import graph_results


def lorenz_approximation(_: float, u: np.ndarray, xi: np.ndarray,
                         polynomial_order: int, use_trig: bool) -> np.ndarray:
    """For a given 1 x 3 u vector, this function calculates the corresponding
    1 x n row of the theta matrix, and multiples that theta row by the n x 3 xi.
    The result is the corresponding 1 x 3 du/dt vector.
    """
    theta = create_library(u.reshape((1, 3)), polynomial_order, use_trig)
    return theta @ xi


def compute_trajectory(u0: np.ndarray, xi: np.ndarray, polynomial_order: int,
                       use_trig: bool) -> np.ndarray:
    """Calculates the trajectory of the model discovered by SINDy.

    Given an initial value for u, we call the lorenz_approximation function
    to calculate the corresponding du/dt. Then, using a numerical method,
    solve_ivp uses that derivative to calculate the next value of u, using the
    xi matrix discovered by SINDy. This process repeats, until we have all
    values of the u trajectory.
    """
    t0 = 0.001
    dt = 0.001
    tmax = 100
    n = int(tmax / dt + 1)

    t = np.linspace(start=t0, stop=tmax, num=n)
    result = solve_ivp(fun=lorenz_approximation,
                       t_span=(t0, tmax),
                       y0=u0,
                       t_eval=t,
                       args=(xi, polynomial_order, use_trig))
    u = result.y.T

    return u


def main() -> None:
    logging.info("Predicting.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", dest="output_dir", default=OUTPUT_DIR)
    shell = get_ipython().__class__.__name__
    argv = [] if (shell == "ZMQInteractiveShell") else sys.argv
    args = parser.parse_args(argv)

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
