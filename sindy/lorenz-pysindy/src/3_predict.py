"""Predicts a trajectory given the output from fitting (which is xi), and
an initial condition."""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pysindy as ps
from IPython import get_ipython

from common import DATA_DIR, OUTPUT_DIR
from utils_graph import graph_results


def compute_trajectory(u0: np.ndarray, model: ps.SINDy) -> np.ndarray:
    """Calculates the trajectory using the model discovered by SINDy.
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
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", dest="output_dir", default=OUTPUT_DIR)
    shell = get_ipython().__class__.__name__
    argv = [] if (shell == "ZMQInteractiveShell") else sys.argv[1:]
    args = parser.parse_args(argv)
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
