"""Predicts a trajectory using the SINDy model."""

import argparse
import logging
import pickle
from pathlib import Path

import h5py
import numpy as np

from common import DATA_DIR, OUTPUT_DIR, get_absolute_dir
from utils_graph import graph_result


def main() -> None:
    logging.info("Predicting.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        dest="data_dir",
                        default=get_absolute_dir(DATA_DIR))
    parser.add_argument("--output_dir",
                        dest="output_dir",
                        default=get_absolute_dir(OUTPUT_DIR))
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "r") as file_read:
        centers = np.array(file_read.get("centers"))
        t = np.array(file_read.get("t"))

    models_file_path = Path(output_dir, "models.pkl")
    with open(models_file_path, "rb") as file_read:
        (modelx, modely) = pickle.load(file_read)

    derivatives_file_path = Path(output_dir, "derivatives.hdf5")
    with h5py.File(derivatives_file_path, "r") as file_read:
        xdot = np.array(file_read.get("xdot"))
        ydot = np.array(file_read.get("ydot"))

    u0x = np.array([centers[0, 0], xdot[0, 0]])
    centers_approximation_x = modelx.simulate(u0x, t)

    u0y = np.array([centers[0, 1], ydot[0, 0]])
    centers_approximation_y = modely.simulate(u0y, t)

    graph_result(centers, centers_approximation_x, centers_approximation_y, t)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
