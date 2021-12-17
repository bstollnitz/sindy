"""Predicts a trajectory using the SINDy model."""

import argparse
import logging
import pickle
from pathlib import Path

import h5py
import numpy as np

from common import DATA_DIR, OUTPUT_DIR, get_absolute_dir
# pylint: disable=unused-import
from utils_graph import graph_results, graph_result_t


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

    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        centers = np.array(file_read.get("centers"))

    output_file_dir = Path(output_dir, "output.obj")
    with open(output_file_dir, "rb") as file_read:
        model = pickle.load(file_read)

    center0 = np.array([(269 + 378) / 2, (433 + 464) / 2])
    t = np.linspace(start=0, stop=centers.shape[0], num=centers.shape[0])
    centers_approximation = model.simulate(center0, t)
    # graph_results(centers, centers_approximation)
    graph_result_t(centers, centers_approximation, t)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
