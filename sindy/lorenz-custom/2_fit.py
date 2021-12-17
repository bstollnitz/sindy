"""Fits the model."""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from common import (DATA_DIR, OUTPUT_DIR, POLYNOMIAL_ORDER, USE_TRIG, THRESHOLD,
                    MAX_ITERATIONS, create_library, get_absolute_dir)


def calculate_regression(theta: np.ndarray, uprime: np.ndarray,
                         threshold: float, max_iterations: int) -> np.ndarray:
    """Finds a xi matrix that fits theta * xi = uprime, using the sequential
    threshdolded least-squares algorithm, which is a regression algorithm that
    promotes sparsity.

    The authors of the SINDy paper designed this algorithm as an alternative
    to LASSO, because they found LASSO to be unstable algorithmically, and
    computationally expensive for very large data sets.
    """
    # Solve Ax = b, theta * xi = uprime.
    xi = np.linalg.lstsq(theta, uprime, rcond=None)[0]
    n = xi.shape[1]

    # Perform regression.
    for _ in range(max_iterations):
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0
        for j in range(n):
            big_indices = np.logical_not(small_indices[:, j])
            xi[big_indices, j] = np.linalg.lstsq(theta[:, big_indices],
                                                 uprime[:, j],
                                                 rcond=None)[0]

    return xi


def main() -> None:
    logging.info("Fitting.")

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
        u = np.array(file_read.get("u"))
        uprime = np.array(file_read.get("uprime"))

    theta = create_library(u, POLYNOMIAL_ORDER, USE_TRIG)
    xi = calculate_regression(theta, uprime, THRESHOLD, MAX_ITERATIONS)
    logging.info("xi:\n %s", xi)

    output_file_dir = Path(output_dir, "output.hdf5")
    with h5py.File(output_file_dir, "w") as file:
        file.create_dataset(name="xi", data=xi)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
