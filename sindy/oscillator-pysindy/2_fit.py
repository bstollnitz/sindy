"""Fits the model using PySindy."""

import argparse
import logging
import pickle
from pathlib import Path

import h5py
import numpy as np
import pysindy as ps
# pylint: disable=unused-import
from pysindy.differentiation import FiniteDifference, SINDyDerivative
from pysindy.optimizers import STLSQ

from common import (DATA_DIR, MAX_ITERATIONS, OUTPUT_DIR, THRESHOLD,
                    get_absolute_dir)


def fit(u: np.ndarray, t: np.ndarray) -> ps.SINDy:
    """Uses PySINDy to find the equation that best fits the data u.
    """
    optimizer = STLSQ(threshold=THRESHOLD, max_iter=MAX_ITERATIONS)

    differentiation_method = FiniteDifference()

    model = ps.SINDy(optimizer=optimizer,
                     differentiation_method=differentiation_method,
                     feature_names=["x", "y"],
                     discrete_time=False)
    model.fit(u, t=t)
    model.print()
    logging.info("xi: %s", model.coefficients().T)

    return model


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
        centers = np.array(file_read.get("centers"))

    t = np.linspace(start=0, stop=centers.shape[0], num=centers.shape[0])

    model = fit(centers, t)

    output_file_dir = Path(output_dir, "output.obj")
    with open(output_file_dir, "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
