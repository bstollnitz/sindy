"""Fits the model using PySindy."""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pysindy as ps
# pylint: disable=unused-import
from pysindy.differentiation import FiniteDifference, SINDyDerivative
from pysindy.optimizers import STLSQ

from common import (DATA_DIR, MAX_ITERATIONS, OUTPUT_DIR, THRESHOLD,
                    get_absolute_dir)


def fit(u: np.ndarray,
        t: np.ndarray) -> Tuple[ps.SINDy, ps.SINDy, np.ndarray, np.ndarray]:
    """Uses PySINDy to find the equation that best fits the data u.
    """
    optimizer = STLSQ(threshold=THRESHOLD, max_iter=MAX_ITERATIONS)

    differentiation_method = FiniteDifference()
    # pylint: disable=protected-access
    udot = differentiation_method._differentiate(u, t)

    # Get a model for the movement in x.
    logging.info("Model for x")
    x = u[:, 0:1]
    xdot = udot[:, 0:1]
    datax = np.hstack((x, xdot))
    modelx = ps.SINDy(optimizer=optimizer,
                      differentiation_method=differentiation_method,
                      feature_names=["x", "xdot"],
                      discrete_time=False)
    modelx.fit(datax, t=t, ensemble=True)
    modelx.print()
    logging.info("coefficients: %s", modelx.coefficients().T)

    # Get a model for the movement in y.
    logging.info("Model for y")
    y = u[:, 1:2]
    ydot = udot[:, 1:2]
    datay = np.hstack((y, ydot))
    modely = ps.SINDy(optimizer=optimizer,
                      differentiation_method=differentiation_method,
                      feature_names=["y", "ydot"],
                      discrete_time=False)
    modely.fit(datay, t=t, ensemble=True)
    modely.print()
    logging.info("coefficients: %s", modely.coefficients().T)

    return (modelx, modely, xdot, ydot)


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
        t = np.array(file_read.get("t"))

    (modelx, modely, xdot, ydot) = fit(centers, t)

    output_file_dir = Path(output_dir, "models.pkl")
    with open(output_file_dir, "wb") as file:
        pickle.dump((modelx, modely), file)

    output_file_dir = Path(output_dir, "derivatives.hdf5")
    with h5py.File(output_file_dir, "w") as file:
        file.create_dataset(name="xdot", data=xdot)
        file.create_dataset(name="ydot", data=ydot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
