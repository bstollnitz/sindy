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


def main() -> None:
    logging.info("Fitting.")
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
        t = np.array(file_read.get("t"))

    optimizer = STLSQ(threshold=THRESHOLD, max_iter=MAX_ITERATIONS)

    # Total variation derivatives with regularization.
    # differentiation_method = SINDyDerivative(
    #     derivative_kws={"trend_filtered", 0, 1e-2})

    # Finite difference derivatives.
    differentiation_method = FiniteDifference()

    model = ps.SINDy(optimizer=optimizer,
                     differentiation_method=differentiation_method,
                     feature_names=["x", "y", "z"],
                     discrete_time=False)
    model.fit(u, t=t)
    model.print()

    output_file_dir = Path(output_dir, "output.obj")
    with open(output_file_dir, "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
