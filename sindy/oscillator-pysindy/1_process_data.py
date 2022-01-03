"""Data proccessing step."""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from common import DATA_DIR, get_absolute_dir
from utils_video import track_object


def main() -> None:
    logging.info("Processing data.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        dest="data_dir",
                        default=get_absolute_dir(DATA_DIR))
    args = parser.parse_args()
    data_dir = args.data_dir

    input_file_name = "damped_oscillator_900.mp4"
    input_file_path = str(Path(data_dir, input_file_name))
    bbox = (269, 433, 378, 464)
    output_file_name = "damped_oscillator_900_tracked.avi"
    output_file_path = str(Path(data_dir, output_file_name))
    u = track_object(input_file_path, bbox, output_file_path)

    t = np.arange(stop=u.shape[0])

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="u", data=u)
        file.create_dataset(name="t", data=t)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
