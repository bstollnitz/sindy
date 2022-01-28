"""Data proccessing step."""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from common import ORIGINAL_DATA_DIR, DATA_DIR
from utils_video import track_object


def main() -> None:
    logging.info("Processing data.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--original-data_dir",
                        dest="original_data_dir",
                        default=ORIGINAL_DATA_DIR)
    args = parser.parse_args()
    original_data_dir = args.original_data_dir
    data_dir = args.data_dir

    input_file_name = "damped_oscillator_900.mp4"
    input_file_path = str(Path(original_data_dir, input_file_name))
    bbox = (269, 433, 378, 464)
    Path(data_dir).mkdir(exist_ok=True)
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
