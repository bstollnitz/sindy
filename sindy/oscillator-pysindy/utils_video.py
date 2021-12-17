"""Utilities related to video."""

import logging
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


def track_object(input_file_path: str, bbox: Tuple[int],
                 output_file_path: str) -> np.ndarray:
    """
    Tracks an object in a video defined by the bounding box passed
    as a parameter. Saves an output video showing the bounding box found.
    Returns an ndarray of shape (frames, 2) containing the
    (x, y) center of the bounding box in each frame.
    """
    logging.info("Tracking object.")

    # Open video.
    capture = cv2.VideoCapture(input_file_path)
    if not capture.isOpened():
        raise Exception(f"Could not open video: {input_file_path}.")

    # Define tracker.
    tracker = cv2.TrackerCSRT_create()

    # Get the codec for an AVI file.
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Read each frame of the video, gather centers of all bounding boxes, and
    # save video showing those bounding boxes.
    color = (0, 0, 255)
    writer = None
    centers = []
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        (ok, frame) = capture.read()
        if not ok:
            raise Exception(f"Cannot read video frame {i}.")

        if i == 0:
            # Uncomment this to select a bounding box interactively.
            # bbox = cv2.selectROI("Frame",
            #                      frame,
            #                      fromCenter=False,
            #                      showCrosshair=True)
            tracker.init(frame, bbox)
            writer = cv2.VideoWriter(output_file_path,
                                     fourcc,
                                     fps=30.0,
                                     frameSize=(frame.shape[1], frame.shape[0]))
        else:
            (ok, bbox) = tracker.update(frame)
            if not ok:
                raise Exception(f"Could not track object in frame {i}.")
        center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        centers.append(center)

        # Color the bounding box and center of an output video frame.
        x = int(center[0])
        y = int(center[1])
        fill_rectangle(frame, (x - 2, y, 2, 5, 5), color)
        outline_rectangle(frame, bbox, color)
        writer.write(frame)

    capture.release()
    writer.release()

    return np.asarray(centers)


def fill_rectangle(frame: np.ndarray, bbox: Tuple[int],
                   color: Tuple[int]) -> None:
    """Draws a filled rectangle with the specified bounding box and color
    on the frame.
    """
    for x in range(bbox[0], bbox[0] + bbox[2]):
        for y in range(bbox[1], bbox[1] + bbox[3]):
            set_color(frame, y, x, color)


def outline_rectangle(frame: np.ndarray, bbox: Tuple[int],
                      color: Tuple[int]) -> None:
    """Draws the outline of a rectangle with the specified bounding box and
    color on the frame.
    """
    for x in range(bbox[0], bbox[0] + bbox[2]):
        set_color(frame, bbox[1], x, color)
        set_color(frame, bbox[1] + bbox[3] - 1, x, color)

    for y in range(bbox[1], bbox[1] + bbox[3]):
        set_color(frame, y, bbox[0], color)
        set_color(frame, y, bbox[0] + bbox[2] - 1, color)


def set_color(frame: np.ndarray, x: int, y: int, color: Tuple[int]) -> None:
    """Sets the pixel at position (x, y) on the frame to the specified color.
    """
    frame[x, y, :] = np.asarray(color)
