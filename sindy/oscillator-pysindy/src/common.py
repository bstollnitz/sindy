"Constants and common code."

from pathlib import Path

# Directory names.
ORIGINAL_DATA_DIR = str(Path(Path(__file__).parent.parent, "original-data"))
DATA_DIR = str(Path(Path(__file__).parent.parent, "data"))
OUTPUT_DIR = str(Path(Path(__file__).parent.parent, "output"))

# Constants used in the Sequential Thresholded Least-Squares algorithm.
THRESHOLD = 0.001
MAX_ITERATIONS = 100
