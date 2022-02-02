"Constants and common code."

from pathlib import Path

# Directory names.
DATA_DIR = str(Path(Path(__file__).parent.parent, "data"))
OUTPUT_DIR = str(Path(Path(__file__).parent.parent, "output"))

# Constants used in the Sequential Thresholded Least-Squares algorithm.
THRESHOLD = 0.025
MAX_ITERATIONS = 10

# The parameters of the Lorenz equation.
SIGMA = 10
RHO = 28
BETA = 8 / 3
