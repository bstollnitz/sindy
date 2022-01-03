"Constants and common code."

from pathlib import Path

# Directory names.
DATA_DIR = "data"
OUTPUT_DIR = "output"

# Constants used in the Sequential Thresholded Least-Squares algorithm.
THRESHOLD = 0.001
MAX_ITERATIONS = 100


def get_absolute_dir(dir_name: str, create_dir: bool = True) -> Path:
    """Creates a directory with the specified name in a location relative to
    the code file, and returns the absolute path to that directory.
    
    This way, the behavior of our code will be the same, regardless of the
    location from where we run the project.
    """
    parent_path = Path(__file__).parent
    path = Path(parent_path, dir_name).resolve()
    if create_dir:
        path.mkdir(exist_ok=True, parents=True)
    return path
