"""Launch the training experiment comparator for this framework."""

from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = SOURCE_DIR.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

from others.experiment_comparison import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(FRAMEWORK_DIR))
