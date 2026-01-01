"""Kept for old commands: the code lives in nets/figures.py (av-world nets figures)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from nets import figures  # noqa: E402

if __name__ == "__main__":
    figures.main(sys.argv[1:])
