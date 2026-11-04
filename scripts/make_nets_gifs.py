"""Kept for old commands: the code lives in nets/gifs.py (av-world nets gifs)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from nets import gifs  # noqa: E402

if __name__ == "__main__":
    gifs.main(sys.argv[1:])
