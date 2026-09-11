import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("SATQUERY_BASE_DIR", "./data/working"))
SCRATCH_DIR = Path(os.environ.get("SATQUERY_SCRATCH_DIR", "./data/tmp"))
INPUT_DIR = Path(os.environ.get("SATQUERY_INPUT_DIR", "./data/input"))

BASE_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
