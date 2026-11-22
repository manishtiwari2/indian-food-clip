"""Small helpers: seeding, device, appending result rows, run metadata.

Used by all experiments E1-E7 (every script that trains a model or writes results/*.csv).
Report: supports reproducibility in every Methods/Results section (seeds and run info per result row).
"""
from __future__ import annotations

import datetime as dt
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Seed in, nothing out: seeds Python, NumPy and PyTorch (CPU + GPU) so runs can be repeated.

    Why: weight start values, shuffling and augmentation are random. With the same seed they repeat,
    so the same code gives (nearly) the same numbers. GPU kernels can still differ slightly.
    Used by: train_cnn.py, finetune_clip.py and other scripts, with the seed taken from the config."""
    # Each library has its own random generator, so each one is seeded.
    # Python's built-in generator.
    random.seed(seed)
    # NumPy's global generator.
    np.random.seed(seed)
    # PyTorch's CPU generator (also seeds the current CUDA device in recent versions).
    torch.manual_seed(seed)
    # PyTorch's generators on every GPU; separate from the CPU one. Does nothing without CUDA.
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the GPU device if one is available, else the CPU.

    Why: the same code then runs on a laptop (CPU) and on Kaggle (GPU) with no change.
    Used by: scripts that train a CNN or run CLIP."""
    # torch.cuda.is_available() is True when a CUDA GPU and driver are usable.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_meta() -> dict[str, str]:
    """Stored with every result row so each number can be traced to a run.

    Returns a dict with run time, torch version, Python version and device name. Why: if two results
    differ, these fields help find out which environment produced each. Used by: result-writing scripts."""
    # GPU name (or "cpu"), plus time, torch version and Python version of this run.
    # get_device_name(0) is the name of GPU number 0.
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    # datetime.now().isoformat(timespec="seconds") gives local time like 2026-01-31T12:00:00 (no fraction).
    return {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
            "python": platform.python_version(), "device": gpu}


def write_rows(rows: list[dict], path: Path, append: bool = False) -> pd.DataFrame:
    """Result rows (list of dicts) in, CSV file written and the full table returned.

    With append=True the new rows are added under the rows already in the file."""
    # Create the folder (e.g. results/) if it does not exist yet.
    # parents=True also creates missing parent folders; exist_ok=True means no error if it already exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    # One dict = one row; dict keys become columns (a key missing from a row becomes NaN).
    df = pd.DataFrame(rows)
    # Append mode: read the old file and put the new rows after it.
    # pd.concat stacks the tables; ignore_index=True renumbers rows 0..n-1.
    if append and path.exists():
        df = pd.concat([pd.read_csv(path, encoding="utf-8"), df], ignore_index=True)
    # Write the CSV (UTF-8 so Devanagari text is stored correctly); index=False leaves out the row-number column.
    df.to_csv(path, index=False, encoding="utf-8")
    # Return the full table that was written (old + new rows when appending).
    return df
