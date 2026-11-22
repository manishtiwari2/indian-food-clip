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
    """Seed in, nothing out: seeds Python, NumPy and PyTorch (CPU + GPU) so runs can be repeated."""
    # Each library has its own random generator, so each one is seeded.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the GPU device if one is available, else the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_meta() -> dict[str, str]:
    """Stored with every result row so each number can be traced to a run."""
    # GPU name (or "cpu"), plus time, torch version and Python version of this run.
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
            "python": platform.python_version(), "device": gpu}


def write_rows(rows: list[dict], path: Path, append: bool = False) -> pd.DataFrame:
    """Result rows (list of dicts) in, CSV file written and the full table returned.

    With append=True the new rows are added under the rows already in the file."""
    # Create the folder (e.g. results/) if it does not exist yet.
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Append mode: read the old file and put the new rows after it.
    if append and path.exists():
        df = pd.concat([pd.read_csv(path, encoding="utf-8"), df], ignore_index=True)
    # Write the CSV (UTF-8 so Devanagari text is stored correctly).
    df.to_csv(path, index=False, encoding="utf-8")
    return df
