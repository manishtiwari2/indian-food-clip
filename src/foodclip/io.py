"""Small helpers: seeding, device, appending result rows, run metadata."""
from __future__ import annotations

import datetime as dt
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_meta() -> dict[str, str]:
    """Stored with every result row so each number can be traced to a run."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
            "python": platform.python_version(), "device": gpu}


def write_rows(rows: list[dict], path: Path, append: bool = False) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if append and path.exists():
        df = pd.concat([pd.read_csv(path, encoding="utf-8"), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df
