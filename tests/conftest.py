"""Shared pytest setup: a small fake Indian food dataset made of random images.

No downloads, no GPU, no open_clip. pytest loads this file before every test file.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Repo root; add src/ to the import path so tests can `import foodclip` without installing it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Four real class names from data/class_names.csv, with 12 fake images each.
CLASSES = ["aloo_gobi", "jalebi", "poha", "rasgulla"]
PER_CLASS = 12


@pytest.fixture()
def fake_indian(tmp_path: Path) -> Path:
    """Kaggle-like nested layout + one exact duplicate + one corrupt file."""
    # Same double-nested folder layout as the real Kaggle dataset.
    root = tmp_path / "indian"
    class_root = root / "Indian Food Images" / "Indian Food Images"
    rng = np.random.default_rng(0)
    # Write 12 random-noise JPEGs per class; one colour channel is fixed so classes look different.
    for c_i, c in enumerate(CLASSES):
        d = class_root / c
        d.mkdir(parents=True)
        for i in range(PER_CLASS):
            arr = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            arr[..., c_i % 3] = 200  # class-specific tint
            Image.fromarray(arr).save(d / f"{i:03d}.jpg")
    # Add one exact duplicate and one file that is not a real image, so cleaning has work to do.
    shutil.copy(class_root / "poha" / "000.jpg", class_root / "poha" / "dup.jpg")
    (class_root / "jalebi" / "broken.jpg").write_bytes(b"not an image")
    return root
