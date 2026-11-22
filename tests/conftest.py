"""Shared pytest setup: a small fake Indian food dataset made of random images.

No downloads, no GPU, no open_clip. pytest loads this file before every test file.

Why fake data: the real dataset is large and not in the repo, so tests build a tiny look-alike in a
temporary folder. It copies the real Kaggle layout (class folders nested two levels deep) and adds two
deliberate problems, one exact duplicate and one corrupt file, so the cleaning code has something to find.
Any name defined here (CLASSES, PER_CLASS, ROOT, the fake_indian fixture) can be used by every test file.
A "fixture" is a function pytest runs for a test that lists its name as an argument.
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
    """Kaggle-like nested layout + one exact duplicate + one corrupt file.

    Returns the dataset root folder. Contents: 4 classes x 12 random-noise JPEGs = 48 good images, plus
    poha/dup.jpg (byte-identical copy of poha/000.jpg) and jalebi/broken.jpg (text, not an image), so 50
    files in total. tmp_path is a fresh temporary folder per test, so tests never affect each other.
    """
    # Same double-nested folder layout as the real Kaggle dataset.
    root = tmp_path / "indian"
    class_root = root / "Indian Food Images" / "Indian Food Images"
    rng = np.random.default_rng(0)  # fixed seed, so the "random" images are the same on every run
    # Write 12 random-noise JPEGs per class; one colour channel is fixed so classes look different.
    for c_i, c in enumerate(CLASSES):
        d = class_root / c
        d.mkdir(parents=True)
        for i in range(PER_CLASS):
            # Random pixels, height 48, width 64, 3 colour channels (noise images are almost never duplicates).
            arr = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            arr[..., c_i % 3] = 200  # class-specific tint
            Image.fromarray(arr).save(d / f"{i:03d}.jpg")
    # Add one exact duplicate and one file that is not a real image, so cleaning has work to do.
    shutil.copy(class_root / "poha" / "000.jpg", class_root / "poha" / "dup.jpg")
    # A file with a .jpg name but plain-text content: opening it as an image must fail.
    (class_root / "jalebi" / "broken.jpg").write_bytes(b"not an image")
    return root
