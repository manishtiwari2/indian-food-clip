"""Dataset listing, cleaning (corrupt + near-duplicate removal), splitting and loading."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_label(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def find_class_root(root: str | Path) -> Path:
    """Kaggle datasets often nest the class folders (e.g. `X/X/<class>/`). Walk down single
    sub-folders until we reach the folder that holds the class folders."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {root}")
    while True:
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        has_images = any(_is_image(p) for p in root.iterdir())
        if len(subdirs) == 1 and not has_images:
            root = subdirs[0]
        else:
            return root


def list_images(root: str | Path) -> tuple[Path, pd.DataFrame]:
    """Return (class_root, df[relpath, label]) — relpath is relative to class_root."""
    class_root = find_class_root(root)
    rows = []
    for class_dir in sorted(d for d in class_root.iterdir() if d.is_dir()):
        label = normalize_label(class_dir.name)
        for p in sorted(class_dir.rglob("*")):
            if _is_image(p):
                rows.append({"relpath": p.relative_to(class_root).as_posix(), "label": label})
    if not rows:
        raise ValueError(f"no images found under {class_root}")
    return class_root, pd.DataFrame(rows)


def is_valid_image(path: str | Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def phash_bits(paths: Sequence[str | Path]) -> np.ndarray:
    """64-bit perceptual hash per image as an (n, 64) uint8 bit matrix."""
    import imagehash

    bits = np.zeros((len(paths), 64), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            bits[i] = imagehash.phash(im.convert("RGB")).hash.flatten().astype(np.uint8)
    return bits


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.int32)
    b = b.astype(np.int32)
    return a @ (1 - b).T + (1 - a) @ b.T


def dedup(df: pd.DataFrame, class_root: Path, max_distance: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy near-duplicate removal over the whole dataset (keeps the first image of a group).
    If a duplicate appears in a *different* class, both copies are removed (label unknown).
    Returns (kept, removed); removed has `duplicate_of` and `reason` columns."""
    df = df.sort_values(["label", "relpath"]).reset_index(drop=True)
    bits = phash_bits([class_root / r for r in df["relpath"]])
    n = len(df)
    removed_of: dict[int, int] = {}
    chunk = 1024
    for start in range(0, n, chunk):
        dist = hamming_matrix(bits[start:start + chunk], bits)
        for off in range(dist.shape[0]):
            i = start + off
            if i in removed_of:
                continue
            for j in np.nonzero(dist[off] <= max_distance)[0]:
                if j > i and j not in removed_of:
                    removed_of[int(j)] = i
    reason = {j: "duplicate" if df.loc[i, "label"] == df.loc[j, "label"] else "cross_class_duplicate"
              for j, i in removed_of.items()}
    # D10: the same photo in two classes has an untrustworthy label, so drop the kept copy too.
    for j, i in list(removed_of.items()):
        if reason[j] == "cross_class_duplicate" and i not in removed_of:
            removed_of[i] = j
            reason[i] = "cross_class_duplicate"
    rem_idx = sorted(removed_of)
    removed = df.loc[rem_idx].copy()
    removed["duplicate_of"] = [df.loc[removed_of[j], "relpath"] for j in rem_idx]
    removed["reason"] = [reason[j] for j in rem_idx]
    kept = df.drop(index=rem_idx).reset_index(drop=True)
    return kept, removed.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, test_per_class: int, seed: int) -> pd.DataFrame:
    """Add a `split` column (train/test). Small classes keep ~30% for test (at least 1)."""
    rng = np.random.default_rng(seed)
    out = []
    for label, group in df.sort_values(["label", "relpath"]).groupby("label", sort=True):
        group = group.iloc[rng.permutation(len(group))].copy()
        n = len(group)
        n_test = test_per_class if n >= 2 * test_per_class else max(1, round(0.3 * n))
        group["split"] = ["test"] * n_test + ["train"] * (n - n_test)
        out.append(group)
    return pd.concat(out).sort_values(["label", "relpath"]).reset_index(drop=True)


def food101_subset(root: str | Path, n_classes: int, per_class: int, seed: int) -> tuple[Path, pd.DataFrame]:
    """Pick n_classes random Food-101 classes and per_class random *test* images each.
    Returns (images_dir, df[relpath, label])."""
    base = Path(root) / "food-101"
    if not base.is_dir():
        base = Path(root)  # allow pointing directly at the food-101 folder
    classes = sorted((base / "meta" / "classes.txt").read_text().split())
    test_lines = (base / "meta" / "test.txt").read_text().split()
    by_class: dict[str, list[str]] = {}
    for line in test_lines:
        cls = line.split("/")[0]
        by_class.setdefault(cls, []).append(line)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(classes, n_classes))
    rows = []
    for cls in chosen:
        for item in sorted(rng.sample(sorted(by_class[cls]), per_class)):
            rows.append({"relpath": f"{item}.jpg", "label": cls})
    return base / "images", pd.DataFrame(rows)


def check_class_names(labels: Sequence[str], names: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (labels missing from class_names.csv, csv rows not present in the dataset)."""
    csv_folders = {normalize_label(f) for f in names["folder"]}
    data_labels = set(labels)
    return sorted(data_labels - csv_folders), sorted(csv_folders - data_labels)


def load_class_names(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["folder"] = df["folder"].map(normalize_label)
    return df.sort_values("folder").reset_index(drop=True)


class SimpleTransform:
    """Torch-only image transform for the CNN (no torchvision needed)."""

    def __init__(self, size: int, train: bool):
        self.size = size
        self.train = train

    def __call__(self, im: Image.Image) -> torch.Tensor:
        im = im.convert("RGB")
        w, h = im.size
        if self.train:
            scale = random.uniform(0.6, 1.0)
            side = int(min(w, h) * scale ** 0.5)
            x = random.randint(0, w - side)
            y = random.randint(0, h - side)
            im = im.crop((x, y, x + side, y + side)).resize((self.size, self.size), Image.BILINEAR)
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            short = int(self.size * 1.14)
            r = short / min(w, h)
            im = im.resize((max(short, round(w * r)), max(short, round(h * r))), Image.BILINEAR)
            w, h = im.size
            x, y = (w - self.size) // 2, (h - self.size) // 2
            im = im.crop((x, y, x + self.size, y + self.size))
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[str | Path], labels: Sequence[int], transform: Callable):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.labels[i]


# ---------------------------------------------------------------- loading committed splits
SPLIT_FILES = {"indian": "indian_split.csv", "food101": "food101_subset.csv"}


def load_dataset(cfg: dict, name: str) -> tuple[list[Path], pd.DataFrame, list[str]]:
    """Return (absolute image paths, split df, sorted class list) for 'indian' or 'food101'.
    Class index = position in the sorted class list (same everywhere)."""
    from .config import resolve

    df = pd.read_csv(resolve(cfg["paths"]["splits"]) / SPLIT_FILES[name], encoding="utf-8")
    if name == "indian":
        root = find_class_root(resolve(cfg["paths"]["indian_raw"]))
    else:
        base = resolve(cfg["paths"]["food101_root"])
        root = (base / "food-101" / "images") if (base / "food-101").is_dir() else base / "images"
    classes = sorted(df["label"].unique())
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
    return [root / r for r in df["relpath"]], df, classes
