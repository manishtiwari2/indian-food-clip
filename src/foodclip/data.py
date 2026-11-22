"""Dataset listing, cleaning (corrupt + near-duplicate removal), splitting and loading.

Cleaning and splitting are run once by scripts/prepare_data.py; loading is used by every experiment:
E1 (CNN transforms and dataset), E2-E5 (load_dataset for features and labels), E7 (train/val split).
Report section: Dataset (cleaning, pHash dedup, decision D10, stratified split, Food-101 subset).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance

# File types counted as images, and the ImageNet colour mean/std used to normalise CNN inputs.
# IMAGE_EXTS: lower-case suffixes accepted as images (a set, so membership tests are O(1)).
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Per-channel (R, G, B) mean, shape (3,), float32 so it broadcasts against (H, W, 3) arrays without upcasting to float64.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# Per-channel (R, G, B) standard deviation, shape (3,), float32.
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_label(name: str) -> str:
    """Folder name in, clean label out ("Butter Chicken" -> "butter_chicken"), so names match everywhere.

    Why: folder names, class_names.csv and split CSVs must use one spelling or labels would not match.
    Used by: list_images, check_class_names, load_class_names."""
    # Strip surrounding whitespace, lower-case, then turn spaces and hyphens into underscores.
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _is_image(p: Path) -> bool:
    """Path in; True if it is a file with an image extension."""
    # is_file() is False for folders and missing paths; suffix is the last extension (".JPG" -> ".jpg" after lower()).
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def find_class_root(root: str | Path) -> Path:
    """Kaggle datasets often nest the class folders (e.g. `X/X/<class>/`). Walk down single
    sub-folders until we reach the folder that holds the class folders."""
    # Accept str or Path.
    root = Path(root)
    # Fail early with a clear message if the dataset path is wrong.
    if not root.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {root}")
    # Go one level down while the folder holds exactly one sub-folder and no images.
    while True:
        # iterdir() lists the direct children of root (not recursive); keep only the folders.
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        # True if any direct child is an image file.
        has_images = any(_is_image(p) for p in root.iterdir())
        # Exactly one sub-folder and no images means a wrapper folder: descend into it. Otherwise root holds the classes.
        if len(subdirs) == 1 and not has_images:
            root = subdirs[0]
        else:
            return root


def list_images(root: str | Path) -> tuple[Path, pd.DataFrame]:
    """Return (class_root, df[relpath, label]) — relpath is relative to class_root."""
    # Folder that directly contains the class folders.
    class_root = find_class_root(root)
    # Python list of dicts; one dict per image, turned into a DataFrame at the end.
    rows = []
    # One row per image: each class folder name becomes the label. Sorted so the order is always the same.
    for class_dir in sorted(d for d in class_root.iterdir() if d.is_dir()):
        # Label = cleaned folder name (same spelling as class_names.csv).
        label = normalize_label(class_dir.name)
        # rglob("*") yields every file and folder below class_dir recursively; sorted() makes the order stable.
        for p in sorted(class_dir.rglob("*")):  # rglob also finds images in sub-folders
            # Skip non-images (folders, text files, ...).
            if _is_image(p):
                # relative_to gives the path below class_root; as_posix() uses "/" so the CSV is the same on Windows and Linux.
                rows.append({"relpath": p.relative_to(class_root).as_posix(), "label": label})
    # An empty dataset is almost always a wrong path, so fail loudly.
    if not rows:
        raise ValueError(f"no images found under {class_root}")
    # DataFrame of shape (n_images, 2) with columns relpath and label.
    return class_root, pd.DataFrame(rows)


def is_valid_image(path: str | Path) -> bool:
    """Image path in; False if the file is broken or cannot be decoded, so corrupt files can be removed.

    Why: scraped data has broken files that would crash training later. Used by: scripts/prepare_data.py."""
    try:
        # verify() checks the file structure; it leaves the image unusable, so the file is opened again.
        # Image.open is lazy: it reads only the header. The with-block closes the file handle afterwards.
        with Image.open(path) as im:
            im.verify()
        # Fully decode the pixels as RGB; this catches files that pass verify() but are cut off.
        with Image.open(path) as im:
            # convert("RGB") forces decoding and gives 3 channels; load() makes sure all pixel data is read.
            im.convert("RGB").load()
        # Both checks passed.
        return True
    # Any error (truncated, unknown format, OS error) means the file is unusable.
    except Exception:
        return False


def phash_bits(paths: Sequence[str | Path]) -> np.ndarray:
    """64-bit perceptual hash per image as an (n, 64) uint8 bit matrix.

    pHash depends on the image's coarse structure, so resized or re-compressed copies of one photo get
    (almost) the same hash. That is why it finds near-duplicates, not only exact copies."""
    # Imported here so the rest of the module works without imagehash installed.
    import imagehash

    # One row of 64 bits (0/1) per image; the 8x8 hash is flattened into 64 values.
    bits = np.zeros((len(paths), 64), dtype=np.uint8)
    for i, p in enumerate(paths):
        # Open lazily; the with-block closes the file after hashing.
        with Image.open(p) as im:
            # phash() resizes to a small grey image and applies a DCT; .hash is an (8, 8) boolean array, flatten() makes it (64,),
            # astype(uint8) turns True/False into 1/0, and the row bits[i] (shape (64,)) is filled with it.
            bits[i] = imagehash.phash(im.convert("RGB")).hash.flatten().astype(np.uint8)
    # Shape (n, 64), dtype uint8.
    return bits


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bit matrices a (n, 64) and b (m, 64) in; (n, m) Hamming distances (number of different bits) out.

    Small distance = similar images. Used by: dedup."""
    # int32 so the matrix products below do not overflow.
    a = a.astype(np.int32)
    b = b.astype(np.int32)
    # Count positions where a=1,b=0 plus positions where a=0,b=1; done for all pairs with two matrix products.
    # a @ (1-b).T has shape (n, m): entry (i, k) counts bits where a_i=1 and b_k=0. (1-a) @ b.T counts the opposite case.
    return a @ (1 - b).T + (1 - a) @ b.T


def dedup(df: pd.DataFrame, class_root: Path, max_distance: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy near-duplicate removal over the whole dataset (keeps the first image of a group).
    If a duplicate appears in a *different* class, both copies are removed (label unknown).
    Returns (kept, removed); removed has `duplicate_of` and `reason` columns.

    Why: the dataset is web-scraped, so the same photo often appears more than once. A duplicate in both
    train and test would leak test images into training and make results look better than they are."""
    # Fixed order, then one pHash per image.
    # sort_values sorts by label then relpath; reset_index(drop=True) renumbers rows 0..n-1 so row numbers match rows of `bits`.
    df = df.sort_values(["label", "relpath"]).reset_index(drop=True)
    # (n, 64) bit matrix; row k belongs to df row k. class_root / relpath builds the absolute path.
    bits = phash_bits([class_root / r for r in df["relpath"]])
    # Number of images.
    n = len(df)
    removed_of: dict[int, int] = {}  # removed row j -> row i it duplicates
    # Compare images in chunks of 1024 rows against all images, to keep the distance matrix small in memory.
    # Chunk size 1024: reason not recorded in the repo.
    chunk = 1024
    for start in range(0, n, chunk):
        dist = hamming_matrix(bits[start:start + chunk], bits)  # (chunk, n)
        # Greedy: each image i that is still kept removes every later image j within max_distance bits.
        for off in range(dist.shape[0]):
            # Row number of this image in df (off is its row inside the chunk).
            i = start + off
            # An image already removed as someone's duplicate must not remove others.
            if i in removed_of:
                continue
            # dist[off] has shape (n,); np.nonzero(bool_array)[0] returns the indices where the condition is True (includes i itself, distance 0).
            for j in np.nonzero(dist[off] <= max_distance)[0]:
                if j > i and j not in removed_of:  # j > i: only later images, so the first copy is kept
                    # int(j) converts numpy int64 to a plain int dict key.
                    removed_of[int(j)] = i
    # Mark each removed image: same class as its kept copy, or a different class.
    reason = {j: "duplicate" if df.loc[i, "label"] == df.loc[j, "label"] else "cross_class_duplicate"
              for j, i in removed_of.items()}
    # D10: the same photo in two classes has an untrustworthy label, so drop the kept copy too.
    for j, i in list(removed_of.items()):
        if reason[j] == "cross_class_duplicate" and i not in removed_of:
            removed_of[i] = j
            reason[i] = "cross_class_duplicate"
    # Build the removed table (with what each row duplicates and why) and the kept table.
    # Row numbers of all removed images, ascending.
    rem_idx = sorted(removed_of)
    # Select those rows by label (index equals row number); copy() so adding columns does not touch df.
    removed = df.loc[rem_idx].copy()
    removed["duplicate_of"] = [df.loc[removed_of[j], "relpath"] for j in rem_idx]
    removed["reason"] = [reason[j] for j in rem_idx]
    # Drop the removed rows and renumber the rest.
    kept = df.drop(index=rem_idx).reset_index(drop=True)
    return kept, removed.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, test_per_class: int, seed: int) -> pd.DataFrame:
    """Add a `split` column (train/test). Small classes keep ~30% for test (at least 1).

    Stratified = the split is done per class, so every class has test images and the test set is
    balanced (the same number per class for all classes that are large enough)."""
    # Seeded numpy random generator, so the shuffles are reproducible.
    rng = np.random.default_rng(seed)
    # List of per-class DataFrames, concatenated at the end.
    out = []
    # Sort first so the same seed always gives the same split, then handle one class at a time.
    for label, group in df.sort_values(["label", "relpath"]).groupby("label", sort=True):
        # Shuffle the rows of this class.
        # rng.permutation(k) returns a shuffled array of 0..k-1; iloc picks the rows in that order; copy() avoids a pandas warning when adding a column.
        group = group.iloc[rng.permutation(len(group))].copy()
        # Number of images in this class.
        n = len(group)
        # Big class (at least 2x test_per_class): fixed test size. Small class: about 30%, at least 1.
        n_test = test_per_class if n >= 2 * test_per_class else max(1, round(0.3 * n))
        # The first n_test shuffled rows are test, the rest are train.
        group["split"] = ["test"] * n_test + ["train"] * (n - n_test)
        out.append(group)
    # Join all classes back into one table in a fixed order.
    # concat stacks the class tables vertically; sort_values/reset_index restore a fixed order and row numbers 0..n-1.
    return pd.concat(out).sort_values(["label", "relpath"]).reset_index(drop=True)


def food101_subset(root: str | Path, n_classes: int, per_class: int, seed: int) -> tuple[Path, pd.DataFrame]:
    """Pick n_classes random Food-101 classes and per_class random *test* images each.
    Returns (images_dir, df[relpath, label])."""
    # Accept both the parent folder and the food-101 folder itself.
    # Usual layout: <root>/food-101/.
    base = Path(root) / "food-101"
    if not base.is_dir():
        base = Path(root)  # allow pointing directly at the food-101 folder
    # Read the official class list and the official test list (lines look like "apple_pie/1011328").
    # read_text() gives the whole file as one string; split() cuts on whitespace into one entry per line; sorted() fixes the order.
    classes = sorted((base / "meta" / "classes.txt").read_text().split())
    # Same for the official test list: a list of strings like "apple_pie/1011328".
    test_lines = (base / "meta" / "test.txt").read_text().split()
    # Group the test images by class (the class is the part before "/").
    by_class: dict[str, list[str]] = {}
    for line in test_lines:
        # Text before the first "/" is the class name.
        cls = line.split("/")[0]
        # setdefault creates an empty list the first time a class is seen, then append adds the entry.
        by_class.setdefault(cls, []).append(line)
    # Seeded random choice of classes, then per_class seeded random test images from each class.
    # Python's own seeded generator (not numpy).
    rng = random.Random(seed)
    # sample draws n_classes distinct classes without replacement; sorted() keeps the output order fixed.
    chosen = sorted(rng.sample(classes, n_classes))
    # One dict per selected image.
    rows = []
    for cls in chosen:
        for item in sorted(rng.sample(sorted(by_class[cls]), per_class)):  # sorted input = same picks per seed
            # The list files have no extension; Food-101 images are .jpg.
            rows.append({"relpath": f"{item}.jpg", "label": cls})
    # DataFrame of shape (n_classes * per_class, 2).
    return base / "images", pd.DataFrame(rows)


def check_class_names(labels: Sequence[str], names: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (labels missing from class_names.csv, csv rows not present in the dataset)."""
    # Set differences in both directions show any mismatch between the CSV and the dataset folders.
    csv_folders = {normalize_label(f) for f in names["folder"]}
    data_labels = set(labels)
    return sorted(data_labels - csv_folders), sorted(csv_folders - data_labels)


def load_class_names(path: str | Path) -> pd.DataFrame:
    """Path of class_names.csv in; table with folder, english, hindi_roman, hindi_devanagari out.

    Sorted by folder, so row i matches class index i (classes are also sorted in load_dataset)."""
    # UTF-8 is needed to read the Devanagari names correctly.
    df = pd.read_csv(path, encoding="utf-8")
    # map applies normalize_label to every value of the folder column.
    df["folder"] = df["folder"].map(normalize_label)
    # Sort by folder and renumber rows 0..n-1.
    return df.sort_values("folder").reset_index(drop=True)


class SimpleTransform:
    """Torch-only image transform for the CNN (no torchvision needed).

    Train mode uses random crop, flip and optional colour jitter (augmentation: the model sees slightly
    different images each epoch, which reduces overfitting). Test mode is deterministic (resize + center
    crop). Both end with ImageNet normalisation. Used by: scripts/train_cnn.py through ImageDataset."""

    def __init__(self, size: int, train: bool, color_jitter: float = 0.0):
        """size = output side in pixels; train=True adds random augmentation, False gives a fixed center crop."""
        # color_jitter = max relative change of brightness and contrast (0 = off, e.g. 0.3 -> factor in [0.7, 1.3]).
        self.size = size
        self.train = train
        self.color_jitter = color_jitter

    def __call__(self, im: Image.Image) -> torch.Tensor:
        """PIL image in; normalised (3, size, size) float tensor out."""
        # Force 3 channels (grey, palette or RGBA images become RGB).
        im = im.convert("RGB")
        # PIL size is (width, height).
        w, h = im.size
        if self.train:
            # Random square crop covering 60-100% of the short side's area, resized to size x size.
            # Random area share between 0.6 and 1.0; the range 0.6-1.0: reason not recorded in the repo.
            scale = random.uniform(0.6, 1.0)
            side = int(min(w, h) * scale ** 0.5)  # square root: scale is an area share, side is a length
            # Random top-left corner so the square fits inside the image (randint includes both ends).
            x = random.randint(0, w - side)
            y = random.randint(0, h - side)
            # crop takes (left, upper, right, lower); resize to a fixed size x size with bilinear interpolation so batches stack.
            im = im.crop((x, y, x + side, y + side)).resize((self.size, self.size), Image.BILINEAR)
            # Flip left-right half of the time (a dish looks the same mirrored).
            if random.random() < 0.5:
                # transpose mirrors the image horizontally.
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            # Optional random brightness and contrast change.
            if self.color_jitter > 0:  # only draws random numbers when on, so jitter=0 matches earlier runs exactly
                j = self.color_jitter
                # ImageEnhance.Brightness(im).enhance(f) multiplies brightness by f (1 = unchanged); Contrast does the same for contrast.
                im = ImageEnhance.Brightness(im).enhance(random.uniform(1 - j, 1 + j))
                im = ImageEnhance.Contrast(im).enhance(random.uniform(1 - j, 1 + j))
        else:
            # Test: resize so the short side is 1.14 x size (keeps aspect ratio), then crop the center.
            # Factor 1.14: reason not recorded in the repo.
            short = int(self.size * 1.14)
            # Scale factor that makes the short side equal `short`.
            r = short / min(w, h)
            # New (width, height); max(short, ...) guards against rounding below `short`.
            im = im.resize((max(short, round(w * r)), max(short, round(h * r))), Image.BILINEAR)
            # Size after resizing.
            w, h = im.size
            # Top-left corner of the centred size x size window (integer division).
            x, y = (w - self.size) // 2, (h - self.size) // 2
            im = im.crop((x, y, x + self.size, y + self.size))
        # np.asarray gives a (size, size, 3) array; /255 scales to [0, 1]; subtracting the (3,) mean and dividing by the (3,) std
        # broadcast over the last axis (per channel).
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        # (H, W, 3) -> (3, H, W), the channel-first layout PyTorch expects. copy() makes the memory contiguous
        # (transpose is only a view); torch.from_numpy shares memory with that array and gives a float32 tensor.
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class ImageDataset(torch.utils.data.Dataset):
    """PyTorch dataset: image paths + int labels + a transform; item i = (transformed image, label).

    A DataLoader calls __len__ and __getitem__ to build batches. Used by: train_cnn.py, finetune_clip.py
    and features.encode_images (with CLIP's preprocess as the transform)."""

    def __init__(self, paths: Sequence[str | Path], labels: Sequence[int], transform: Callable):
        """Store paths, labels and the transform (CNN transform or CLIP preprocess)."""
        # list() copies into plain lists so later changes to the inputs do not matter.
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        """Number of images."""
        # One item per path.
        return len(self.paths)

    def __getitem__(self, i: int):
        """Index in; (image tensor, label) out. The image is read from disk only when needed."""
        # Open lazily; the with-block closes the file. convert("RGB") decodes it with 3 channels, then the transform makes a tensor.
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        # DataLoader stacks the tensors into a batch of shape (batch, 3, H, W) and the labels into (batch,).
        return x, self.labels[i]


# ---------------------------------------------------------------- loading committed splits
# Committed split CSV for each dataset name (inside paths.splits).
SPLIT_FILES = {"indian": "indian_split.csv", "food101": "food101_subset.csv"}


def load_dataset(cfg: dict, name: str) -> tuple[list[Path], pd.DataFrame, list[str]]:
    """Return (absolute image paths, split df, sorted class list) for 'indian' or 'food101'.
    Class index = position in the sorted class list (same everywhere).
    Used by every experiment script; reads the committed CSV in splits/ so everyone gets the same split."""
    # Imported inside the function (only needed here).
    # config.resolve: relative paths are taken from the repo root, absolute paths (e.g. Kaggle) pass through unchanged.
    from .config import resolve

    # Read the committed split CSV (relpath, label, and split for the Indian set).
    df = pd.read_csv(resolve(cfg["paths"]["splits"]) / SPLIT_FILES[name], encoding="utf-8")
    # Find the folder the relative paths start from.
    if name == "indian":
        root = find_class_root(resolve(cfg["paths"]["indian_raw"]))
    else:
        base = resolve(cfg["paths"]["food101_root"])
        root = (base / "food-101" / "images") if (base / "food-101").is_dir() else base / "images"
    # Integer label y = position of the label in the sorted class list.
    # unique() gives the distinct labels; sorted() fixes the order, so the class index is the same everywhere.
    classes = sorted(df["label"].unique())
    # map looks each label up in a {label: index} dict and stores the int in the new column y.
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
    # Absolute paths (one per df row, same order), the DataFrame (now with column y) and the class list.
    return [root / r for r in df["relpath"]], df, classes


def train_val_split(train_idx: np.ndarray, y: np.ndarray, val_fraction: float, n_classes: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a stratified validation part of the training split (used to pick the best epoch).

    Input: training row numbers, labels y, validation share, class count, seed. Output: (train_idx,
    val_idx). Why: choosing the best epoch on the test set would be cheating, so a slice of the training
    data is used instead. Used by: train_cnn.py and finetune_clip.py."""
    # Imported here so the module works without scikit-learn when this function is not called.
    from sklearn.model_selection import train_test_split

    # Validation size; stratify by class only if there is room for at least one image per class.
    n_val = int(round(len(train_idx) * val_fraction))
    # y[train_idx] picks the labels of the training rows (fancy indexing); stratify keeps class proportions equal in both parts.
    stratify = y[train_idx] if n_val >= n_classes else None
    # Returns (train indices, validation indices); the test set is never touched here.
    # test_size=n_val is the validation count; random_state=seed makes the shuffle reproducible. Both outputs are 1-D arrays of row numbers.
    return train_test_split(train_idx, test_size=n_val, random_state=seed, stratify=stratify)
