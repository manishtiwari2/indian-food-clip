"""WP1: clean + split the Indian dataset, and (optionally) build the Food-101 subset.

What it does: lists the scraped Indian images, drops corrupt files and near-duplicates (perceptual hash),
then makes a stratified train/test split per class. With --food101 it also picks a fixed random
subset of the Food-101 test set as the Western comparison set.
Experiment: none directly; it builds the data that every experiment (E1-E7) uses.
Writes: splits/indian_split.csv, splits/indian_removed.csv, splits/food101_subset.csv.
Report section: "Dataset".

Pipeline position: this is the FIRST step. It reads raw images from disk (paths.indian_raw in the config)
and writes small CSV files. Every later script (train_cnn, extract_features, run_zeroshot, run_fewshot)
reads those CSVs through foodclip.data.load_dataset, so all experiments see the same images and the
same train/test split.

Why dedup and a stratified split: the dataset is scraped from Google, so the same picture can appear
twice. If one copy lands in train and one in test, the test score is inflated. Stratified means each
class is split on its own, so every class has test images.

  python scripts/prepare_data.py
  python scripts/prepare_data.py --food101 --download-food101
  python scripts/prepare_data.py --set paths.indian_raw="/kaggle/input/indian-food-images-dataset"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the library in src/ importable without installing it.
# __file__ is this script; parents[1] is the repo root; we add <repo>/src to Python's import search path
# so that "import foodclip" works even though the package is not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# base_parser: argparse parser with the shared --config and --set options.
# config_from_args: loads configs/default.yaml and applies any --set overrides.
# resolve: turns a path from the config into an absolute path relative to the repo root.
from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
# Data helpers (all in src/foodclip/data.py):
#   list_images: walk the class folders and return (class_root, DataFrame[relpath, label]).
#   is_valid_image: True if PIL can open and decode the file.
#   dedup: drop near-duplicate images using a perceptual hash (similar-looking images get similar hashes).
#   stratified_split: per-class random train/test split with a fixed seed.
#   food101_subset: pick a fixed random subset of Food-101 test images.
#   load_class_names / check_class_names: read data/class_names.csv and compare it to the folder names.
from foodclip.data import (  # noqa: E402
    check_class_names, dedup, food101_subset, is_valid_image, list_images, load_class_names, stratified_split,
)


def prepare_indian(cfg: dict) -> None:
    """Clean, deduplicate and split the Indian dataset, then save the split and the removed-image list.

    Input: the raw image folders (one sub-folder per dish) at cfg["paths"]["indian_raw"].
    Output: splits/indian_split.csv (kept images + train/test column) and splits/indian_removed.csv
    (dropped images; columns relpath, label, duplicate_of, reason, where reason is corrupt, duplicate
    or cross_class_duplicate). Both are small text
    files that are committed to git; the images themselves are never committed.
    """
    # Create the output folder and list every image with its class label.
    # resolve() makes the config path absolute; mkdir(exist_ok) avoids an error if it already exists.
    out = resolve(cfg["paths"]["splits"])
    out.mkdir(parents=True, exist_ok=True)
    class_root, df = list_images(resolve(cfg["paths"]["indian_raw"]))  # df columns: relpath, label
    print(f"found {len(df)} images in {df['label'].nunique()} classes under {class_root}")

    # Drop images that cannot be opened, and remember them with the reason "corrupt".
    # ok is a list of True/False, one per image, in the same order as df.
    ok = [is_valid_image(class_root / r) for r in df["relpath"]]
    # df[[not v ...]] selects the rows where the image was NOT valid; assign() adds the two extra columns
    # so this table has the same columns as the table dedup() returns and the two can be stacked.
    corrupt = df[[not v for v in ok]].assign(duplicate_of="", reason="corrupt")  # same columns as dedup output
    # Keep only the valid rows (boolean list used as a mask).
    df = df[ok]
    # Remove near-duplicate images (small perceptual-hash distance), so test images are not copies of train.
    # dedup returns two tables: kept images and removed images (with which image each one duplicated).
    kept, removed = dedup(df, class_root, cfg["split"]["dedup_max_distance"])
    # Stack the corrupt rows on top of the duplicate rows into one "removed" table.
    removed = pd.concat([corrupt, removed], ignore_index=True)
    # Split each class into train and test with the fixed seed, so the split is reproducible.
    # test_per_class test images per class; classes with fewer than 2 x test_per_class images get about 30%
    # (at least 1) for test (D10). The rest go to train.
    split = stratified_split(kept, cfg["split"]["test_per_class"], cfg["seed"])

    # Save the split and the list of removed images, then print a short summary.
    # utf-8 and index=False keep the CSV clean and readable on any machine.
    split.to_csv(out / "indian_split.csv", index=False, encoding="utf-8")
    removed.to_csv(out / "indian_removed.csv", index=False, encoding="utf-8")
    counts = split.groupby("split").size().to_dict()  # e.g. {"test": n, "train": n}
    print(f"removed {len(removed)} (corrupt {len(corrupt)}, duplicates {len(removed) - len(corrupt)})")
    print(f"split: {counts}; min images/class {split.groupby('label').size().min()}")

    # Check that the dataset folders and data/class_names.csv list the same classes (needed for prompts).
    # The zero-shot step looks up each folder name in class_names.csv to build text prompts. A mismatch makes
    # it fall back to the folder name for that class (with a WARNING), which usually gives worse prompts.
    # missing = folders without a csv row; extra = csv rows without a folder.
    missing, extra = check_class_names(split["label"].unique(), load_class_names(resolve(cfg["paths"]["class_names"])))
    if missing or extra:
        print("WARNING class_names.csv does not match dataset folders:")
        print("  in dataset, not in csv:", missing)
        print("  in csv, not in dataset:", extra)
    else:
        print("class_names.csv matches all dataset folders")


def prepare_food101(cfg: dict, download: bool) -> None:
    """Build the Western comparison set: a fixed random subset of Food-101 test images.

    Used only for the Indian-vs-Western gap (E4/E6): the same CLIP model is run zero-shot on these
    images. Food-101 is never used for training. Writes splits/food101_subset.csv.
    """
    root = resolve(cfg["paths"]["food101_root"])
    # Optionally download Food-101 with torchvision (imported here so it is only needed for this step).
    if download:
        from torchvision.datasets import Food101  # ~5 GB download

        # Creating the dataset object with download=True fetches and unpacks the files into root.
        Food101(str(root), split="test", download=True)
    # Pick n_classes classes and per_class test images each, using the config seed.
    # The first return value (the class folder root) is not needed here, hence "_".
    _, df = food101_subset(root, cfg["food101"]["n_classes"], cfg["food101"]["per_class"], cfg["seed"])
    df["split"] = "test"  # Food-101 is only used for testing (zero-shot), never for training
    # Save the subset list next to the Indian split.
    out = resolve(cfg["paths"]["splits"]) / "food101_subset.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"food101 subset: {len(df)} images, {df['label'].nunique()} classes -> {out}")


def main() -> None:
    """Read command-line options and config, then prepare the Indian and/or Food-101 data."""
    # Command-line options on top of the shared --config / --set options.
    # base_parser(__doc__) reuses this file's top docstring as the --help text.
    p = base_parser(__doc__)
    # action="store_true" makes these flags: False by default, True when written on the command line.
    p.add_argument("--food101", action="store_true", help="also build the Food-101 subset")
    p.add_argument("--download-food101", action="store_true", help="download Food-101 with torchvision first")
    p.add_argument("--skip-indian", action="store_true")
    args = p.parse_args()
    # Load the YAML config and apply any --set key=value overrides into one nested dict.
    cfg = config_from_args(args)
    # Run the Indian step unless skipped; run the Food-101 step only when asked.
    if not args.skip_indian:
        prepare_indian(cfg)
    if args.food101 or args.download_food101:
        prepare_food101(cfg, args.download_food101)


# Run main() only when the file is executed as a script, not when it is imported (e.g. by the tests).
if __name__ == "__main__":
    main()
