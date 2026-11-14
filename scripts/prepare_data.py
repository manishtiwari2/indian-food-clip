"""WP1: clean + split the Indian dataset, and (optionally) build the Food-101 subset.

  python scripts/prepare_data.py
  python scripts/prepare_data.py --food101 --download-food101
  python scripts/prepare_data.py --set paths.indian_raw="/kaggle/input/indian-food-images-dataset"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import (  # noqa: E402
    check_class_names, dedup, food101_subset, is_valid_image, list_images, load_class_names, stratified_split,
)


def prepare_indian(cfg: dict) -> None:
    out = resolve(cfg["paths"]["splits"])
    out.mkdir(parents=True, exist_ok=True)
    class_root, df = list_images(resolve(cfg["paths"]["indian_raw"]))
    print(f"found {len(df)} images in {df['label'].nunique()} classes under {class_root}")

    ok = [is_valid_image(class_root / r) for r in df["relpath"]]
    corrupt = df[[not v for v in ok]].assign(duplicate_of="", reason="corrupt")
    df = df[ok]
    kept, removed = dedup(df, class_root, cfg["split"]["dedup_max_distance"])
    removed = pd.concat([corrupt, removed], ignore_index=True)
    split = stratified_split(kept, cfg["split"]["test_per_class"], cfg["seed"])

    split.to_csv(out / "indian_split.csv", index=False, encoding="utf-8")
    removed.to_csv(out / "indian_removed.csv", index=False, encoding="utf-8")
    counts = split.groupby("split").size().to_dict()
    print(f"removed {len(removed)} (corrupt {len(corrupt)}, duplicates {len(removed) - len(corrupt)})")
    print(f"split: {counts}; min images/class {split.groupby('label').size().min()}")

    missing, extra = check_class_names(split["label"].unique(), load_class_names(resolve(cfg["paths"]["class_names"])))
    if missing or extra:
        print("WARNING class_names.csv does not match dataset folders:")
        print("  in dataset, not in csv:", missing)
        print("  in csv, not in dataset:", extra)
    else:
        print("class_names.csv matches all dataset folders")


def prepare_food101(cfg: dict, download: bool) -> None:
    root = resolve(cfg["paths"]["food101_root"])
    if download:
        from torchvision.datasets import Food101  # ~5 GB download

        Food101(str(root), split="test", download=True)
    _, df = food101_subset(root, cfg["food101"]["n_classes"], cfg["food101"]["per_class"], cfg["seed"])
    df["split"] = "test"
    out = resolve(cfg["paths"]["splits"]) / "food101_subset.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"food101 subset: {len(df)} images, {df['label'].nunique()} classes -> {out}")


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--food101", action="store_true", help="also build the Food-101 subset")
    p.add_argument("--download-food101", action="store_true", help="download Food-101 with torchvision first")
    p.add_argument("--skip-indian", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    if not args.skip_indian:
        prepare_indian(cfg)
    if args.food101 or args.download_food101:
        prepare_food101(cfg, args.download_food101)


if __name__ == "__main__":
    main()
