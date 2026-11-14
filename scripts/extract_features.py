"""WP3: encode every image once per CLIP model and cache to features/<dataset>__<model>.npz.

  python scripts/extract_features.py
  python scripts/extract_features.py --models ViT-B-32 --datasets indian --force
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import SPLIT_FILES, load_dataset  # noqa: E402
from foodclip.features import encode_images, feature_path, load_clip, save_features  # noqa: E402
from foodclip.io import get_device  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="*", help="model names to run (default: all in config)")
    p.add_argument("--datasets", nargs="*", default=["indian", "food101"])
    p.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    datasets = [d for d in args.datasets if (resolve(cfg["paths"]["splits"]) / SPLIT_FILES[d]).exists()]
    for d in set(args.datasets) - set(datasets):
        print(f"skip {d}: no split file (run prepare_data.py first)")

    for m in cfg["clip"]["models"]:
        if args.models and m["name"] not in args.models:
            continue
        todo = [d for d in datasets if args.force or not feature_path(cfg, d, m).exists()]
        if not todo:
            print(f"{m['name']}: cached, skipping")
            continue
        model, preprocess, _ = load_clip(m, device)
        for d in todo:
            paths, df, _ = load_dataset(cfg, d)
            feats = encode_images(model, preprocess, paths, device, cfg["clip"]["batch_size"],
                                  cfg["clip"]["num_workers"])
            save_features(feature_path(cfg, d, m), feats, df["relpath"].tolist())
            print(f"{m['name']} {d}: {feats.shape} -> {feature_path(cfg, d, m)}")
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
