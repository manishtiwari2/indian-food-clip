"""WP3: encode every image once per CLIP model and cache to features/<dataset>__<model>.npz.

What it does: runs each CLIP image encoder from the config over all Indian and Food-101 images and saves
the L2-normalised image embeddings. Later steps only load this cache, so the expensive image encoding
is done once and E2-E5 become cheap to run and re-run.
Experiment: preparation step for E2-E4 (run_zeroshot.py) and E5 (run_fewshot.py).
Writes: features/<dataset>__<model>__<pretrained>.npz (gitignored, no result CSV or figure).
Report section: Methods "CLIP zero-shot and prompt engineering" and "Few-shot linear probe" (shared features).

  python scripts/extract_features.py
  python scripts/extract_features.py --models ViT-B-32 --datasets indian --force
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import SPLIT_FILES, load_dataset  # noqa: E402
from foodclip.features import encode_images, feature_path, load_clip, save_features  # noqa: E402
from foodclip.io import get_device  # noqa: E402


def main() -> None:
    """Compute and cache CLIP image features for every chosen model and dataset."""
    # Read options: which models, which datasets, and whether to overwrite an existing cache.
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="*", help="model names to run (default: all in config)")
    p.add_argument("--datasets", nargs="*", default=["indian", "food101"])
    p.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    # Keep only datasets whose split CSV exists; tell the user about the others.
    datasets = [d for d in args.datasets if (resolve(cfg["paths"]["splits"]) / SPLIT_FILES[d]).exists()]
    for d in set(args.datasets) - set(datasets):
        print(f"skip {d}: no split file (run prepare_data.py first)")

    # Loop over the CLIP models listed in the config.
    for m in cfg["clip"]["models"]:
        # Skip models not asked for on the command line.
        if args.models and m["name"] not in args.models:
            continue
        # Datasets still to encode for this model (all of them with --force).
        todo = [d for d in datasets if args.force or not feature_path(cfg, d, m).exists()]
        if not todo:
            print(f"{m['name']}: cached, skipping")
            continue
        # Load the model once, then encode each dataset and save features in split-CSV order.
        model, preprocess, _ = load_clip(m, device)
        for d in todo:
            paths, df, _ = load_dataset(cfg, d)
            feats = encode_images(model, preprocess, paths, device, cfg["clip"]["batch_size"],
                                  cfg["clip"]["num_workers"])  # shape (n_images, embedding_dim)
            # Store relpaths too, so a later load can check the order still matches the split.
            save_features(feature_path(cfg, d, m), feats, df["relpath"].tolist())
            print(f"{m['name']} {d}: {feats.shape} -> {feature_path(cfg, d, m)}")
        # Free GPU memory before loading the next model.
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
