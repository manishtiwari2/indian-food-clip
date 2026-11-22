"""WP3: encode every image once per CLIP model and cache to features/<dataset>__<model>.npz.

What it does: runs each CLIP image encoder from the config over all Indian and Food-101 images and saves
the L2-normalised image embeddings. Later steps only load this cache, so the expensive image encoding
is done once and E2-E5 become cheap to run and re-run.
Experiment: preparation step for E2-E4 (run_zeroshot.py) and E5 (run_fewshot.py).
Writes: features/<dataset>__<model>__<pretrained>.npz (gitignored, no result CSV or figure).
Report section: Methods "CLIP zero-shot and prompt engineering" and "Few-shot linear probe" (shared features).

Pipeline position: runs after prepare_data.py (needs the split CSVs) and before run_zeroshot.py and
run_fewshot.py, which load the .npz files this script writes.

Background: CLIP has an image encoder that turns a picture into a vector (an "embedding", e.g. 512
numbers) and a text encoder that does the same for text. The encoder is frozen (not trained here), so
each image needs to be encoded only once. L2-normalised means every vector is scaled to length 1, so
a dot product between two vectors equals their cosine similarity.

  python scripts/extract_features.py
  python scripts/extract_features.py --models ViT-B-32 --datasets indian --force
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
# (adds <repo>/src to Python's import path so "import foodclip" works)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
# SPLIT_FILES: dict mapping dataset name ("indian"/"food101") to its split CSV file name.
# load_dataset: read that CSV, returning (image paths, table, class names).
from foodclip.data import SPLIT_FILES, load_dataset  # noqa: E402
# load_clip: build an open_clip model + image preprocessing + tokenizer from a config entry.
# encode_images: run the image encoder over a list of files in batches, return normalised embeddings.
# feature_path: where the cache file for (dataset, model) lives. save_features: write the .npz file.
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
    device = get_device()  # "cuda" if a GPU is available, otherwise CPU (much slower)
    # Keep only datasets whose split CSV exists; tell the user about the others.
    datasets = [d for d in args.datasets if (resolve(cfg["paths"]["splits"]) / SPLIT_FILES[d]).exists()]
    for d in set(args.datasets) - set(datasets):
        print(f"skip {d}: no split file (run prepare_data.py first)")

    # Loop over the CLIP models listed in the config.
    # Each entry m is a dict with at least "name" (e.g. ViT-B-32) and "pretrained" (weights tag).
    for m in cfg["clip"]["models"]:
        # Skip models not asked for on the command line.
        if args.models and m["name"] not in args.models:
            continue
        # Datasets still to encode for this model (all of them with --force).
        # This makes the script resumable: finished (dataset, model) pairs are not recomputed.
        todo = [d for d in datasets if args.force or not feature_path(cfg, d, m).exists()]
        if not todo:
            print(f"{m['name']}: cached, skipping")
            continue
        # Load the model once, then encode each dataset and save features in split-CSV order.
        # The text tokenizer (third value) is not needed here, hence "_".
        model, preprocess, _ = load_clip(m, device)
        for d in todo:
            paths, df, _ = load_dataset(cfg, d)
            # preprocess resizes/normalises each image the way this CLIP model expects.
            feats = encode_images(model, preprocess, paths, device, cfg["clip"]["batch_size"],
                                  cfg["clip"]["num_workers"])  # shape (n_images, embedding_dim)
            # Store relpaths too, so a later load can check the order still matches the split.
            # Row i of feats is image i of the split CSV; if the CSV is reordered the check fails loudly.
            save_features(feature_path(cfg, d, m), feats, df["relpath"].tolist())
            print(f"{m['name']} {d}: {feats.shape} -> {feature_path(cfg, d, m)}")
        # Free GPU memory before loading the next model.
        del model
        gc.collect()
        torch.cuda.empty_cache()


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
