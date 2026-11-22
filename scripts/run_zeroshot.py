"""WP3 / E2-E4: CLIP zero-shot on the Indian test split (3 languages x templates) and on the
Food-101 subset (English names). Needs cached features from extract_features.py.

What it does: for each CLIP model, turns class names into text prompts, encodes them, and labels each test
image with the most similar prompt (no training). It compares prompt languages (English description,
romanised Hindi, Devanagari) and templates on the Indian set, and runs the same model on Food-101.
Experiments: E2 (zero-shot accuracy), E3 (prompt language and templates), E4 (Indian vs Western bias).
Report sections: Methods "CLIP zero-shot and prompt engineering"; Results "Zero-shot and prompt language"
(Fig. zeroshot_language) and "Indian vs Western bias" (Fig. zeroshot_bias, worst_classes).

Writes results/zeroshot.csv, results/zeroshot_per_class.csv, results/zeroshot_confusions.csv
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import load_class_names, load_dataset  # noqa: E402
from foodclip.features import encode_text, feature_path, load_clip, load_features, model_tag  # noqa: E402
from foodclip.fewshot import zero_shot_scores  # noqa: E402
from foodclip.io import get_device, run_meta, write_rows  # noqa: E402
from foodclip.metrics import per_class_accuracy, summarize, top_confusions  # noqa: E402
from foodclip.prompts import build_prompts, food101_name, valid_combo  # noqa: E402

FOOD101_TEMPLATES = ["plain", "food"]  # no "Indian food" wording for the Western set


def indian_names(cfg: dict, classes: list[str], language: str) -> list[str]:
    """Return the class name in the chosen language for each class, in class-index order."""
    names = load_class_names(resolve(cfg["paths"]["class_names"])).set_index("folder")  # look up rows by folder
    out = []
    # Use the name from class_names.csv; if a class is missing, fall back to its folder name.
    for c in classes:
        if c in names.index:
            out.append(str(names.loc[c, language]))
        else:
            print(f"WARNING {c} missing from class_names.csv; using folder name")
            out.append(c.replace("_", " "))
    return out


def main() -> None:
    """Score every model x dataset x language x template combination zero-shot and save the results."""
    # Read options and set up empty result lists.
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="*")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    rows, per_class, confusions = [], [], []

    # Loop over CLIP models; skip models not asked for on the command line.
    for m in cfg["clip"]["models"]:
        if args.models and m["name"] not in args.models:
            continue
        # Only the text encoder and tokenizer are used here; image features come from the cache.
        model, _, tokenizer = load_clip(m, device)
        for dataset in ("indian", "food101"):
            # Skip a dataset if its image features were not cached yet.
            fpath = feature_path(cfg, dataset, m)
            if not fpath.exists():
                print(f"skip {dataset} {m['name']}: no cached features")
                continue
            # Load cached features and keep only the test images.
            _, df, classes = load_dataset(cfg, dataset)
            feats = load_features(fpath, df["relpath"].tolist())
            test = df["split"].to_numpy() == "test"  # boolean mask over all rows
            x, y = feats[test], df["y"].to_numpy()[test]  # x shape (n_test, embedding_dim)
            # Indian: every valid language x template pair. Food-101: English names with two templates.
            if dataset == "indian":
                combos = [(lang, t) for lang in cfg["zeroshot"]["languages"] for t in cfg["zeroshot"]["templates"]
                          if valid_combo(lang, t)]
            else:
                combos = [("english", t) for t in FOOD101_TEMPLATES]
            for lang, template in combos:
                # Build prompts from class names, encode them, and score each image against each class.
                names = indian_names(cfg, classes, lang) if dataset == "indian" else [food101_name(c) for c in classes]
                text = encode_text(model, tokenizer, build_prompts(names, template), device)  # (n_classes, dim)
                scores = zero_shot_scores(x, text)  # scaled cosine similarity, shape (n_test, n_classes)
                # Save one summary row; the key columns say exactly which setting produced it.
                key = {"dataset": dataset, "model": m["name"], "pretrained": m["pretrained"],
                       "language": lang, "template": template}
                rows.append({**key, **summarize(scores, y), **run_meta()})
                # Per-class accuracy for every setting (used for the hardest-dishes figure).
                pred = scores.argmax(1)
                per_class += [{**key, **r} for r in per_class_accuracy(y, pred, classes).to_dict("records")]
                # Most common wrong (true, predicted) pairs, saved only for the "food" template.
                if template == "food":
                    confusions += [{**key, **r} for r in top_confusions(y, pred, classes).to_dict("records")]
                print(f"{model_tag(m):45s} {dataset:8s} {lang:17s} {template:12s} top1 {rows[-1]['top1']:.3f}")
        # Free GPU memory before loading the next model.
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Write all three result CSVs (overwritten on each run).
    res = resolve(cfg["paths"]["results"])
    write_rows(rows, res / "zeroshot.csv")
    write_rows(per_class, res / "zeroshot_per_class.csv")
    write_rows(confusions, res / "zeroshot_confusions.csv")
    print(f"wrote {len(rows)} rows to {res / 'zeroshot.csv'}")


if __name__ == "__main__":
    main()
