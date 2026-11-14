"""WP3 / E2-E4: CLIP zero-shot on the Indian test split (3 languages x templates) and on the
Food-101 subset (English names). Needs cached features from extract_features.py.

Writes results/zeroshot.csv, results/zeroshot_per_class.csv, results/zeroshot_confusions.csv
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

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
    names = load_class_names(resolve(cfg["paths"]["class_names"])).set_index("folder")
    out = []
    for c in classes:
        if c in names.index:
            out.append(str(names.loc[c, language]))
        else:
            print(f"WARNING {c} missing from class_names.csv; using folder name")
            out.append(c.replace("_", " "))
    return out


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="*")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    rows, per_class, confusions = [], [], []

    for m in cfg["clip"]["models"]:
        if args.models and m["name"] not in args.models:
            continue
        model, _, tokenizer = load_clip(m, device)
        for dataset in ("indian", "food101"):
            fpath = feature_path(cfg, dataset, m)
            if not fpath.exists():
                print(f"skip {dataset} {m['name']}: no cached features")
                continue
            _, df, classes = load_dataset(cfg, dataset)
            feats = load_features(fpath, df["relpath"].tolist())
            test = df["split"].to_numpy() == "test"
            x, y = feats[test], df["y"].to_numpy()[test]
            if dataset == "indian":
                combos = [(lang, t) for lang in cfg["zeroshot"]["languages"] for t in cfg["zeroshot"]["templates"]
                          if valid_combo(lang, t)]
            else:
                combos = [("english", t) for t in FOOD101_TEMPLATES]
            for lang, template in combos:
                names = indian_names(cfg, classes, lang) if dataset == "indian" else [food101_name(c) for c in classes]
                text = encode_text(model, tokenizer, build_prompts(names, template), device)
                scores = zero_shot_scores(x, text)
                key = {"dataset": dataset, "model": m["name"], "pretrained": m["pretrained"],
                       "language": lang, "template": template}
                rows.append({**key, **summarize(scores, y), **run_meta()})
                pred = scores.argmax(1)
                per_class += [{**key, **r} for r in per_class_accuracy(y, pred, classes).to_dict("records")]
                if template == "food":
                    confusions += [{**key, **r} for r in top_confusions(y, pred, classes).to_dict("records")]
                print(f"{model_tag(m):45s} {dataset:8s} {lang:17s} {template:12s} top1 {rows[-1]['top1']:.3f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    res = resolve(cfg["paths"]["results"])
    write_rows(rows, res / "zeroshot.csv")
    write_rows(per_class, res / "zeroshot_per_class.csv")
    write_rows(confusions, res / "zeroshot_confusions.csv")
    print(f"wrote {len(rows)} rows to {res / 'zeroshot.csv'}")


if __name__ == "__main__":
    main()
