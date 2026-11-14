"""WP4 / E5: linear probe on cached CLIP features of the Indian set, k shots per class x seeds.

Writes results/fewshot.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import load_dataset  # noqa: E402
from foodclip.features import feature_path, load_features  # noqa: E402
from foodclip.fewshot import linear_probe, sample_k_shot  # noqa: E402
from foodclip.io import run_meta, write_rows  # noqa: E402
from foodclip.metrics import summarize  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="*")
    args = p.parse_args()
    cfg = config_from_args(args)
    fs = cfg["fewshot"]
    _, df, classes = load_dataset(cfg, "indian")
    split, y = df["split"].to_numpy(), df["y"].to_numpy()
    train, test = split == "train", split == "test"
    rows = []
    for m in cfg["clip"]["models"]:
        if args.models and m["name"] not in args.models:
            continue
        fpath = feature_path(cfg, "indian", m)
        if not fpath.exists():
            print(f"skip {m['name']}: no cached features")
            continue
        feats = load_features(fpath, df["relpath"].tolist())
        x_tr, y_tr, x_te, y_te = feats[train], y[train], feats[test], y[test]
        for k in fs["shots"]:
            for seed in (fs["seeds"] if k > 0 else fs["seeds"][:1]):  # "all" is deterministic
                idx = sample_k_shot(y_tr, k, seed)
                scores = linear_probe(x_tr[idx], y_tr[idx], x_te, len(classes), fs["C"], seed)
                rows.append({"model": m["name"], "pretrained": m["pretrained"], "shots": k if k > 0 else "all",
                             "seed": seed, "n_train": len(idx), **summarize(scores, y_te), **run_meta()})
                print(f"{m['name']:28s} shots {rows[-1]['shots']!s:>4} seed {seed} top1 {rows[-1]['top1']:.3f}")
    out = resolve(cfg["paths"]["results"]) / "fewshot.csv"
    write_rows(rows, out)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
