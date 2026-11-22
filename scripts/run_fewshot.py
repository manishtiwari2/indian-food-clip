"""WP4 / E5: linear probe on cached CLIP features of the Indian set, k shots per class x seeds.

What it does: for each CLIP model, trains a logistic-regression classifier on the frozen image features of
k labelled training images per class, and tests it on the Indian test split. Each k is repeated with
several seeds (different random images) so the curve can show a mean and spread. Shots value 0 means
"use all training images".
Experiment: E5 (few-shot linear probe); its results also feed the E6 gap table in make_figures.py.
Report sections: Methods "Few-shot linear probe"; Results "How many shots close the gap?" (Fig. fewshot_curve).

Writes results/fewshot.csv (column class_weight: balanced or none). With class weights on, one extra
unweighted "all" row per model (seed 0) is added as a reference.

Pipeline position: runs after extract_features.py (reads the cached image features; no images or text
encoder are used) and before make_figures.py. It is the supervised counterpart of run_zeroshot.py:
zero-shot uses 0 labelled images, this uses k per class, so together they show how many labels are
needed to close the Indian-vs-Western gap.

Linear probe = keep CLIP frozen and train only one linear layer (here logistic regression) on top of
its image vectors. Only the linear layer is trained, CLIP stays frozen.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
# (adds <repo>/src to Python's import path so "import foodclip" works)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import load_dataset  # noqa: E402
# feature_path: where the cached features for (dataset, model) are. load_features: read them and check the order.
from foodclip.features import feature_path, load_features  # noqa: E402
# sample_k_shot: pick k random training row indices per class (all rows when k is 0).
# linear_probe: fit logistic regression on the training features, return class probabilities for the test features.
from foodclip.fewshot import linear_probe, sample_k_shot  # noqa: E402
from foodclip.io import run_meta, write_rows  # noqa: E402
from foodclip.metrics import summarize  # noqa: E402


def main() -> None:
    """Run the linear probe for every model, shot count and seed, and save one row per run."""
    # Read options and the few-shot settings from the config.
    # base_parser adds the shared --config and --set options; the script's docstring is the help text.
    p = base_parser(__doc__)
    # --models takes zero or more model names to restrict the run (nargs="*"); default None means all models.
    p.add_argument("--models", nargs="*")
    # parse_args reads sys.argv into a Namespace; config_from_args loads the YAML and applies --set overrides.
    args = p.parse_args()
    cfg = config_from_args(args)
    fs = cfg["fewshot"]  # shots list, seeds list, regularisation strength C, class_weight
    # class_weight "balanced" only changes the "all" runs (19-34 train images per class); for k<=16 every class
    # has k images, so all weights are 1. We added the weights after a first run without them, where "all" was
    # worse than k=16, and after looking at the test accuracy (D12). So it is a post-hoc choice.
    class_weight = fs.get("class_weight")  # "balanced" or None; a missing key keeps the old behaviour
    # Load the Indian split and make boolean masks for train and test rows.
    # load_dataset returns (image paths, table, sorted class names); the paths are not needed here (the "_").
    # df has one row per image, with columns relpath, label, split and the integer class index y.
    _, df, classes = load_dataset(cfg, "indian")
    # split: (n_images,) array of "train"/"test" strings; y: (n_images,) integer class indices.
    split, y = df["split"].to_numpy(), df["y"].to_numpy()
    # Boolean masks of shape (n_images,): True where the row belongs to the train / test split.
    # A row with any other split value would be in neither mask and would not be used.
    train, test = split == "train", split == "test"
    rows = []  # one dict per (model, shots, seed) run; written to CSV at the end
    # Loop over CLIP models; skip models not asked for or without cached features.
    for m in cfg["clip"]["models"]:
        # --models filter: if given, skip every model whose name is not listed.
        if args.models and m["name"] not in args.models:
            continue
        # Path of the cached .npz feature file for the Indian set and this model.
        fpath = feature_path(cfg, "indian", m)
        # No cache means extract_features.py has not run for this model, so there is nothing to probe.
        if not fpath.exists():
            print(f"skip {m['name']}: no cached features")
            continue
        # Load the cached image features (no image encoding needed here) and split them.
        # load_features raises ValueError if the cached image list differs from df["relpath"], so row i of
        # feats (n_images, embedding_dim) is image i of df and lines up with y.
        feats = load_features(fpath, df["relpath"].tolist())
        # Boolean-mask indexing keeps only the masked rows, so the train/test arrays are separate copies.
        x_tr, y_tr, x_te, y_te = feats[train], y[train], feats[test], y[test]  # x: (n, embedding_dim)

        def run_one(k: int, seed: int, class_weight: str | None) -> None:
            """Sample k images per class with this seed, fit the probe, and store its test metrics."""
            # Pick k random training images per class (all of them when k is 0).
            # Different seeds give different random picks, which shows how much the result depends on luck.
            idx = sample_k_shot(y_tr, k, seed)
            # Train logistic regression on those features; scores are class probabilities (n_test, n_classes).
            # fs["C"] is the inverse regularisation strength (smaller C = stronger regularisation).
            # Its value (0.316) was set by the lead; reason not recorded in the repo.
            # x_tr[idx] is (len(idx), embedding_dim) (fancy indexing picks those rows); len(classes) sets the
            # width of the returned table. Inside linear_probe: sklearn LogisticRegression (max_iter=3000,
            # random_state=seed) is fitted, then predict_proba gives one probability per class per test image.
            scores = linear_probe(x_tr[idx], y_tr[idx], x_te, len(classes), fs["C"], seed, class_weight)
            # One result row; shots 0 is written as "all". n_train = number of training images actually used.
            # summarize(scores, y_te) adds n_test, top1, top5, macro_f1 (argmax over classes = prediction);
            # run_meta() adds run time, torch and Python versions and device name. ** merges dicts into this row.
            rows.append({"model": m["name"], "pretrained": m["pretrained"], "shots": k if k > 0 else "all",
                         "seed": seed, "class_weight": class_weight or "none", "n_train": len(idx),
                         **summarize(scores, y_te), **run_meta()})
            # Progress line: :28s pads the model name to 28 characters, :>4 right-aligns the shots value in 4,
            # :>8 right-aligns the weights label, :.3f prints top1 with 3 decimals.
            print(f"{m['name']:28s} shots {rows[-1]['shots']!s:>4} seed {seed} weights {rows[-1]['class_weight']:>8}"
                  f" top1 {rows[-1]['top1']:.3f}")

        # Every shot count with every seed; the figure later averages over seeds.
        # k = shots per class from the config list (0 means all training images). For k > 0 every seed is run;
        # for k = 0 only the first seed, because the full training set gives the same fit for any seed.
        for k in fs["shots"]:
            for seed in (fs["seeds"] if k > 0 else fs["seeds"][:1]):  # "all" is deterministic
                run_one(k, seed, class_weight)
        # Only when weights are on: also fit the "all" setting with no weights (class_weight=None), first seed.
        if class_weight is not None:  # extra reference row: "all" without weights (not in curve or gap)
            run_one(0, fs["seeds"][0], None)
    # Write all rows (the file is overwritten on each run).
    # resolve makes the relative results path absolute from the repo root; "/" joins the file name onto it.
    out = resolve(cfg["paths"]["results"]) / "fewshot.csv"
    # Build a DataFrame from the list of row dicts (one CSV row per dict) and write it as UTF-8 CSV.
    write_rows(rows, out)
    print(f"wrote {len(rows)} rows to {out}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
