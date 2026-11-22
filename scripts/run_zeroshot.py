"""WP3 / E2-E4: CLIP zero-shot on the Indian test split (3 languages x templates) and on the
Food-101 subset (English names). Needs cached features from extract_features.py.

What it does: for each CLIP model, turns class names into text prompts, encodes them, and labels each test
image with the most similar prompt (no training). It compares prompt languages (English description,
romanised Hindi, Devanagari) and templates on the Indian set, and runs the same model on Food-101.
Experiments: E2 (zero-shot accuracy), E3 (prompt language and templates), E4 (Indian vs Western bias).
Report sections: Methods "CLIP zero-shot and prompt engineering"; Results "Zero-shot and prompt language"
(Fig. zeroshot_language) and "Indian vs Western bias" (Fig. zeroshot_bias, worst_classes).

Writes results/zeroshot.csv, results/zeroshot_per_class.csv, results/zeroshot_confusions.csv

Pipeline position: runs after extract_features.py (reads the cached image features, so no images are
opened here) and before make_figures.py (which plots the CSVs written here).

How zero-shot works: for each class we write a sentence such as "a photo of <dish>, a type of Indian
food", encode it with CLIP's text encoder, and get one text vector per class. An image is assigned to
the class whose text vector has the highest cosine similarity with the image vector. No weights are
trained here; classification uses only the text prompts.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
# (adds <repo>/src to Python's import path so "import foodclip" works)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
# load_class_names: read data/class_names.csv (folder, english, hindi_roman, hindi_devanagari).
# load_dataset: read the split CSV into (paths, table, class names).
from foodclip.data import load_class_names, load_dataset  # noqa: E402
# load_clip: model + tokenizer. encode_text: encodes each prompt, averages the prompts of one class
# (matters for the "ensemble" template) and normalises the result to length 1.
# feature_path / load_features: find and read the cached image features. model_tag: short model label for printing.
from foodclip.features import encode_text, feature_path, load_clip, load_features, model_tag  # noqa: E402
# zero_shot_scores: image-vs-text similarity matrix (scaled cosine similarity).
from foodclip.fewshot import zero_shot_scores  # noqa: E402
from foodclip.io import get_device, run_meta, write_rows  # noqa: E402
# summarize: top1/top5/macro_f1. per_class_accuracy: accuracy per class. top_confusions: most common
# (true class, predicted class) mistakes.
from foodclip.metrics import per_class_accuracy, summarize, top_confusions  # noqa: E402
# build_prompts: put class names into a template sentence. food101_name: pretty English name for a
# Food-101 folder (e.g. apple_pie -> apple pie). valid_combo: the Hindi sentence template (hindi_native)
# is only used with Devanagari names; every other language/template pair is allowed.
from foodclip.prompts import build_prompts, food101_name, valid_combo  # noqa: E402

FOOD101_TEMPLATES = ["plain", "food"]  # no "Indian food" wording for the Western set


def indian_names(cfg: dict, classes: list[str], language: str) -> list[str]:
    """Return the class name in the chosen language for each class, in class-index order.

    `language` is a column of class_names.csv (english, hindi_roman or hindi_devanagari). The order of the
    result must match the class numbers used for the labels, so we loop over `classes` (already in that order).
    """
    # set_index("folder") lets us look up a row by the dataset folder name instead of by position.
    # resolve makes the config path absolute; load_class_names reads the CSV sorted by folder.
    names = load_class_names(resolve(cfg["paths"]["class_names"])).set_index("folder")  # look up rows by folder
    out = []
    # Use the name from class_names.csv; if a class is missing, fall back to its folder name.
    for c in classes:
        if c in names.index:
            # names.loc[row label, column] picks one cell; str() makes sure it is plain text.
            out.append(str(names.loc[c, language]))
        else:
            print(f"WARNING {c} missing from class_names.csv; using folder name")
            # Fallback name: the folder name with underscores replaced by spaces.
            out.append(c.replace("_", " "))
    # List of n_classes strings, aligned with the class indices.
    return out


def main() -> None:
    """Score every model x dataset x language x template combination zero-shot and save the results."""
    # Read options and set up empty result lists.
    # base_parser adds the shared --config and --set options; the script's docstring is the help text.
    p = base_parser(__doc__)
    # --models takes zero or more model names to restrict the run (nargs="*"); default None means all models.
    p.add_argument("--models", nargs="*")
    # parse_args reads sys.argv; config_from_args loads the YAML and applies --set overrides.
    args = p.parse_args()
    cfg = config_from_args(args)
    # torch device: "cuda" if a GPU is available, else "cpu" (used for the text encoder only).
    device = get_device()
    # rows: one summary per setting. per_class: accuracy per class per setting.
    # confusions: most frequent wrong predictions.
    rows, per_class, confusions = [], [], []

    # Loop over CLIP models; skip models not asked for on the command line.
    for m in cfg["clip"]["models"]:
        # --models filter: if given, skip every model whose name is not listed.
        if args.models and m["name"] not in args.models:
            continue
        # Only the text encoder and tokenizer are used here; image features come from the cache.
        # (the image preprocess function, second value, is not needed)
        model, _, tokenizer = load_clip(m, device)
        # Both datasets are scored with the same model, so the Indian and Western numbers are comparable.
        for dataset in ("indian", "food101"):
            # Skip a dataset if its image features were not cached yet.
            fpath = feature_path(cfg, dataset, m)
            if not fpath.exists():
                print(f"skip {dataset} {m['name']}: no cached features")
                continue
            # Load cached features and keep only the test images.
            # load_features checks the stored file list against df["relpath"], so rows line up with labels.
            _, df, classes = load_dataset(cfg, dataset)
            feats = load_features(fpath, df["relpath"].tolist())
            # df["split"].to_numpy() is an (n_images,) array of strings; == "test" gives a boolean mask of that shape.
            test = df["split"].to_numpy() == "test"  # boolean mask over all rows
            # Boolean-mask indexing keeps only the test rows: x (n_test, embedding_dim), y (n_test,) int labels.
            x, y = feats[test], df["y"].to_numpy()[test]  # x shape (n_test, embedding_dim)
            # Indian: every valid language x template pair. Food-101: English names with two templates.
            # combos is a list of (language, template) pairs; the config lists give the languages and templates,
            # and valid_combo drops pairs that make no sense (hindi_native template with non-Devanagari names).
            if dataset == "indian":
                combos = [(lang, t) for lang in cfg["zeroshot"]["languages"] for t in cfg["zeroshot"]["templates"]
                          if valid_combo(lang, t)]
            else:
                combos = [("english", t) for t in FOOD101_TEMPLATES]
            # Every setting is scored on the test split. Picking the best setting afterwards (as the report and
            # make_figures.py do) is a choice made on the test set, so it favours zero-shot.
            for lang, template in combos:
                # Build prompts from class names, encode them, and score each image against each class.
                # names: list of n_classes strings in class-index order. build_prompts returns one inner list of
                # filled sentences per class (several for "ensemble").
                names = indian_names(cfg, classes, lang) if dataset == "indian" else [food101_name(c) for c in classes]
                # encode_text: tokenize and encode each class's prompts, average them, renormalise to length 1.
                text = encode_text(model, tokenizer, build_prompts(names, template), device)  # (n_classes, dim)
                # zero_shot_scores = 100 * x @ text.T: (n_test, dim) @ (dim, n_classes); higher = more similar.
                scores = zero_shot_scores(x, text)  # scaled cosine similarity, shape (n_test, n_classes)
                # Save one summary row; the key columns say exactly which setting produced it.
                # key: columns identifying this setting; copied into every row written below.
                key = {"dataset": dataset, "model": m["name"], "pretrained": m["pretrained"],
                       "language": lang, "template": template}
                # summarize adds n_test, top1, top5, macro_f1; run_meta adds run time, versions and device.
                # ** unpacks each dict into the new row dict.
                rows.append({**key, **summarize(scores, y), **run_meta()})
                # Per-class accuracy for every setting (used for the hardest-dishes figure).
                # argmax(1): for each image, the class with the highest score is the prediction.
                # argmax(1) works along axis 1 (the class axis): scores (n_test, n_classes) -> pred (n_test,) ints.
                pred = scores.argmax(1)
                # per_class_accuracy gives a table (class, n, accuracy), worst first; to_dict("records") turns each
                # table row into a dict, and += appends them all to the list after adding the key columns.
                per_class += [{**key, **r} for r in per_class_accuracy(y, pred, classes).to_dict("records")]
                # Most common wrong (true, predicted) pairs, saved only for the "food" template.
                if template == "food":
                    # top_confusions: table (true, predicted, count) of the most frequent mistakes, up to 20 rows.
                    confusions += [{**key, **r} for r in top_confusions(y, pred, classes).to_dict("records")]
                # Progress line: model_tag is "<architecture>__<pretrained>"; :45s/:8s/:17s/:12s pad to that width,
                # :.3f prints top1 with 3 decimals.
                print(f"{model_tag(m):45s} {dataset:8s} {lang:17s} {template:12s} top1 {rows[-1]['top1']:.3f}")
        # Free GPU memory before loading the next model.
        # del drops this script's reference to the model; gc.collect() runs Python's garbage collector so the
        # model object is really freed; torch.cuda.empty_cache() returns cached GPU memory to the driver
        # (does nothing harmful on a CPU-only machine).
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Write all three result CSVs (overwritten on each run).
    # Overwritten on each run (like fewshot.csv): a full re-run replaces the old results.
    # Only the CNN and fine-tune CSVs append.
    res = resolve(cfg["paths"]["results"])
    # write_rows builds a DataFrame from the list of dicts and saves it as UTF-8 CSV (creates the folder if needed).
    write_rows(rows, res / "zeroshot.csv")
    write_rows(per_class, res / "zeroshot_per_class.csv")
    write_rows(confusions, res / "zeroshot_confusions.csv")
    # Final message: number of summary rows and where they were saved.
    print(f"wrote {len(rows)} rows to {res / 'zeroshot.csv'}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
