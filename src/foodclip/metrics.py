"""Classification metrics shared by every experiment.

Metrics: top-1 accuracy, top-5 accuracy, macro-F1, per-class accuracy, and the most common confusions.
Used by E1-E5 and E7 (train_cnn, run_zeroshot, run_fewshot, finetune_clip); E6 builds on their CSVs.
Report: the metric definitions in Methods and every Results subsection.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def topk_accuracy(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """Scores (n, C) and true labels (n,) in; share of images whose true class is in the k best scores out.

    Top-1 = the best guess is right; top-5 = the true class is among the five best guesses. Returns a
    number between 0 and 1. Used by: summarize (and so by every experiment)."""
    # k cannot be larger than the number of classes (scores.shape[1] = C, the number of columns).
    k = min(k, scores.shape[1])
    # Sort each row from highest to lowest score and keep the first k class indices.
    # np.argsort(-scores, axis=1) returns, per row, the class indices ordered by ascending value of -scores,
    # i.e. best class first; shape (n, C). Negating is needed because argsort only sorts ascending.
    # [:, :k] keeps all rows and the first k columns, the k best class indices per image.
    topk = np.argsort(-scores, axis=1)[:, :k]  # (n, k); minus sign = descending order
    # A hit if any of the k guesses equals the true label; the mean of hits is the accuracy.
    # topk == y[:, None] -> (n, k) booleans; np.any(..., axis=1) -> (n,) True if any column matches;
    # np.mean of booleans = share of True; float() makes a plain Python number.
    return float(np.mean(np.any(topk == y[:, None], axis=1)))  # y[:, None] is (n, 1), compared to all k columns


def summarize(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """scores: (n, C) — higher is better. y: (n,) int labels.

    Returns n_test, top-1, top-5 and macro-F1 (F1 of each class, then the plain average over classes,
    so every class counts the same no matter how many test images it has)."""
    # Predicted class = the class with the highest score.
    # np.asarray(y) makes sure y is a numpy array (no copy if it already is one).
    y = np.asarray(y)
    # argmax(1) = index of the largest value along axis 1 (the class axis) -> (n,) predicted class indices.
    pred = scores.argmax(1)
    # labels=all C classes, so a class that is never predicted still counts (with F1 = 0).
    # Entries: n_test = number of images; top1/top5 from topk_accuracy; macro_f1 from sklearn f1_score:
    #   average="macro" computes F1 per class and takes the unweighted mean over the classes in labels;
    #   labels=np.arange(C) lists every class index 0..C-1 (np.arange makes that integer range);
    #   zero_division=0 gives F1 = 0 (without a warning) when a class has no predictions and no true images.
    return {
        "n_test": int(len(y)),
        "top1": topk_accuracy(scores, y, 1),
        "top5": topk_accuracy(scores, y, 5),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(scores.shape[1]), zero_division=0)),
    }


def per_class_accuracy(y: np.ndarray, pred: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """True and predicted labels in; table of (class, n, accuracy) out, worst class first.

    Why: overall accuracy can hide classes that fail completely. Needs class names in index order.
    Used by: the scripts that write per-class result files and by make_figures.py."""
    # List of dicts, one per class; becomes the DataFrame at the end.
    rows = []
    # For each class: accuracy = share of its test images that were predicted as that class.
    for c, name in enumerate(names):
        mask = y == c  # True for test images of class c
        # Skip classes with no test images (their accuracy would be undefined).
        if mask.any():
            # mask.sum() = number of test images of the class; (pred[mask] == c).mean() = share predicted correctly.
            rows.append({"class": name, "n": int(mask.sum()), "accuracy": float((pred[mask] == c).mean())})
    # Sort from lowest to highest accuracy so the hardest classes come first.
    # pd.DataFrame(rows) builds the table from the dicts; sort_values("accuracy") orders rows ascending by that
    # column; reset_index(drop=True) renumbers rows 0..m-1 and discards the old index.
    return pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)


def top_confusions(y: np.ndarray, pred: np.ndarray, names: Sequence[str], n: int = 20) -> pd.DataFrame:
    """True and predicted labels in; the n most frequent (true, predicted) mistakes with their count out.

    Why: shows which dishes the model mixes up (e.g. two similar curries), useful for error analysis
    in the report. Used by: the scripts that write confusion files."""
    # Keep only wrong predictions and turn class indices into class names.
    # wrong is a boolean (n,) mask, True where the prediction differs from the truth.
    wrong = y != pred
    # np.asarray(names)[indices] looks up the name for each index array; the DataFrame has one row per mistake.
    pairs = pd.DataFrame({"true": np.asarray(names)[y[wrong]], "predicted": np.asarray(names)[pred[wrong]]})
    # No mistakes: return an empty table with the same columns.
    if pairs.empty:
        return pd.DataFrame(columns=["true", "predicted", "count"])
    # Count each (true, predicted) pair; value_counts already sorts from most to least frequent.
    # DataFrame.value_counts() counts identical rows -> a Series indexed by (true, predicted); reset_index(name="count")
    # turns the index into columns "true" and "predicted" and names the counts column "count".
    out = pairs.value_counts().reset_index(name="count")
    # head(n) keeps the first n rows, i.e. the n most frequent mistakes.
    return out.head(n)
