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
    """Scores (n, C) and true labels (n,) in; share of images whose true class is in the k best scores out."""
    # k cannot be larger than the number of classes.
    k = min(k, scores.shape[1])
    # Sort each row from highest to lowest score and keep the first k class indices.
    topk = np.argsort(-scores, axis=1)[:, :k]  # (n, k); minus sign = descending order
    # A hit if any of the k guesses equals the true label; the mean of hits is the accuracy.
    return float(np.mean(np.any(topk == y[:, None], axis=1)))  # y[:, None] is (n, 1), compared to all k columns


def summarize(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """scores: (n, C) — higher is better. y: (n,) int labels.

    Returns n_test, top-1, top-5 and macro-F1 (F1 of each class, then the plain average over classes,
    so every class counts the same no matter how many test images it has)."""
    # Predicted class = the class with the highest score.
    y = np.asarray(y)
    pred = scores.argmax(1)
    # labels=all C classes, so a class that is never predicted still counts (with F1 = 0).
    return {
        "n_test": int(len(y)),
        "top1": topk_accuracy(scores, y, 1),
        "top5": topk_accuracy(scores, y, 5),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(scores.shape[1]), zero_division=0)),
    }


def per_class_accuracy(y: np.ndarray, pred: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """True and predicted labels in; table of (class, n, accuracy) out, worst class first."""
    rows = []
    # For each class: accuracy = share of its test images that were predicted as that class.
    for c, name in enumerate(names):
        mask = y == c  # True for test images of class c
        if mask.any():
            rows.append({"class": name, "n": int(mask.sum()), "accuracy": float((pred[mask] == c).mean())})
    # Sort from lowest to highest accuracy so the hardest classes come first.
    return pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)


def top_confusions(y: np.ndarray, pred: np.ndarray, names: Sequence[str], n: int = 20) -> pd.DataFrame:
    """True and predicted labels in; the n most frequent (true, predicted) mistakes with their count out."""
    # Keep only wrong predictions and turn class indices into class names.
    wrong = y != pred
    pairs = pd.DataFrame({"true": np.asarray(names)[y[wrong]], "predicted": np.asarray(names)[pred[wrong]]})
    # No mistakes: return an empty table with the same columns.
    if pairs.empty:
        return pd.DataFrame(columns=["true", "predicted", "count"])
    # Count each (true, predicted) pair; value_counts already sorts from most to least frequent.
    out = pairs.value_counts().reset_index(name="count")
    return out.head(n)
