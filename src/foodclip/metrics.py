"""Classification metrics shared by every experiment."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def topk_accuracy(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean(np.any(topk == y[:, None], axis=1)))


def summarize(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """scores: (n, C) — higher is better. y: (n,) int labels."""
    y = np.asarray(y)
    pred = scores.argmax(1)
    return {
        "n_test": int(len(y)),
        "top1": topk_accuracy(scores, y, 1),
        "top5": topk_accuracy(scores, y, 5),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(scores.shape[1]), zero_division=0)),
    }


def per_class_accuracy(y: np.ndarray, pred: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for c, name in enumerate(names):
        mask = y == c
        if mask.any():
            rows.append({"class": name, "n": int(mask.sum()), "accuracy": float((pred[mask] == c).mean())})
    return pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)


def top_confusions(y: np.ndarray, pred: np.ndarray, names: Sequence[str], n: int = 20) -> pd.DataFrame:
    wrong = y != pred
    pairs = pd.DataFrame({"true": np.asarray(names)[y[wrong]], "predicted": np.asarray(names)[pred[wrong]]})
    if pairs.empty:
        return pd.DataFrame(columns=["true", "predicted", "count"])
    out = pairs.value_counts().reset_index(name="count")
    return out.head(n)
