"""E2/E5: zero-shot scoring and few-shot linear probes on cached CLIP features."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def zero_shot_scores(image_feats: np.ndarray, text_feats: np.ndarray) -> np.ndarray:
    """Cosine similarity (features are already L2-normalised) scaled like CLIP."""
    return 100.0 * image_feats @ text_feats.T


def sample_k_shot(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Indices of k random examples per class (all examples if k <= 0 or class is smaller)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        members = np.nonzero(y == c)[0]
        if 0 < k < len(members):
            members = rng.choice(members, size=k, replace=False)
        idx.extend(sorted(members.tolist()))
    return np.asarray(sorted(idx))


def linear_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int,
                 C: float = 0.316, seed: int = 0) -> np.ndarray:
    """Multinomial logistic regression; returns (n_test, n_classes) class probabilities."""
    clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    clf.fit(x_train, y_train)
    proba = np.zeros((len(x_test), n_classes), dtype=np.float32)
    proba[:, clf.classes_] = clf.predict_proba(x_test)
    return proba
