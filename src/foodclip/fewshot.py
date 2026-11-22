"""E2/E5: zero-shot scoring and few-shot linear probes on cached CLIP features.

Used by E2-E4 (scripts/run_zeroshot.py: zero_shot_scores) and E5 (scripts/run_fewshot.py:
sample_k_shot, linear_probe). Report sections: Methods, "CLIP zero-shot and prompt engineering"
and "Few-shot linear probe".
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def zero_shot_scores(image_feats: np.ndarray, text_feats: np.ndarray) -> np.ndarray:
    """Cosine similarity (features are already L2-normalised) scaled like CLIP.

    image_feats (n, D) and text_feats (C, D) in, scores (n, C) out. The best-scoring class is the prediction."""
    # Both inputs have length 1, so the dot product equals cosine similarity. 100 is CLIP's logit scale.
    return 100.0 * image_feats @ text_feats.T  # (n, D) @ (D, C) -> (n, C)


def sample_k_shot(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Indices of k random examples per class (all examples if k <= 0 or class is smaller)."""
    # Own random generator from the seed, so the same seed always picks the same shots.
    rng = np.random.default_rng(seed)
    idx = []
    # For every class: find its rows and pick k of them at random (without repeats).
    for c in np.unique(y):
        members = np.nonzero(y == c)[0]  # row numbers of class c
        if 0 < k < len(members):
            members = rng.choice(members, size=k, replace=False)
        idx.extend(sorted(members.tolist()))
    # Return all chosen row numbers in increasing order.
    return np.asarray(sorted(idx))


def linear_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int,
                 C: float = 0.316, seed: int = 0, class_weight: str | dict | None = None) -> np.ndarray:
    """Multinomial logistic regression; returns (n_test, n_classes) class probabilities.

    Trained on frozen CLIP image features: CLIP itself is not changed, only this linear classifier is learned.
    C is the inverse regularisation strength (smaller C = stronger L2 penalty, less overfitting on few shots).
    class_weight: None (default, no weights) or "balanced" (passed to sklearn; weight 1 for every class
    when all classes have the same number of training images)."""
    # Fit logistic regression on the training features and labels.
    clf = LogisticRegression(C=C, max_iter=3000, random_state=seed, class_weight=class_weight)
    clf.fit(x_train, y_train)
    # Put the probabilities into a full (n_test, n_classes) table. clf.classes_ are the classes seen in
    # training; a class never seen keeps probability 0, so columns always match class indices.
    proba = np.zeros((len(x_test), n_classes), dtype=np.float32)
    proba[:, clf.classes_] = clf.predict_proba(x_test)
    return proba
