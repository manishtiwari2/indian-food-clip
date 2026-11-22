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
    # text_feats.T flips (C, D) to (D, C); "@" is matrix multiplication, so entry [i, c] is the dot product of
    # image i with class c's text vector. Multiplying by 100.0 only stretches the scores; it does not change
    # which class is best (the argmax stays the same).
    return 100.0 * image_feats @ text_feats.T  # (n, D) @ (D, C) -> (n, C)


def sample_k_shot(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Indices of k random examples per class (all examples if k <= 0 or class is smaller).

    Input: training labels y, shots per class k, seed. Output: sorted row numbers into the training set.
    Why: few-shot (E5) asks how accuracy grows with k labelled images per class; the seed makes each
    draw repeatable and different seeds show how much the choice of shots matters.
    Used by: scripts/run_fewshot.py."""
    # Own random generator from the seed, so the same seed always picks the same shots.
    # np.random.default_rng(seed) makes a numpy Generator that does not touch numpy's global random state.
    rng = np.random.default_rng(seed)
    # Python list that collects the chosen row numbers of all classes.
    idx = []
    # For every class: find its rows and pick k of them at random (without repeats).
    # np.unique(y) = the sorted distinct labels in y, one entry per class present.
    for c in np.unique(y):
        # y == c is a boolean array (n,); np.nonzero(...)[0] turns it into the row numbers where it is True.
        members = np.nonzero(y == c)[0]  # row numbers of class c
        # Subsample only when k is positive and smaller than the class size; otherwise keep every row of the class.
        if 0 < k < len(members):
            # rng.choice(..., size=k, replace=False) draws k distinct row numbers at random from members.
            members = rng.choice(members, size=k, replace=False)
        # .tolist() converts numpy ints to Python ints; sorted() orders them; extend() appends them all to idx.
        idx.extend(sorted(members.tolist()))
    # Return all chosen row numbers in increasing order, as an int numpy array of shape (n_selected,).
    return np.asarray(sorted(idx))


def linear_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int,
                 C: float = 0.316, seed: int = 0, class_weight: str | dict | None = None) -> np.ndarray:
    """Multinomial logistic regression; returns (n_test, n_classes) class probabilities.

    Trained on frozen CLIP image features: CLIP itself is not changed, only this linear classifier is learned.
    C is the inverse regularisation strength (smaller C = stronger L2 penalty, less overfitting on few shots).
    The default 0.316 was set by the lead; reason not recorded in the repo.
    class_weight: None (default, no weights) or "balanced" (passed to sklearn; weight 1 for every class
    when all classes have the same number of training images)."""
    # Fit logistic regression on the training features and labels.
    # LogisticRegression arguments (sklearn):
    #   C            inverse L2 regularisation strength (see docstring);
    #   max_iter     upper limit on optimiser iterations before sklearn stops (and warns if not converged);
    #                the value 3000: reason not recorded in the repo;
    #   random_state seed for any randomness inside the solver, taken from the seed argument;
    #   class_weight None = every class counts equally in the loss, "balanced" = classes re-weighted by frequency.
    # Not set here, so sklearn defaults apply (default solver and penalty, see the sklearn documentation).
    clf = LogisticRegression(C=C, max_iter=3000, random_state=seed, class_weight=class_weight)
    # fit learns one weight vector per class from x_train (n_train, D) and y_train (n_train,) integer labels.
    clf.fit(x_train, y_train)
    # Put the probabilities into a full (n_test, n_classes) table. clf.classes_ are the classes seen in
    # training; a class never seen keeps probability 0, so columns always match class indices.
    # np.zeros makes the (n_test, n_classes) float32 table of zeros.
    proba = np.zeros((len(x_test), n_classes), dtype=np.float32)
    # predict_proba(x_test) gives (n_test, len(clf.classes_)) probabilities (each row sums to 1) in the order of
    # clf.classes_; indexing the columns with clf.classes_ writes each into its class's column.
    proba[:, clf.classes_] = clf.predict_proba(x_test)
    return proba
