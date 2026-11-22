"""Unit tests for metrics, few-shot probe, prompts, CNN, CLIP fine-tune head and config.

All inputs are tiny hand-made arrays or random tensors, so the tests run on CPU in seconds.

Why: these are the building blocks behind every reported number. Each test uses inputs small enough
that the right answer can be worked out by hand, so a failing test means a real bug in the formula.
No real images or CLIP weights are used. Unlike the smoke test, these call library functions directly.
"""
import numpy as np
import pytest
import torch

from foodclip.cnn import SmallResNet, predict, train_one_epoch
from foodclip.config import load_config
from foodclip.fewshot import linear_probe, sample_k_shot, zero_shot_scores
from foodclip.metrics import per_class_accuracy, summarize, top_confusions, topk_accuracy
from foodclip.prompts import TEMPLATES, build_prompts, valid_combo


def test_topk_and_summary():
    """Top-k accuracy is correct for small k and for k larger than the number of classes.

    Proves: the main metric is right. Top-k = the true class is among the k highest-scored classes.
    Why it matters: top-1 and top-5 are the headline numbers of every experiment.
    """
    # 3 images, 3 classes: only the first image has the right class ranked first.
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    y = np.array([0, 1, 2])
    assert topk_accuracy(scores, y, 1) == pytest.approx(1 / 3)
    assert topk_accuracy(scores, y, 2) == pytest.approx(2 / 3)
    assert topk_accuracy(scores, y, 5) == 1.0  # k larger than classes is clipped
    # The summary row counts test images and gives a macro F1 between 0 and 1.
    s = summarize(scores, y)
    assert s["n_test"] == 3 and 0 <= s["macro_f1"] <= 1


def test_per_class_and_confusions():
    """Per-class accuracy and the most common mistake are computed correctly.

    Proves: the tables behind the "hardest dishes" figures and the confusion analysis are right.
    y = true labels, pred = predicted labels (not scores).
    """
    # Class "a" is right once out of twice; class "b" is always right.
    y, pred = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    pc = per_class_accuracy(y, pred, ["a", "b"])
    assert dict(zip(pc["class"], pc["accuracy"])) == {"a": 0.5, "b": 1.0}
    # The top confusion is "a" predicted as "b", seen once.
    conf = top_confusions(y, pred, ["a", "b"])
    assert conf.iloc[0].to_dict() == {"true": "a", "predicted": "b", "count": 1}


def test_k_shot_sampling():
    """k-shot sampling takes exactly k images per class and is repeatable with a seed.

    Proves: the few-shot experiment (E5) gives every class the same number of labelled examples, and
    that a seed fixes which ones are picked (different seeds are used to average out luck).
    """
    # 5 classes with 10 images each; ask for 3 per class.
    y = np.repeat(np.arange(5), 10)
    idx = sample_k_shot(y, 3, seed=1)
    assert np.bincount(y[idx]).tolist() == [3] * 5
    # Same seed gives the same images, and k=0 means use every image.
    assert np.array_equal(idx, sample_k_shot(y, 3, seed=1))
    assert len(sample_k_shot(y, 0, seed=1)) == 50  # 0 = all


def test_linear_probe_and_zero_shot_on_separable_features():
    """Both the linear probe and zero-shot scoring are near perfect on easy features.

    Proves: the two CLIP-feature classifiers work end to end. Linear probe = logistic regression trained
    on image features; zero-shot = compare image features to class text embeddings with no training.
    Features are unit length, as real CLIP features are after normalisation.
    """
    # Build 4 well separated clusters of unit-length 16-d features.
    rng = np.random.default_rng(0)
    centers = np.eye(4, 16)
    y = np.repeat(np.arange(4), 20)
    x = centers[y] + 0.05 * rng.standard_normal((80, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    # Train the probe on even rows and test on odd rows.
    proba = linear_probe(x[::2], y[::2], x[1::2], n_classes=4, C=10)
    assert topk_accuracy(proba, y[1::2], 1) > 0.95
    # Zero-shot: the cluster centres act as the class text embeddings.
    assert topk_accuracy(zero_shot_scores(x, centers), y, 1) > 0.95


def test_class_weight_reaches_logistic_regression(monkeypatch):
    """The class_weight, C and max_iter settings are passed to scikit-learn unchanged.

    Proves: the probe's settings really reach LogisticRegression. A "spy" (fake class) is swapped in with
    monkeypatch, so we check the arguments without training a real model.
    """
    seen = []

    # Fake LogisticRegression that only records the arguments it was built with.
    class Spy:
        def __init__(self, **kw):
            seen.append(kw)
            self.classes_ = np.arange(2)

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5)

    # Swap in the spy, then call the probe once without and once with balanced weights.
    monkeypatch.setattr("foodclip.fewshot.LogisticRegression", Spy)
    x, y = np.zeros((4, 3)), np.array([0, 0, 1, 1])
    linear_probe(x, y, x, n_classes=2)
    linear_probe(x, y, x, n_classes=2, class_weight="balanced")
    # The spy saw the right weights and the default C and max_iter values.
    assert seen[0]["class_weight"] is None and seen[1]["class_weight"] == "balanced"  # default = old behaviour
    assert seen[0]["C"] == 0.316 and seen[0]["max_iter"] == 3000


def test_balanced_weights_do_nothing_when_classes_are_equal_and_matter_when_not():
    """Balanced weights change nothing for equal class sizes but change results for unequal ones.

    Proves: class_weight="balanced" behaves as theory says (weights are all equal when classes are equal),
    so the k-shot rows are unaffected while the "all images" row, where class sizes differ, is corrected.
    """
    # Equal class sizes: plain and balanced probes give exactly the same output.
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(4), 16)  # exactly k=16 images per class, like the k-shot rows
    x = np.eye(4, 8)[y] + 0.5 * rng.standard_normal((64, 8))
    plain = linear_probe(x, y, x, n_classes=4)
    assert np.array_equal(plain, linear_probe(x, y, x, n_classes=4, class_weight="balanced"))
    # Unequal class sizes: the two probes give different outputs.
    keep = np.r_[np.arange(16), np.arange(16, 32), np.arange(32, 48), np.arange(48, 54)]  # last class uneven
    a = linear_probe(x[keep], y[keep], x, n_classes=4)
    b = linear_probe(x[keep], y[keep], x, n_classes=4, class_weight="balanced")
    assert not np.array_equal(a, b)


def test_fewshot_class_weight_in_config():
    """The config default is balanced, and it can be switched off with null from the CLI.

    Proves: configs/default.yaml and the --set override agree with what the few-shot script expects.
    """
    assert load_config()["fewshot"]["class_weight"] == "balanced"
    assert load_config(overrides=["fewshot.class_weight=null"])["fewshot"]["class_weight"] is None


def test_prompts():
    """Prompts are built per class and template, and bad language/template pairs are rejected.

    Proves: the text fed to CLIP is built correctly. A prompt is a sentence such as "a photo of poha";
    the "ensemble" template makes several sentences per class whose embeddings are averaged.
    """
    # The ensemble template gives one prompt per template line, and the prompt contains the class name.
    p = build_prompts(["poha", "jalebi"], "ensemble")
    assert len(p) == 2 and len(p[0]) == len(TEMPLATES["ensemble"]) and "poha" in p[0][0]
    # hindi_native templates are only allowed with Devanagari names.
    assert not valid_combo("english", "hindi_native")
    assert valid_combo("hindi_devanagari", "hindi_native")
    # An unknown template name raises an error.
    with pytest.raises(KeyError):
        build_prompts(["x"], "nope")


def test_cnn_forward_and_one_step():
    """The CNN gives one score per class and can train and predict for one step.

    Proves: the E1 baseline network has the right output shape and the train/predict helpers run
    without error and give a finite loss. It does not check accuracy (random data, one step).
    """
    # A batch of 8 random 64x64 images gives an 8 x 4 score matrix.
    torch.manual_seed(0)
    model = SmallResNet(n_classes=4)
    x, y = torch.randn(8, 3, 64, 64), torch.randint(0, 4, (8,))
    assert model(x).shape == (8, 4)
    # One training epoch on a one-batch "loader", then prediction on the same batch.
    loader = [(x, y)]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, _ = train_one_epoch(model, loader, opt, torch.device("cpu"))
    scores, ys = predict(model, loader, torch.device("cpu"))
    assert np.isfinite(loss) and scores.shape == (8, 4) and len(ys) == 8


def test_clip_classifier_starts_as_zero_shot_and_trains():
    """The fine-tune model first gives the same scores as zero-shot, then its encoder learns.

    Proves two things for E7: (1) the head built from text embeddings reproduces zero-shot scores, so
    "epoch 0 = zero-shot" is true; (2) gradients reach the image encoder, so it is really being fine-tuned.
    """
    from foodclip.finetune import ClipClassifier

    # A tiny linear layer plays the image encoder; 3 random unit vectors play the class text embeddings.
    torch.manual_seed(0)
    visual = torch.nn.Linear(6, 4)  # stands in for the CLIP image encoder
    text = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    model = ClipClassifier(visual, text, logit_scale=100.0)
    x, y = torch.randn(5, 6), torch.tensor([0, 1, 2, 0, 1])
    # Before training, the model output equals the zero-shot scores.
    with torch.no_grad():
        f = visual(x).numpy()
        expected = zero_shot_scores(f / np.linalg.norm(f, axis=1, keepdims=True), text)
        assert np.allclose(model(x).numpy(), expected, atol=1e-4)
    # After one training step, the encoder weights have changed.
    before = visual.weight.detach().clone()
    loss, _ = train_one_epoch(model, [(x, y)], torch.optim.AdamW(model.parameters(), lr=1e-2), torch.device("cpu"))
    assert np.isfinite(loss) and not torch.equal(before, visual.weight)


def test_config_overrides():
    """`--set key=value` overrides work for numbers, paths and lists.

    Proves: values typed on the command line are parsed to the right type (int, string, list), which is
    how Kaggle paths and short smoke-test settings are passed without editing the YAML file.
    """
    cfg = load_config(overrides=["cnn.epochs=3", "paths.indian_raw=/tmp/x", "fewshot.shots=[1,2]"])
    assert cfg["cnn"]["epochs"] == 3
    assert cfg["paths"]["indian_raw"] == "/tmp/x"
    assert cfg["fewshot"]["shots"] == [1, 2]


def test_color_jitter_off_is_unchanged_and_on_changes_brightness():
    """Jitter 0 gives repeatable output; jitter above 0 changes the training image.

    Proves: the CNN's colour-jitter augmentation (random brightness/contrast changes) can be turned off,
    is random only when on, and is never applied at validation time. The image is plain grey so any
    change in the output must come from the jitter.
    """
    import random

    from PIL import Image

    from foodclip.data import SimpleTransform

    # A plain grey test image.
    im = Image.new("RGB", (64, 64), (120, 120, 120))
    # With jitter off and the same seed, two runs give the same tensor.
    random.seed(0)
    off = SimpleTransform(32, True, 0.0)(im)
    random.seed(0)
    off_again = SimpleTransform(32, True, 0.0)(im)
    assert torch.equal(off, off_again)
    # With jitter on and the same seed, the tensor differs from the jitter-off one.
    random.seed(0)
    on = SimpleTransform(32, True, 0.5)(im)
    assert on.shape == off.shape and not torch.equal(on, off)
    # validation transform ignores jitter
    assert torch.equal(SimpleTransform(32, False, 0.5)(im), SimpleTransform(32, False, 0.0)(im))
