import numpy as np
import pytest
import torch

from foodclip.cnn import SmallResNet, predict, train_one_epoch
from foodclip.config import load_config
from foodclip.fewshot import linear_probe, sample_k_shot, zero_shot_scores
from foodclip.metrics import per_class_accuracy, summarize, top_confusions, topk_accuracy
from foodclip.prompts import TEMPLATES, build_prompts, valid_combo


def test_topk_and_summary():
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    y = np.array([0, 1, 2])
    assert topk_accuracy(scores, y, 1) == pytest.approx(1 / 3)
    assert topk_accuracy(scores, y, 2) == pytest.approx(2 / 3)
    assert topk_accuracy(scores, y, 5) == 1.0  # k larger than classes is clipped
    s = summarize(scores, y)
    assert s["n_test"] == 3 and 0 <= s["macro_f1"] <= 1


def test_per_class_and_confusions():
    y, pred = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    pc = per_class_accuracy(y, pred, ["a", "b"])
    assert dict(zip(pc["class"], pc["accuracy"])) == {"a": 0.5, "b": 1.0}
    conf = top_confusions(y, pred, ["a", "b"])
    assert conf.iloc[0].to_dict() == {"true": "a", "predicted": "b", "count": 1}


def test_k_shot_sampling():
    y = np.repeat(np.arange(5), 10)
    idx = sample_k_shot(y, 3, seed=1)
    assert np.bincount(y[idx]).tolist() == [3] * 5
    assert np.array_equal(idx, sample_k_shot(y, 3, seed=1))
    assert len(sample_k_shot(y, 0, seed=1)) == 50  # 0 = all


def test_linear_probe_and_zero_shot_on_separable_features():
    rng = np.random.default_rng(0)
    centers = np.eye(4, 16)
    y = np.repeat(np.arange(4), 20)
    x = centers[y] + 0.05 * rng.standard_normal((80, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    proba = linear_probe(x[::2], y[::2], x[1::2], n_classes=4, C=10)
    assert topk_accuracy(proba, y[1::2], 1) > 0.95
    assert topk_accuracy(zero_shot_scores(x, centers), y, 1) > 0.95


def test_prompts():
    p = build_prompts(["poha", "jalebi"], "ensemble")
    assert len(p) == 2 and len(p[0]) == len(TEMPLATES["ensemble"]) and "poha" in p[0][0]
    assert not valid_combo("english", "hindi_native")
    assert valid_combo("hindi_devanagari", "hindi_native")
    with pytest.raises(KeyError):
        build_prompts(["x"], "nope")


def test_cnn_forward_and_one_step():
    torch.manual_seed(0)
    model = SmallResNet(n_classes=4)
    x, y = torch.randn(8, 3, 64, 64), torch.randint(0, 4, (8,))
    assert model(x).shape == (8, 4)
    loader = [(x, y)]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, _ = train_one_epoch(model, loader, opt, torch.device("cpu"))
    scores, ys = predict(model, loader, torch.device("cpu"))
    assert np.isfinite(loss) and scores.shape == (8, 4) and len(ys) == 8


def test_config_overrides():
    cfg = load_config(overrides=["cnn.epochs=3", "paths.indian_raw=/tmp/x", "fewshot.shots=[1,2]"])
    assert cfg["cnn"]["epochs"] == 3
    assert cfg["paths"]["indian_raw"] == "/tmp/x"
    assert cfg["fewshot"]["shots"] == [1, 2]


def test_color_jitter_off_is_unchanged_and_on_changes_brightness():
    import random

    from PIL import Image

    from foodclip.data import SimpleTransform

    im = Image.new("RGB", (64, 64), (120, 120, 120))
    random.seed(0)
    off = SimpleTransform(32, True, 0.0)(im)
    random.seed(0)
    off_again = SimpleTransform(32, True, 0.0)(im)
    assert torch.equal(off, off_again)
    random.seed(0)
    on = SimpleTransform(32, True, 0.5)(im)
    assert on.shape == off.shape and not torch.equal(on, off)
    # validation transform ignores jitter
    assert torch.equal(SimpleTransform(32, False, 0.5)(im), SimpleTransform(32, False, 0.0)(im))
