"""End-to-end smoke run of the scripts on synthetic data (CPU, < 1 min).
CLIP features are faked so open_clip is not needed."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import CLASSES, ROOT


def run(script, *overrides, env=None):
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--set", *overrides]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


# Stand-in for open_clip: the text "embedding" of a prompt is the one-hot vector of the class whose
# name (any language in data/class_names.csv) it contains, so it lines up with the fake image features.
FAKE_OPEN_CLIP = '''
import csv
import torch

_NAMES = {}
with open(%r, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["folder"] in %r:
            idx = sorted(%r).index(row["folder"])
            for col in ("english", "hindi_roman", "hindi_devanagari"):
                _NAMES[row[col]] = idx


class _Model(torch.nn.Module):
    def encode_text(self, tokens):
        return tokens


def create_model_and_transforms(name, pretrained=None, device=None):
    return _Model(), None, None


def get_tokenizer(name):
    def tokenize(texts):
        out = torch.full((len(texts), 32), 0.01)
        for i, t in enumerate(texts):
            for n, idx in _NAMES.items():
                if n in t:
                    out[i, idx] = 1.0
        return out
    return tokenize
'''


def fake_open_clip_env(tmp_path):
    pkg = tmp_path / "fake_clip" / "open_clip"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        FAKE_OPEN_CLIP % (str(ROOT / "data" / "class_names.csv"), CLASSES, CLASSES), encoding="utf-8")
    return {**os.environ, "PYTHONPATH": str(pkg.parent)}


def test_pipeline_end_to_end(fake_indian, tmp_path):
    o = [f"paths.indian_raw={fake_indian.as_posix()}", f"paths.splits={(tmp_path / 'splits').as_posix()}",
         f"paths.results={(tmp_path / 'results').as_posix()}", f"paths.features={(tmp_path / 'feat').as_posix()}",
         f"paths.figures={(tmp_path / 'fig').as_posix()}", f"paths.checkpoints={(tmp_path / 'ckpt').as_posix()}",
         "split.test_per_class=3", "split.dedup_max_distance=0"]

    out = run("prepare_data.py", *o)
    assert "removed 2" in out
    split = pd.read_csv(tmp_path / "splits" / "indian_split.csv")
    assert len(split) == 48 and (split.split == "test").sum() == 12

    run("train_cnn.py", *o, "cnn.epochs=1", "cnn.image_size=32", "cnn.batch_size=8", "cnn.num_workers=0")
    cnn = pd.read_csv(tmp_path / "results" / "cnn.csv")
    assert {"top1", "top5", "macro_f1", "run_at"} <= set(cnn.columns)

    # fake CLIP features for the first model in the config
    rng = np.random.default_rng(0)
    y = split["label"].astype("category").cat.codes.to_numpy()
    feats = np.eye(4, 32)[y] + 0.1 * rng.standard_normal((len(y), 32))
    (tmp_path / "feat").mkdir()
    np.savez(tmp_path / "feat" / "indian__ViT-B-32__openai.npz", feats=feats.astype(np.float32),
             relpath=split["relpath"].to_numpy(dtype=str))
    run("run_fewshot.py", *o, "fewshot.shots=[1,2,0]", "fewshot.seeds=[0,1]")
    fs = pd.read_csv(tmp_path / "results" / "fewshot.csv")
    # 2 seeds x 2 shot levels + 1 "all" (balanced) + 1 extra unweighted "all" reference row
    assert len(fs) == 6
    assert fs["class_weight"].value_counts().to_dict() == {"balanced": 5, "none": 1}
    extra = fs[fs.class_weight == "none"].iloc[0]
    assert str(extra.shots) == "all" and extra.seed == 0

    run("run_zeroshot.py", *o, env=fake_open_clip_env(tmp_path))
    zs = pd.read_csv(tmp_path / "results" / "zeroshot.csv")
    # 3 languages x 4 templates + hindi_native (Devanagari only); Food-101 skipped (no features)
    assert len(zs) == 13 and set(zs.dataset) == {"indian"}
    assert {"model", "pretrained", "language", "template", "top1", "top5", "macro_f1", "run_at"} <= set(zs.columns)
    assert (zs.top1 > 0.9).all()  # class order, name lookup and scoring line up in every language
    pc = pd.read_csv(tmp_path / "results" / "zeroshot_per_class.csv")
    assert len(pc) == 13 * len(CLASSES)
    assert (tmp_path / "results" / "zeroshot_confusions.csv").exists()

    run("make_figures.py", *o)
    for fig in ("zeroshot_bias.pdf", "zeroshot_language.pdf", "worst_classes.pdf"):
        assert (tmp_path / "fig" / fig).exists()
    assert (tmp_path / "fig" / "fewshot_curve.pdf").exists()
    assert (tmp_path / "fig" / "cnn_curves.pdf").exists()
    gap = pd.read_csv(tmp_path / "results" / "gap.csv")
    balanced_all = fs[(fs.shots.astype(str) == "all") & (fs.class_weight == "balanced")].top1.mean()
    assert gap["indian_all_shot_probe_top1"].iloc[0] == pytest.approx(balanced_all)  # extra row not used

    # an old fewshot.csv without the class_weight column must still work
    fs.drop(columns="class_weight").iloc[:5].to_csv(tmp_path / "results" / "fewshot.csv", index=False)
    run("make_figures.py", *o)
    assert (tmp_path / "results" / "gap.csv").exists()
