"""End-to-end smoke run of the scripts on synthetic data (CPU, < 1 min).
CLIP features are faked so open_clip is not needed.

Runs prepare_data, train_cnn, run_fewshot, run_zeroshot and make_figures in order,
and checks that each one writes the expected CSVs and figures.

What it proves: the scripts work together, not just the library functions. Each script is started as a
separate process (like a user would), reads what the previous one wrote, and writes into a temporary folder,
so the real splits/ and results/ folders are never touched. It does not prove CLIP accuracy; the fake
features are built to be easy, so it only proves the plumbing (paths, columns, class order, file names).
Synthetic data: the fake_indian fixture from conftest.py (4 classes x 12 images, 1 duplicate, 1 corrupt).
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import CLASSES, ROOT


def run(script, *overrides, env=None):
    """Run one script with config overrides as a new process; fail the test if it crashes.

    script = file name in scripts/; overrides = "key=value" strings for --set; env = optional environment
    (used to make the script import the fake open_clip). Returns the script's printed output.
    """
    # sys.executable = the same Python that runs pytest, so installed packages are the same.
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--set", *overrides]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    # Exit code 0 means success; on failure the message shows the script's output to help debugging.
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


# Stand-in for open_clip: the text "embedding" of a prompt is the one-hot vector of the class whose
# name (any language in data/class_names.csv) it contains, so it lines up with the fake image features.
# This string is written to disk as a Python package named open_clip (see fake_open_clip_env). The %r
# placeholders are filled in with the csv path and the class list. If the scripts' text-to-class lookup,
# class order or scoring were wrong, zero-shot accuracy would drop, which the test checks for.
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
    """Write the fake open_clip package to disk and return an env that imports it first.

    PYTHONPATH makes Python look in our temp folder before site-packages, so `import open_clip` in the
    script finds the fake one even if the real open_clip is installed.
    """
    pkg = tmp_path / "fake_clip" / "open_clip"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        FAKE_OPEN_CLIP % (str(ROOT / "data" / "class_names.csv"), CLASSES, CLASSES), encoding="utf-8")
    return {**os.environ, "PYTHONPATH": str(pkg.parent)}


def test_pipeline_end_to_end(fake_indian, tmp_path):
    """All scripts run one after another on fake data and produce correct outputs.

    One long test on purpose: each step needs the files written by the step before it, the same order
    as the "How to run" list in CLAUDE.md. `o` is the shared override list that redirects all paths.
    """
    # Send every input and output path to the temp folder, and use a tiny split with exact dedup.
    o = [f"paths.indian_raw={fake_indian.as_posix()}", f"paths.splits={(tmp_path / 'splits').as_posix()}",
         f"paths.results={(tmp_path / 'results').as_posix()}", f"paths.features={(tmp_path / 'feat').as_posix()}",
         f"paths.figures={(tmp_path / 'fig').as_posix()}", f"paths.checkpoints={(tmp_path / 'ckpt').as_posix()}",
         "split.test_per_class=3", "split.dedup_max_distance=0"]

    # Step 1: clean and split. The duplicate and broken file are removed; 4 classes x 3 test images.
    out = run("prepare_data.py", *o)
    assert "removed 2" in out
    split = pd.read_csv(tmp_path / "splits" / "indian_split.csv")
    assert len(split) == 48 and (split.split == "test").sum() == 12

    # Step 2: train the CNN for one short epoch and check the result columns.
    run("train_cnn.py", *o, "cnn.epochs=1", "cnn.image_size=32", "cnn.batch_size=8", "cnn.num_workers=0")
    cnn = pd.read_csv(tmp_path / "results" / "cnn.csv")
    assert {"top1", "top5", "macro_f1", "run_at"} <= set(cnn.columns)

    # Step 3: save fake CLIP features (class one-hot plus noise) for the first model in the config.
    # This replaces extract_features.py, which needs real CLIP. File name = dataset__model__pretrained.npz,
    # the pattern the real cache uses. relpath lets the scripts match each feature row to its image.
    rng = np.random.default_rng(0)
    y = split["label"].astype("category").cat.codes.to_numpy()
    feats = np.eye(4, 32)[y] + 0.1 * rng.standard_normal((len(y), 32))
    (tmp_path / "feat").mkdir()
    np.savez(tmp_path / "feat" / "indian__ViT-B-32__openai.npz", feats=feats.astype(np.float32),
             relpath=split["relpath"].to_numpy(dtype=str))
    # Step 4: few-shot probe with 2 seeds; check the number of rows and the class_weight column.
    run("run_fewshot.py", *o, "fewshot.shots=[1,2,0]", "fewshot.seeds=[0,1]")
    fs = pd.read_csv(tmp_path / "results" / "fewshot.csv")
    # 2 seeds x 2 shot levels + 1 "all" (balanced) + 1 extra unweighted "all" reference row
    assert len(fs) == 6
    assert fs["class_weight"].value_counts().to_dict() == {"balanced": 5, "none": 1}
    extra = fs[fs.class_weight == "none"].iloc[0]
    assert str(extra.shots) == "all" and extra.seed == 0

    # Step 5: zero-shot with the fake open_clip; every language/template row must score well.
    run("run_zeroshot.py", *o, env=fake_open_clip_env(tmp_path))
    zs = pd.read_csv(tmp_path / "results" / "zeroshot.csv")
    # 3 languages x 4 templates + hindi_native (Devanagari only); Food-101 skipped (no features)
    assert len(zs) == 13 and set(zs.dataset) == {"indian"}
    assert {"model", "pretrained", "language", "template", "top1", "top5", "macro_f1", "run_at"} <= set(zs.columns)
    assert (zs.top1 > 0.9).all()  # class order, name lookup and scoring line up in every language
    pc = pd.read_csv(tmp_path / "results" / "zeroshot_per_class.csv")
    assert len(pc) == 13 * len(CLASSES)
    assert (tmp_path / "results" / "zeroshot_confusions.csv").exists()

    # Step 6: make figures and the gap table; the gap uses only the balanced "all" rows.
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
