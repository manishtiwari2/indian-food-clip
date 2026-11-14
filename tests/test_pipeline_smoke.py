"""End-to-end smoke run of the scripts on synthetic data (CPU, < 1 min).
CLIP features are faked so open_clip is not needed."""
import subprocess
import sys

import numpy as np
import pandas as pd
from conftest import ROOT


def run(script, *overrides):
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--set", *overrides]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


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
    assert len(fs) == 5  # 2 seeds x 2 shot levels + 1 "all"

    run("make_figures.py", *o)
    assert (tmp_path / "fig" / "fewshot_curve.pdf").exists()
    assert (tmp_path / "fig" / "cnn_curves.pdf").exists()
    assert (tmp_path / "results" / "gap.csv").exists()
