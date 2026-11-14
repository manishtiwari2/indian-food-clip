"""All report figures + E6 gap table, built only from results/*.csv (never from memory).

Writes report/figures/*.pdf and results/gap.csv. Each figure is skipped if its CSV is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.io import write_rows  # noqa: E402

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"skip: {path.name} not found")
        return None
    return pd.read_csv(path, encoding="utf-8")


def fig_bias(zs: pd.DataFrame, out: Path) -> None:
    sub = zs[zs["template"] == "food"]
    bars = [("Indian: English description", (sub.dataset == "indian") & (sub.language == "english")),
            ("Indian: romanised Hindi name", (sub.dataset == "indian") & (sub.language == "hindi_roman")),
            ("Food-101: English name", (sub.dataset == "food101") & (sub.language == "english"))]
    models = list(dict.fromkeys(sub["model"]))
    w = 0.8 / len(bars)
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, (label, mask) in enumerate(bars):
        vals = [sub[mask & (sub.model == m)]["top1"].mean() for m in models]
        ax.bar(np.arange(len(models)) + i * w, vals, w, label=label)
    ax.set_xticks(np.arange(len(models)) + w, models)
    ax.set_ylabel("zero-shot top-1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Indian vs Western zero-shot accuracy (template: 'a photo of {}, a type of food.')")
    fig.savefig(out / "zeroshot_bias.pdf")
    plt.close(fig)


def fig_language(zs: pd.DataFrame, out: Path) -> None:
    sub = zs[zs["dataset"] == "indian"]
    table = sub.pivot_table(index="model", columns="language", values="top1", aggfunc="max")
    ax = table.plot.bar(figsize=(6, 3), rot=0)
    ax.set_ylabel("best zero-shot top-1 over templates")
    ax.set_ylim(0, 1)
    ax.set_title("Prompt language (Indian test set)")
    ax.figure.savefig(out / "zeroshot_language.pdf")
    plt.close(ax.figure)


def fig_fewshot_and_gap(fsh: pd.DataFrame, zs: pd.DataFrame | None, out: Path, res: Path) -> None:
    fsh = fsh.copy()
    n_all = fsh.loc[fsh.shots == "all", "n_train"].max()
    fsh["k"] = fsh["shots"].map(lambda s: np.nan if s == "all" else int(s))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    gap_rows = []
    for i, (model, g) in enumerate(fsh.groupby("model", sort=False)):
        curve = g.dropna(subset=["k"]).groupby("k")["top1"].agg(["mean", "std"]).reset_index()
        color = f"C{i}"
        ax.errorbar(curve["k"], curve["mean"], yerr=curve["std"].fillna(0), marker="o", color=color,
                    label=model, capsize=2)
        all_acc = g.loc[g.shots == "all", "top1"].mean()
        ax.scatter([curve["k"].max() * 2], [all_acc], marker="*", color=color, s=60)
        west = np.nan
        if zs is not None:
            w = zs[(zs.dataset == "food101") & (zs.model == model) & (zs.template == "food")]["top1"]
            west = w.mean() if len(w) else np.nan
            if not np.isnan(west):
                ax.axhline(west, color=color, ls="--", lw=0.8)
        reached = curve[curve["mean"] >= west]["k"] if not np.isnan(west) else pd.Series(dtype=float)
        ind_zs = (zs[(zs.dataset == "indian") & (zs.model == model)]["top1"].max() if zs is not None else np.nan)
        gap_rows.append({"model": model, "western_zero_shot_top1": west, "indian_best_zero_shot_top1": ind_zs,
                         "indian_all_shot_probe_top1": all_acc,
                         "shots_to_reach_western": int(reached.min()) if len(reached) else "not reached"})
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"labelled images per class (star = all {int(n_all)} training images)")
    ax.set_ylabel("top-1 (Indian test)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Few-shot linear probe; dashed = same model's Food-101 zero-shot")
    fig.savefig(out / "fewshot_curve.pdf")
    plt.close(fig)
    write_rows(gap_rows, res / "gap.csv")
    print(pd.DataFrame(gap_rows).to_string(index=False))


def fig_worst_classes(pc: pd.DataFrame, out: Path, n: int = 20) -> None:
    sub = pc[(pc.dataset == "indian") & (pc.template == "food") & (pc.language == "hindi_roman")]
    if sub.empty:
        return
    model = sub["model"].iloc[-1]  # last model in config order
    worst = sub[sub.model == model].nsmallest(n, "accuracy")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(worst["class"], worst["accuracy"])
    ax.invert_yaxis()
    ax.set_xlabel("zero-shot accuracy")
    ax.set_title(f"{n} hardest Indian dishes ({model}, romanised Hindi name)")
    fig.savefig(out / "worst_classes.pdf")
    plt.close(fig)


def fig_cnn(hist: pd.DataFrame, out: Path) -> None:
    run = hist["run"].iloc[-1]
    h = hist[hist.run == run]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h.epoch, h.train_acc, label="train acc")
    ax.plot(h.epoch, h.val_acc, label="val acc")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"CNN from scratch (run: {run})")
    fig.savefig(out / "cnn_curves.pdf")
    plt.close(fig)


def main() -> None:
    cfg = config_from_args(base_parser(__doc__).parse_args())
    res, out = resolve(cfg["paths"]["results"]), resolve(cfg["paths"]["figures"])
    out.mkdir(parents=True, exist_ok=True)
    zs = _read(res / "zeroshot.csv")
    if zs is not None:
        fig_bias(zs, out)
        fig_language(zs, out)
    if (fsh := _read(res / "fewshot.csv")) is not None:
        fig_fewshot_and_gap(fsh, zs, out, res)
    if (pc := _read(res / "zeroshot_per_class.csv")) is not None:
        fig_worst_classes(pc, out)
    if (hist := _read(res / "cnn_history.csv")) is not None:
        fig_cnn(hist, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
