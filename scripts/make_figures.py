"""All report figures + E6 gap table, built only from results/*.csv (never from memory).

What it does: reads the result CSVs written by the other scripts and draws every figure used in the report.
It also builds the E6 gap table: Western (Food-101) zero-shot vs Indian zero-shot vs the Indian linear
probe, and how many shots per class the probe needs to reach the Western zero-shot accuracy.
Experiment: E6 (gap table) plus the figures for E1, E2-E4 and E5.
Figures: cnn_curves, cnn_worst_classes ("CNN baseline"); zeroshot_language ("Zero-shot and prompt language");
zeroshot_bias, worst_classes ("Indian vs Western bias"); fewshot_curve ("How many shots close the gap?").

Writes report/figures/*.pdf and results/gap.csv. Each figure is skipped if its CSV is missing.

Reads (all in results/): zeroshot.csv, zeroshot_per_class.csv (from run_zeroshot.py), fewshot.csv (from
run_fewshot.py), cnn.csv, cnn_history.csv, cnn_per_class.csv (from train_cnn.py). Run it last.

Why it only reads CSVs: every number in a figure must come from a real run, so this script never
recomputes or types in a result. If a CSV is missing the figure is skipped instead of faked.

Vocabulary: top-1 = the right class is the model's first guess; "shots" = labelled training images per
class; "gap" = Western (Food-101) zero-shot top-1 compared with Indian accuracy for the same model;
gap.csv lists the numbers side by side (there is no difference column).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

# Use a non-GUI backend so figures can be saved on a server or Kaggle without a screen.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.io import write_rows  # noqa: E402

# Shared figure style: small font, sharp output, tight borders around each saved figure.
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def _read(path: Path) -> pd.DataFrame | None:
    """Read a result CSV, or return None (with a message) if it does not exist yet."""
    if not path.exists():
        print(f"skip: {path.name} not found")
        return None
    return pd.read_csv(path, encoding="utf-8")


def fig_bias(zs: pd.DataFrame, out: Path) -> None:
    """Bar chart of zero-shot top-1 per model: Indian (two name languages) vs Food-101, same 'food' template.

    Input zs = rows of results/zeroshot.csv; out = figures folder. Output: zeroshot_bias.pdf.
    This is the main "Indian vs Western" picture. Taller Food-101 bars mean a higher accuracy on Food-101;
    the two datasets differ (how they were built, Asian dishes in Food-101), so this is not a pure cuisine
    effect (see the report). Devanagari is left out here; fig_language covers all languages.
    """
    # Use only the "food" template, which is the same wording for both datasets (fair comparison).
    sub = zs[zs["template"] == "food"]
    # One bar group per (label, row mask) pair; a mask is a True/False column that selects matching CSV rows.
    bars = [("Indian: English description", (sub.dataset == "indian") & (sub.language == "english")),
            ("Indian: romanised Hindi name", (sub.dataset == "indian") & (sub.language == "hindi_roman")),
            ("Food-101: English name", (sub.dataset == "food101") & (sub.language == "english"))]
    models = list(dict.fromkeys(sub["model"]))  # unique model names, keeping first-seen order
    w = 0.8 / len(bars)  # bar width so each model's group fills 80% of its slot
    fig, ax = plt.subplots(figsize=(6, 3))  # figure size in inches (width, height)
    # Draw one set of bars per group, shifted sideways by i * w.
    for i, (label, mask) in enumerate(bars):
        # One value per model; .mean() averages if the CSV has more than one row for the same setting.
        vals = [sub[mask & (sub.model == m)]["top1"].mean() for m in models]
        ax.bar(np.arange(len(models)) + i * w, vals, w, label=label)
    # Axis labels, legend outside the plot, title, then save.
    ax.set_xticks(np.arange(len(models)) + w, models)  # tick under the middle bar
    ax.set_ylabel("zero-shot top-1")
    ax.set_ylim(0, 1)  # accuracy is a fraction between 0 and 1; a fixed range keeps figures comparable
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Indian vs Western zero-shot accuracy\n(template: 'a photo of {}, a type of food.')", fontsize=9)
    fig.savefig(out / "zeroshot_bias.pdf")  # PDF is vector graphics, so it stays sharp in the LaTeX report
    plt.close(fig)  # free the figure's memory


def fig_language(zs: pd.DataFrame, out: Path) -> None:
    """Bar chart of the best zero-shot top-1 per model and prompt language on the Indian test set.

    Input zs = results/zeroshot.csv rows. Output: zeroshot_language.pdf. Shows whether English,
    romanised Hindi or Devanagari names work best. "Best over templates" means each bar uses the template
    with the highest top-1 on the test set. This is a post-hoc choice and favours zero-shot (optimistic).
    """
    # Table: rows = model, columns = language, value = best top-1 over all templates.
    sub = zs[zs["dataset"] == "indian"]
    table = sub.pivot_table(index="model", columns="language", values="top1", aggfunc="max")
    # Plot the table as grouped bars and save.
    ax = table.plot.bar(figsize=(6, 3), rot=0)  # rot=0 keeps the model names horizontal
    ax.set_ylabel("best zero-shot top-1 over templates")
    ax.set_ylim(0, 1)
    ax.set_title("Prompt language (Indian test set)")
    ax.figure.savefig(out / "zeroshot_language.pdf")
    plt.close(ax.figure)


def fig_fewshot_and_gap(fsh: pd.DataFrame, zs: pd.DataFrame | None, out: Path, res: Path) -> None:
    """Draw the few-shot curve per model and write the E6 gap table (results/gap.csv).

    Inputs: fsh = results/fewshot.csv rows (linear probe, columns include shots, seed, top1, n_train);
    zs = zero-shot rows (or None); out = figures folder; res = results folder.
    Outputs: fewshot_curve.pdf and gap.csv. Answers the project question "how many labelled shots
    close the Indian-vs-Western gap?": the answer is the smallest k in 1-16 whose mean accuracy reaches the
    model's Food-101 zero-shot accuracy, or "not reached". The "all" point is not checked.
    """
    # Keep only the class-weighted rows when they exist, so the curve uses one setting.
    fsh = fsh.copy()
    if "class_weight" in fsh.columns and (fsh["class_weight"] == "balanced").any():
        fsh = fsh[fsh["class_weight"] == "balanced"].copy()  # drop the extra unweighted "all" reference row
    n_all =fsh.loc[fsh.shots == "all", "n_train"].max()  # number of training images in the "all" run
    fsh["k"] = fsh["shots"].map(lambda s: np.nan if s == "all" else int(s))  # numeric shots; "all" -> NaN
    fig, ax = plt.subplots(figsize=(6, 3.2))
    gap_rows = []  # one dict per model; becomes results/gap.csv
    # One curve (and one gap-table row) per model, each in its own colour.
    for i, (model, g) in enumerate(fsh.groupby("model", sort=False)):
        # Mean and std of top-1 over seeds for each k; error bars show the spread across seeds.
        curve = g.dropna(subset=["k"]).groupby("k")["top1"].agg(["mean", "std"]).reset_index()
        color = f"C{i}"
        ax.errorbar(curve["k"], curve["mean"], yerr=curve["std"].fillna(0), marker="o", color=color,
                    label=model, capsize=2)
        # The "all training images" result is drawn as a star to the right of the largest k.
        all_acc = g.loc[g.shots == "all", "top1"].mean()
        ax.scatter([curve["k"].max() * 2], [all_acc], marker="*", color=color, s=60)
        # Same model's Food-101 zero-shot top-1 ("food" template) is the Western target; draw it as a dashed line.
        west = np.nan
        if zs is not None:
            w = zs[(zs.dataset == "food101") & (zs.model == model) & (zs.template == "food")]["top1"]
            west = w.mean() if len(w) else np.nan
            if not np.isnan(west):
                ax.axhline(west, color=color, ls="--", lw=0.8)
        # Shot counts whose mean accuracy reaches the Western target (empty if there is no target).
        # (An old fewshot.csv without the class_weight column also works: the check above is skipped.)
        reached = curve[curve["mean"] >= west]["k"] if not np.isnan(west) else pd.Series(dtype=float)
        # Best Indian zero-shot top-1 over all languages and templates for this model. The best setting is
        # picked on the test set (post-hoc), so this number is optimistic for zero-shot.
        ind_zs = (zs[(zs.dataset == "indian") & (zs.model == model)]["top1"].max() if zs is not None else np.nan)
        # One gap-table row: smallest k that reaches the target, or "not reached".
        gap_rows.append({"model": model, "western_zero_shot_top1": west, "indian_best_zero_shot_top1": ind_zs,
                         "indian_all_shot_probe_top1": all_acc,
                         "shots_to_reach_western": int(reached.min()) if len(reached) else "not reached"})
    # Log-2 x axis so small and large shot counts are both easy to read; then labels and save.
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"labelled images per class (star = all {int(n_all)} training images)")
    ax.set_ylabel("top-1 (Indian test)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Few-shot linear probe; dashed = same model's Food-101 zero-shot")
    fig.savefig(out / "fewshot_curve.pdf")
    plt.close(fig)
    # Save the gap table and print it.
    write_rows(gap_rows, res / "gap.csv")
    print(pd.DataFrame(gap_rows).to_string(index=False))


def fig_worst_classes(pc: pd.DataFrame, zs: pd.DataFrame | None, out: Path, n: int = 20) -> None:
    """Hardest Indian dishes for the model with the best Indian top-1 (romanised Hindi name, 'food' template).

    Inputs: pc = results/zeroshot_per_class.csv (one row per model/setting/class), zs = results/zeroshot.csv
    (used to choose the model), n = how many dishes to show. Output: worst_classes.pdf.
    Shows the dishes with the lowest zero-shot accuracy.
    """
    # Per-class rows for the Indian set with romanised Hindi names and the "food" template.
    sub = pc[(pc.dataset == "indian") & (pc.template == "food") & (pc.language == "hindi_roman")]
    if sub.empty:
        return  # nothing to plot for this setting, so skip the figure
    # Choose the model with the best overall top-1 for this setting (fallback: best mean per-class accuracy).
    if zs is not None:
        best = zs[(zs.dataset == "indian") & (zs.template == "food") & (zs.language == "hindi_roman")]
        model = best.loc[best["top1"].idxmax(), "model"]  # idxmax gives the row label of the highest top-1
    else:
        model = sub.groupby("model")["accuracy"].mean().idxmax()
    worst = sub[sub.model == model].sort_values(["accuracy", "class"]).head(n)  # ties broken by name
    # Horizontal bars, hardest class at the top, then save.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(worst["class"], worst["accuracy"])
    ax.invert_yaxis()  # put the lowest accuracy at the top
    ax.set_xlim(0, 1)
    ax.set_xlabel("zero-shot accuracy")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"{n} hardest Indian dishes ({model}, romanised Hindi name)", fontsize=9)
    fig.savefig(out / "worst_classes.pdf")
    plt.close(fig)


def fig_cnn(hist: pd.DataFrame, out: Path) -> None:
    """Validation accuracy per epoch for every CNN run (train accuracy dashed).

    Input hist = results/cnn_history.csv (columns run, epoch, train_acc, val_acc). Output: cnn_curves.pdf.
    Shows how the from-scratch CNN baseline (E1) learns and whether it overfits.
    """
    fig, ax = plt.subplots(figsize=(6, 3.2))
    # One colour per run: solid line = validation, dashed line = train; a big gap means overfitting.
    for i, (run, h) in enumerate(hist.groupby("run", sort=False)):
        colour = f"C{i}"
        ax.plot(h.epoch, h.val_acc, color=colour, label=f"{run} (val)")
        ax.plot(h.epoch, h.train_acc, color=colour, linestyle="--", alpha=0.5)
    # Labels, legend, title, then save.
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("CNN from scratch: validation (solid) and train (dashed) accuracy")
    fig.savefig(out / "cnn_curves.pdf")
    plt.close(fig)


def fig_cnn_worst(summary: pd.DataFrame, per_class: pd.DataFrame, out: Path, n: int = 15) -> None:
    """Hardest and easiest classes for the CNN run with the best test top-1.

    Inputs: summary = results/cnn.csv (one row per run), per_class = results/cnn_per_class.csv.
    Output: cnn_worst_classes.pdf. Tells us which dishes even a trained-on-this-data CNN finds hard.
    """
    # Pick the best run, sort its classes by accuracy, and take the n lowest and n highest.
    run = summary.loc[summary["top1"].idxmax(), "run"]
    pc = per_class[per_class.run == run].sort_values("accuracy")
    worst, best = pc.head(n), pc.tail(n)
    # Two side-by-side bar charts: hardest (left) and easiest (right).
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8), sharex=True)
    axes[0].barh(worst["class"], worst["accuracy"], color="C3")
    axes[0].set_title(f"{n} hardest classes")
    axes[1].barh(best["class"], best["accuracy"], color="C2")
    axes[1].set_title(f"{n} easiest classes")
    # Same axis settings on both panels.
    for ax in axes:
        ax.set_xlabel("test accuracy")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle(f"CNN per-class accuracy (run: {run})", fontsize=9)
    fig.savefig(out / "cnn_worst_classes.pdf")
    plt.close(fig)


def main() -> None:
    """Build each figure whose result CSV exists, plus the gap table.

    Each block below follows the same pattern: read a CSV (None if missing) and, if present, call the
    matching drawing function. This is why the script can be run at any time, even with partial results.
    """
    # Read config and create the figures folder.
    cfg = config_from_args(base_parser(__doc__).parse_args())
    res, out = resolve(cfg["paths"]["results"]), resolve(cfg["paths"]["figures"])
    out.mkdir(parents=True, exist_ok=True)
    # Zero-shot figures (E2-E4). zs is also reused below to find the Western target and the best model.
    zs = _read(res / "zeroshot.csv")
    if zs is not None:
        fig_bias(zs, out)
        fig_language(zs, out)
    # Few-shot curve and gap table (E5, E6); zs may be None, then the Western line is left out.
    if (fsh := _read(res / "fewshot.csv")) is not None:
        fig_fewshot_and_gap(fsh, zs, out, res)
    # Hardest Indian dishes under zero-shot (E4).
    if (pc := _read(res / "zeroshot_per_class.csv")) is not None:
        fig_worst_classes(pc, zs, out)
    # CNN training curves and per-class bars (E1).
    if (hist := _read(res / "cnn_history.csv")) is not None:
        fig_cnn(hist, out)
        cnn_sum, cnn_pc = _read(res / "cnn.csv"), _read(res / "cnn_per_class.csv")
        if cnn_sum is not None and cnn_pc is not None:
            fig_cnn_worst(cnn_sum, cnn_pc, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
