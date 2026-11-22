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

# matplotlib: plotting library; pyplot is its state-based drawing interface.
import matplotlib  # noqa: E402

# Use a non-GUI backend so figures can be saved on a server or Kaggle without a screen.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
# numpy: array maths (arange, nan). pandas: tables (DataFrame) and reading/writing CSV.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# base_parser/config_from_args: command-line options and config dict. resolve: relative path -> absolute from repo root.
from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
# write_rows: list of dicts -> DataFrame -> CSV file.
from foodclip.io import write_rows  # noqa: E402

# Shared figure style: small font, sharp output, tight borders around each saved figure.
# rcParams.update sets global matplotlib defaults for every figure drawn below. font.size 9 (points) and
# figure.dpi 150: reason not recorded in the repo. savefig.bbox "tight" crops the saved file to the drawn content.
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def _read(path: Path) -> pd.DataFrame | None:
    """Read a result CSV, or return None (with a message) if it does not exist yet."""
    # A missing file is not an error: the caller skips that figure.
    if not path.exists():
        print(f"skip: {path.name} not found")
        return None
    # read_csv returns a DataFrame with one row per CSV line and one column per header (UTF-8 for Devanagari).
    return pd.read_csv(path, encoding="utf-8")


def fig_bias(zs: pd.DataFrame, out: Path) -> None:
    """Bar chart of zero-shot top-1 per model: Indian (two name languages) vs Food-101, same 'food' template.

    Input zs = rows of results/zeroshot.csv; out = figures folder. Output: zeroshot_bias.pdf.
    This is the main "Indian vs Western" picture. Taller Food-101 bars mean a higher accuracy on Food-101;
    the two datasets differ (how they were built, Asian dishes in Food-101), so this is not a pure cuisine
    effect (see the report). Devanagari is left out here; fig_language covers all languages.
    """
    # Use only the "food" template, which is the same wording for both datasets (fair comparison).
    # zs["template"] == "food" is a boolean Series (one value per row); indexing zs with it keeps the True rows.
    sub = zs[zs["template"] == "food"]
    # One bar group per (label, row mask) pair; a mask is a True/False column that selects matching CSV rows.
    bars = [("Indian: English description", (sub.dataset == "indian") & (sub.language == "english")),
            ("Indian: romanised Hindi name", (sub.dataset == "indian") & (sub.language == "hindi_roman")),
            ("Food-101: English name", (sub.dataset == "food101") & (sub.language == "english"))]
    # dict.fromkeys(iterable) makes a dict whose keys are the unique items; dicts keep insertion order, so list()
    # of it gives each model name once, in the order it first appears in the CSV (set() would lose that order).
    models = list(dict.fromkeys(sub["model"]))  # unique model names, keeping first-seen order
    # Each slot is 1 unit wide on the x axis; the len(bars) = 3 bars share 0.8 of it (the 80% value: reason not
    # recorded in the repo).
    w = 0.8 / len(bars)  # bar width so each model's group fills 80% of its slot
    # subplots() with no counts makes one figure with one axes; figsize (6, 3): reason not recorded in the repo.
    fig, ax = plt.subplots(figsize=(6, 3))  # figure size in inches (width, height)
    # Draw one set of bars per group, shifted sideways by i * w.
    for i, (label, mask) in enumerate(bars):
        # One value per model; .mean() averages if the CSV has more than one row for the same setting.
        # mask & (...) is an element-wise AND of two boolean Series; ["top1"] selects that column of the kept rows.
        # vals is a list with one float per model, in the same order as models.
        vals = [sub[mask & (sub.model == m)]["top1"].mean() for m in models]
        # ax.bar(x positions, heights, width): bar i of model j sits at j + i*w; label feeds the legend.
        # np.arange(len(models)) is [0, 1, ..., n_models-1], one slot per model.
        ax.bar(np.arange(len(models)) + i * w, vals, w, label=label)
    # Axis labels, legend outside the plot, title, then save.
    # set_xticks(positions, labels): put the model names at x = slot + w, i.e. under the middle bar of 3.
    ax.set_xticks(np.arange(len(models)) + w, models)  # tick under the middle bar
    ax.set_ylabel("zero-shot top-1")
    ax.set_ylim(0, 1)  # accuracy is a fraction between 0 and 1; a fixed range keeps figures comparable
    # loc="upper left" is the legend corner that is anchored at bbox_to_anchor=(1.01, 1), given in axes
    # coordinates (0..1), i.e. just right of the plot's top-right corner. fontsize 7: reason not recorded in the repo.
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    # Two-line title (the \n in the string is a line break); title fontsize 9: reason not recorded in the repo.
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
    # Keep only the Indian-dataset rows (boolean mask on the dataset column).
    sub = zs[zs["dataset"] == "indian"]
    # pivot_table reshapes long rows into a grid: index="model" -> one row per model, columns="language" -> one column
    # per language, values="top1" is what fills the cells, aggfunc="max" takes the largest top1 when several CSV rows
    # (the templates) fall into the same cell. Result shape (n_models, n_languages); a missing combination is NaN.
    table = sub.pivot_table(index="model", columns="language", values="top1", aggfunc="max")
    # Plot the table as grouped bars and save.
    # DataFrame.plot.bar draws one group of bars per row (model) and one bar per column (language), with a legend;
    # it returns the matplotlib Axes. figsize (6, 3): reason not recorded in the repo.
    ax = table.plot.bar(figsize=(6, 3), rot=0)  # rot=0 keeps the model names horizontal
    ax.set_ylabel("best zero-shot top-1 over templates")
    # Fixed y range 0..1 (accuracy is a fraction), same as the other figures.
    ax.set_ylim(0, 1)
    ax.set_title("Prompt language (Indian test set)")
    # ax.figure is the Figure that owns these axes; save it as a PDF in the figures folder.
    ax.figure.savefig(out / "zeroshot_language.pdf")
    # Close the figure so it is not kept in memory and does not get mixed into the next figure.
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
    # Work on a copy so the caller's DataFrame is not changed by the filtering and the new column below.
    fsh = fsh.copy()
    if "class_weight" in fsh.columns and (fsh["class_weight"] == "balanced").any():
        fsh = fsh[fsh["class_weight"] == "balanced"].copy()  # drop the extra unweighted "all" reference row
    # .loc[row mask, column] gives the n_train values of the "all" rows (one per model); max() takes the largest.
    n_all =fsh.loc[fsh.shots == "all", "n_train"].max()  # number of training images in the "all" run
    # Series.map applies the lambda to every value: "all" -> NaN (not a number), anything else -> int.
    # int(s) also accepts a numeric string such as "4" (the column may hold text because it also contains "all").
    fsh["k"] = fsh["shots"].map(lambda s: np.nan if s == "all" else int(s))  # numeric shots; "all" -> NaN
    # One figure, one axes; figsize (6, 3.2) inches: reason not recorded in the repo.
    fig, ax = plt.subplots(figsize=(6, 3.2))
    gap_rows = []  # one dict per model; becomes results/gap.csv
    # One curve (and one gap-table row) per model, each in its own colour.
    # groupby("model", sort=False) yields (model name, sub-table) pairs in first-seen order, not alphabetical;
    # enumerate adds the counter i, used to pick matplotlib's colour C0, C1, ...
    for i, (model, g) in enumerate(fsh.groupby("model", sort=False)):
        # Mean and std of top-1 over seeds for each k; error bars show the spread across seeds.
        # dropna(subset=["k"]) removes the "all" rows (k is NaN). groupby("k")["top1"] groups the top1 values by shot
        # count (one group per k, with one value per seed); .agg(["mean", "std"]) returns a table with columns mean and
        # std per k. pandas std uses ddof=1 (sample standard deviation, divides by n-1), so a k with a single seed
        # gets NaN. reset_index turns k from the index back into a normal column. curve: (n_k, 3) = k, mean, std.
        curve = g.dropna(subset=["k"]).groupby("k")["top1"].agg(["mean", "std"]).reset_index()
        # Matplotlib's colour cycle names "C0", "C1", ...; the same colour is reused for this model's star and dashed line.
        color = f"C{i}"
        # errorbar draws the line through (k, mean) with a vertical bar of +/- yerr at each point; yerr is the std,
        # with NaN replaced by 0 (fillna(0)) so single-seed points get no bar. marker="o" = round points;
        # capsize=2 = width of the bar end caps in points (the value 2: reason not recorded in the repo);
        # label feeds the legend.
        ax.errorbar(curve["k"], curve["mean"], yerr=curve["std"].fillna(0), marker="o", color=color,
                    label=model, capsize=2)
        # The "all training images" result is drawn as a star to the right of the largest k.
        # top1 of this model's "all" rows (normally one row, so mean() just returns that value; NaN if there is none).
        all_acc = g.loc[g.shots == "all", "top1"].mean()
        # scatter at x = 2 * largest k, y = all_acc; marker="*" is a star; s=60 is the marker area in points^2
        # (the value 60: reason not recorded in the repo). On the log-2 x axis this sits one tick right of the last k.
        ax.scatter([curve["k"].max() * 2], [all_acc], marker="*", color=color, s=60)
        # Same model's Food-101 zero-shot top-1 ("food" template) is the Western target; draw it as a dashed line.
        # west defaults to NaN (not a number), meaning "no Western target known".
        west = np.nan
        if zs is not None:
            # Food-101 zero-shot top1 values for this model with the "food" template (a Series, possibly empty).
            w = zs[(zs.dataset == "food101") & (zs.model == model) & (zs.template == "food")]["top1"]
            west = w.mean() if len(w) else np.nan
            if not np.isnan(west):
                # axhline draws a horizontal line across the whole axes at y = west; ls="--" is dashed; lw is the
                # line width in points (0.8: reason not recorded in the repo).
                ax.axhline(west, color=color, ls="--", lw=0.8)
        # Shot counts whose mean accuracy reaches the Western target (empty if there is no target).
        # (An old fewshot.csv without the class_weight column also works: the check above is skipped.)
        # curve["mean"] >= west is a boolean Series; the filtered "k" column holds the shot counts that reach it.
        # Without a target, an empty float Series is used, so "reached" below is empty.
        reached = curve[curve["mean"] >= west]["k"] if not np.isnan(west) else pd.Series(dtype=float)
        # Best Indian zero-shot top-1 over all languages and templates for this model. The best setting is
        # picked on the test set (post-hoc), so this number is optimistic for zero-shot.
        ind_zs = (zs[(zs.dataset == "indian") & (zs.model == model)]["top1"].max() if zs is not None else np.nan)
        # One gap-table row: smallest k that reaches the target, or "not reached".
        # reached.min() is the smallest such k (a float, so int() makes it a whole number); an empty Series -> text.
        gap_rows.append({"model": model, "western_zero_shot_top1": west, "indian_best_zero_shot_top1": ind_zs,
                         "indian_all_shot_probe_top1": all_acc,
                         "shots_to_reach_western": int(reached.min()) if len(reached) else "not reached"})
    # Log-2 x axis so small and large shot counts are both easy to read; then labels and save.
    # set_xscale("log", base=2): x positions are spaced by log2, so 1, 2, 4, 8, 16, 32 are equally far apart.
    ax.set_xscale("log", base=2)
    # int(n_all) turns the (possibly float) maximum into a whole number for the label text.
    ax.set_xlabel(f"labelled images per class (star = all {int(n_all)} training images)")
    ax.set_ylabel("top-1 (Indian test)")
    # y range 0..1 (accuracy is a fraction). Legend fontsize 7: reason not recorded in the repo.
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Few-shot linear probe; dashed = same model's Food-101 zero-shot")
    fig.savefig(out / "fewshot_curve.pdf")
    # free the figure
    plt.close(fig)
    # Save the gap table and print it.
    # write_rows writes one CSV row per dict; to_string(index=False) prints the table without the row numbers.
    write_rows(gap_rows, res / "gap.csv")
    print(pd.DataFrame(gap_rows).to_string(index=False))


def fig_worst_classes(pc: pd.DataFrame, zs: pd.DataFrame | None, out: Path, n: int = 20) -> None:
    """Hardest Indian dishes for the model with the best Indian top-1 (romanised Hindi name, 'food' template).

    Inputs: pc = results/zeroshot_per_class.csv (one row per model/setting/class), zs = results/zeroshot.csv
    (used to choose the model), n = how many dishes to show. Output: worst_classes.pdf.
    Shows the dishes with the lowest zero-shot accuracy.
    """
    # Per-class rows for the Indian set with romanised Hindi names and the "food" template.
    # Three boolean masks combined with & (element-wise AND); the parentheses are needed because & binds tightly.
    sub = pc[(pc.dataset == "indian") & (pc.template == "food") & (pc.language == "hindi_roman")]
    # DataFrame.empty is True when there are no rows.
    if sub.empty:
        return  # nothing to plot for this setting, so skip the figure
    # Choose the model with the best overall top-1 for this setting (fallback: best mean per-class accuracy).
    if zs is not None:
        # zero-shot summary rows for the same setting (one row per model).
        best = zs[(zs.dataset == "indian") & (zs.template == "food") & (zs.language == "hindi_roman")]
        model = best.loc[best["top1"].idxmax(), "model"]  # idxmax gives the row label of the highest top-1
    else:
        # Fallback: mean accuracy over each model's classes; idxmax gives the model name with the largest mean.
        model = sub.groupby("model")["accuracy"].mean().idxmax()
    # Keep this model's rows, sort ascending by accuracy then by class name, keep the first n rows: (n, columns).
    worst = sub[sub.model == model].sort_values(["accuracy", "class"]).head(n)  # ties broken by name
    # Horizontal bars, hardest class at the top, then save.
    # figsize (6, 4) inches: reason not recorded in the repo.
    fig, ax = plt.subplots(figsize=(6, 4))
    # barh = horizontal bars: y = class names (categories, drawn bottom to top), width = accuracy.
    ax.barh(worst["class"], worst["accuracy"])
    ax.invert_yaxis()  # put the lowest accuracy at the top
    # accuracy range 0..1
    ax.set_xlim(0, 1)
    ax.set_xlabel("zero-shot accuracy")
    # Smaller class-name labels on the y axis (size 7: reason not recorded in the repo).
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"{n} hardest Indian dishes ({model}, romanised Hindi name)", fontsize=9)
    fig.savefig(out / "worst_classes.pdf")
    # free the figure
    plt.close(fig)


def fig_cnn(hist: pd.DataFrame, out: Path) -> None:
    """Validation accuracy per epoch for every CNN run (train accuracy dashed).

    Input hist = results/cnn_history.csv (columns run, epoch, train_acc, val_acc). Output: cnn_curves.pdf.
    Shows how the from-scratch CNN baseline (E1) learns and whether it overfits.
    """
    # figsize (6, 3.2) inches: reason not recorded in the repo.
    fig, ax = plt.subplots(figsize=(6, 3.2))
    # One colour per run: solid line = validation, dashed line = train; a big gap means overfitting.
    for i, (run, h) in enumerate(hist.groupby("run", sort=False)):
        # matplotlib colour-cycle name, shared by the run's two lines
        colour = f"C{i}"
        # plot(x, y) draws a line through the points; the legend gets one entry per run (validation line only).
        ax.plot(h.epoch, h.val_acc, color=colour, label=f"{run} (val)")
        # Train curve: dashed; alpha=0.5 makes it half transparent (the value 0.5: reason not recorded in the repo).
        ax.plot(h.epoch, h.train_acc, color=colour, linestyle="--", alpha=0.5)
    # Labels, legend, title, then save.
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    # accuracy range 0..1
    ax.set_ylim(0, 1)
    # ncol=2 puts the legend entries in two columns; fontsize 7 and ncol 2: reason not recorded in the repo.
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("CNN from scratch: validation (solid) and train (dashed) accuracy")
    fig.savefig(out / "cnn_curves.pdf")
    # free the figure
    plt.close(fig)


def fig_cnn_worst(summary: pd.DataFrame, per_class: pd.DataFrame, out: Path, n: int = 15) -> None:
    """Hardest and easiest classes for the CNN run with the best test top-1.

    Inputs: summary = results/cnn.csv (one row per run), per_class = results/cnn_per_class.csv.
    Output: cnn_worst_classes.pdf. Tells us which dishes even a trained-on-this-data CNN finds hard.
    """
    # Pick the best run, sort its classes by accuracy, and take the n lowest and n highest.
    # idxmax = row label of the largest top1; .loc[label, "run"] reads that row's run name.
    run = summary.loc[summary["top1"].idxmax(), "run"]
    # This run's per-class rows, sorted from lowest to highest accuracy.
    pc = per_class[per_class.run == run].sort_values("accuracy")
    # head(n) = n lowest-accuracy rows, tail(n) = n highest-accuracy rows.
    worst, best = pc.head(n), pc.tail(n)
    # Two side-by-side bar charts: hardest (left) and easiest (right).
    # subplots(1, 2) = 1 row, 2 columns: axes is an array of 2 Axes; sharex=True gives both the same x range.
    # figsize (7, 3.8) inches: reason not recorded in the repo.
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8), sharex=True)
    # Left panel uses colour C3, right panel C2 (colour choice: reason not recorded in the repo).
    axes[0].barh(worst["class"], worst["accuracy"], color="C3")
    axes[0].set_title(f"{n} hardest classes")
    axes[1].barh(best["class"], best["accuracy"], color="C2")
    axes[1].set_title(f"{n} easiest classes")
    # Same axis settings on both panels.
    for ax in axes:
        ax.set_xlabel("test accuracy")
        # accuracy range 0..1
        ax.set_xlim(0, 1)
        # Class-name label size 6: reason not recorded in the repo.
        ax.tick_params(axis="y", labelsize=6)
    # suptitle = one title above both panels.
    fig.suptitle(f"CNN per-class accuracy (run: {run})", fontsize=9)
    fig.savefig(out / "cnn_worst_classes.pdf")
    # free the figure
    plt.close(fig)


def main() -> None:
    """Build each figure whose result CSV exists, plus the gap table.

    Each block below follows the same pattern: read a CSV (None if missing) and, if present, call the
    matching drawing function. This is why the script can be run at any time, even with partial results.
    """
    # Read config and create the figures folder.
    # Parse the command line (--config, --set) and load the config dict.
    cfg = config_from_args(base_parser(__doc__).parse_args())
    # res = results folder (input CSVs, gap.csv output); out = figures folder (PDF output), both absolute paths.
    res, out = resolve(cfg["paths"]["results"]), resolve(cfg["paths"]["figures"])
    # parents=True creates missing parent folders; exist_ok=True means no error if it already exists.
    out.mkdir(parents=True, exist_ok=True)
    # Zero-shot figures (E2-E4). zs is also reused below to find the Western target and the best model.
    zs = _read(res / "zeroshot.csv")
    if zs is not None:
        fig_bias(zs, out)
        fig_language(zs, out)
    # Few-shot curve and gap table (E5, E6); zs may be None, then the Western line is left out.
    # := assigns and tests in one step: fsh is the DataFrame, or None when the CSV is missing.
    if (fsh := _read(res / "fewshot.csv")) is not None:
        fig_fewshot_and_gap(fsh, zs, out, res)
    # Hardest Indian dishes under zero-shot (E4).
    if (pc := _read(res / "zeroshot_per_class.csv")) is not None:
        fig_worst_classes(pc, zs, out)
    # CNN training curves and per-class bars (E1).
    if (hist := _read(res / "cnn_history.csv")) is not None:
        fig_cnn(hist, out)
        # The per-class bars need both the summary CSV and the per-class CSV.
        cnn_sum, cnn_pc = _read(res / "cnn.csv"), _read(res / "cnn_per_class.csv")
        if cnn_sum is not None and cnn_pc is not None:
            fig_cnn_worst(cnn_sum, cnn_pc, out)
    # Tell the user where the figures were written (also printed when some were skipped).
    print(f"figures in {out}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
