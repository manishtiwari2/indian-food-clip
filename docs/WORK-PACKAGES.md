# Work packages (equal weight, ~14 h each)

Lead assigns one WP per person (fill the Owner column). Claude has already written the code; the owner
**runs it, checks it, fixes problems, and writes the report section**.

| WP | Owner | Blocked by | Report section |
|---|---|---|---|
| WP1 Data & repo | TBD | — | 1 Introduction, 3 Dataset |
| WP2 CNN baseline | TBD | WP1 split | 2 Related work, 4.1 + 5.1 CNN |
| WP3 Zero-shot, language, bias | TBD | WP1 split | 4.2 + 5.2 Zero-shot, 5.3 Bias |
| WP4 Few-shot & gap | TBD | WP1 split, WP3 features | 4.3 + 5.4 Few-shot, 6 Conclusion, Abstract |

## WP1 — Data & repo (~14 h)
- [ ] Create GitHub repo, push initial code, add teammates, protect `main` (no direct push). (2 h)
- [ ] Run `prepare_data.py` on Kaggle; fix any `class_names.csv` folder mismatches it reports. (3 h)
- [ ] Look through removed duplicates + 5 random images per class; note label noise. (3 h)
- [ ] Get a Hindi speaker to check all 80 `hindi_roman` / `hindi_devanagari` names. (2 h)
- [ ] Run `prepare_data.py --food101`; commit both split CSVs (after lead's OK). (1 h)
- [ ] Write Introduction + Dataset section (table: classes, images, removed, split sizes). (3 h)

## WP2 — CNN baseline (~14 h)
- [ ] Run `train_cnn.py` with defaults; commit `results/cnn.csv`, `results/cnn_history.csv`. (3 h)
- [ ] Try ≤3 changes (image size, epochs, augmentation) via `--set`; record each run. (4 h)
- [ ] Figures: training curves, worst-classes confusion (from `make_figures.py`). (2 h)
- [ ] Write Related work (CLIP, Food-101, Indian food datasets) + CNN method/results. (5 h)

## WP3 — Zero-shot, language, bias (~14 h)
- [ ] Run `extract_features.py` for all 3 models on both datasets (share `features/` with WP4 via a Kaggle dataset). (3 h)
- [ ] Run `run_zeroshot.py`; check Food-101 B/32 is near the paper (~84%) — if far off, stop and report. (3 h)
- [ ] Analyse: English description vs romanized Hindi vs Devanagari; per-template; worst dishes. (3 h)
- [ ] Write Zero-shot method/results + Bias section (Indian vs Western table, worst-20 figure). (5 h)

## WP4 — Few-shot & gap (~14 h)
- [ ] Run `run_fewshot.py` (3 models × 6 shot levels × 3 seeds). (3 h)
- [ ] Run `make_figures.py`; check `results/gap.csv` (shots to reach Western zero-shot). (2 h)
- [ ] Optional (lead decides, D13): short fine-tune of ViT-B/16. (3 h)
- [ ] Write Few-shot method/results, Conclusion, Abstract; final page-count pass (8–12 pages). (6 h)

## Shared (all, last week)
- Read the whole report once; check every number against `results/*.csv`.
- Compile PDF (A4, ELCVIA template), tag release `v1.0`, one person submits the Google form.

## Issue commands (lead runs these after creating the repo; edit `--assignee`)
```bash
gh issue create --title "wp1: data pipeline, splits and repo setup" --body "See docs/WORK-PACKAGES.md#wp1. DoD in docs/TEAM-GUIDE.md section 5." --label wp1
gh issue create --title "wp2: cnn from scratch baseline" --body "See docs/WORK-PACKAGES.md#wp2." --label wp2
gh issue create --title "wp3: clip zero-shot, prompt language and bias study" --body "See docs/WORK-PACKAGES.md#wp3." --label wp3
gh issue create --title "wp4: few-shot linear probe and gap curve" --body "See docs/WORK-PACKAGES.md#wp4." --label wp4
gh issue create --title "report: assemble, proofread and submit" --body "Shared final tasks. See docs/WORK-PACKAGES.md#shared-all-last-week." --label report
```
(Create the labels first: `gh label create wp1` … `gh label create report`.)
