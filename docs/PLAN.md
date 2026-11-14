# P11 — Zero-shot vs Few-shot Indian Food Recognition with CLIP

## Context
- DL course project, Team 11, Project P11. Same team as CPS: Manish Tiwari, Aradhya Patel, Divyanshi Rathour, Anuradha Tiwari.
- Research question: can CLIP recognise Indian dishes it rarely saw? How big is the Indian-vs-Western gap, and how many labelled shots close it?
- Deliverables: (1) report PDF in **ELCVIA LaTeX template** (A4, Times 11pt, 8–12 pages), (2) **one GitHub repo**, (3) Google form, 1 entry/team: team no., title, `Name (Email, Roll No)` ×4, report PDF, GitHub link.
- Decisions made: dataset = Kaggle 80-class + Food-101; CPS-style Git flow in a new repo; **Claude builds the full pipeline**, team runs/verifies/writes; work split into 4 equal packages, **you assign names**.

## Dataset (decided)
- **Indian**: `kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset` — 80 classes, 4,000 imgs (~50/class), ~377 MB, Google-scraped, no split, licence "Other" (academic use only — state in report).
  - Clean: perceptual-hash dedup (`imagehash`), drop corrupt files, log removed counts.
  - Split: stratified, fixed seed 42, ~35 train / 15 test per class; save `splits/indian_split.csv` (committed).
  - Names: one CSV `data/class_names.csv` with `english`, `hindi_roman`, `hindi_devanagari` per class (80 rows; a teammate checks the Hindi).
- **Western**: Food-101 (`torchvision.datasets.Food101`, test split). Fair subset: 80 random classes × 50 imgs (seed 42), list committed.
- Compute: free Kaggle T4 notebooks (30 GPU-h/week) — everything fits; image features cached to `.npy` once.

## Experiments
| # | Experiment | Detail |
|---|---|---|
| E1 | CNN from scratch | small ResNet-style CNN, 128px, augmentation, ~50 epochs; baseline |
| E2 | CLIP zero-shot | OpenAI ViT-B/32 + ViT-B/16 (`open_clip`); prompts: plain name, "a photo of {}, a type of Indian food", ensemble |
| E3 | Language of prompt | English vs romanized Hindi vs Devanagari; + multilingual `laion/CLIP-ViT-B-32-xlm-roberta-base` |
| E4 | Bias study | same models/prompts zero-shot on Indian-80 vs Food-101-80 subset; per-class worst dishes |
| E5 | Few-shot | linear probe (logistic regression on cached features) at 1/2/4/8/16/all shots, 3 seeds; optional short fine-tune of ViT-B/16 |
| E6 | Gap curve | shots needed for Indian accuracy to reach Western zero-shot accuracy |
- Metrics: top-1, top-5, macro-F1, per-class accuracy, confusion matrix (top confused pairs). Results saved as CSV in `results/`, figures in `report/figures/`.
- Sanity check: CLIP zero-shot on Food-101 should land near the paper's ~84% (B/32) / ~89% (B/16); if far off, pipeline is wrong.

## Repo (new, e.g. `indian-food-clip`, owner/org = you decide)
```
README.md          how to run, results table, team
CLAUDE.md          rules/boundaries for Claude (below)
docs/PLAN.md       this plan;  docs/TEAM-GUIDE.md  teammate guide;  docs/DECISIONS.md  decision log
requirements.txt   pinned: torch, torchvision, open_clip_torch, scikit-learn, imagehash, pandas, matplotlib
src/foodclip/      data.py  prompts.py  cnn.py  features.py  fewshot.py  metrics.py  config.py  io.py
scripts/           prepare_data.py  train_cnn.py  extract_features.py  run_zeroshot.py  run_fewshot.py  make_figures.py
notebooks/         kaggle_pipeline.ipynb, one section per WP (calls scripts, no logic)
configs/*.yaml     all hyperparams/seeds
splits/  results/  committed (small);  data/  features/  checkpoints/  gitignored
report/            ELCVIA LaTeX (main.tex, elcvia.sty/cls, refs.bib, figures/)
tests/             split has no leakage, class-name CSV has 80 rows, metrics on toy input
```
- Python 3.11, pip + requirements.txt (no pnpm here).

## Work packages — equal weight (~14 h each), you assign names
Claude builds code; the owner runs it on Kaggle, checks outputs, fixes issues with Claude, and writes their report section.
| WP | Scope | Report section |
|---|---|---|
| WP1 Data & repo | download, dedup, split, Food-101 subset, Hindi name CSV, repo/CI setup, README | Intro + Dataset |
| WP2 CNN baseline | E1 runs, curves, tuning, confusion analysis | Related work + CNN method/results |
| WP3 Zero-shot & language | E2, E3, E4 runs, prompt tables, bias figures | Zero-shot + Bias |
| WP4 Few-shot | E5, E6 runs, shot curves, seeds/variance | Few-shot + Conclusion + Abstract |
- Shared (all): final read-through, references, form submission (one person).

## Phases (deadline: you fill in)
1. Setup — repo, CLAUDE.md, guide, issues per WP (Claude drafts `gh issue create` commands; you run/approve).
2. Data — WP1 must finish first (others blocked on split + features).
3. Experiments — WP2/3/4 in parallel.
4. Report — LaTeX sections, figures, page limit 8–12.
5. Submit — tag `v1.0`, PDF in repo, Google form.

## Decisions you own (logged in docs/DECISIONS.md)
Repo name/owner · WP→person · deadline & internal milestones · final model/prompt set · whether to do the optional fine-tune · what goes in the report · who submits the form. Claude proposes; nothing is final until you say so.

## Boundaries for Claude (goes into CLAUDE.md)
- Work only on the task/issue named; one step at a time; report what changed.
- **Never invent or estimate results.** Numbers in README/report come only from `results/*.csv` produced by real runs.
- Never change the split, seeds, class list, or metrics without your OK (they invalidate earlier results).
- No GitHub writes (push, PR, issue, merge, comment) without you approving the exact command. No `git add .`; stage by name; confirm branch before commit.
- **No `Co-Authored-By` / "Generated with" lines** — commits show only the human author.
- No dataset/model files committed; no secrets (Kaggle API key stays in `~/.kaggle`, never in repo).
- Don't write report prose as final — draft only, owner rewrites in own words (academic integrity).
- Flag, don't silently fix, anything outside the current WP (open an issue instead).

## Teammate guide (docs/TEAM-GUIDE.md) contents
- Setup: clone, `pip install -r requirements.txt`, Kaggle API key, open the WP notebook on Kaggle with GPU.
- Flow (same as CPS): issue → branch `type/<wp-id>-<slug>` → PR "Closes #N" → teammate review (≥15 words) → merge commit; never push to main.
- Commits: Conventional Commits, lowercase; your own GitHub-linked email.
- How to use Claude safely: give it the issue, check every output yourself, paste real numbers only.
- Definition of done per WP: code merged, results CSV + figures committed, report section drafted.

## Open items
- Roll numbers for the form (not in the given list).
- Deadline date.
- Repo name and owner (personal account vs class org).

## Verification
- `pytest tests/` passes (no train/test leakage, 80 classes, metric sanity).
- Smoke run: 2 classes end-to-end in <2 min on CPU.
- Food-101 zero-shot sanity check matches paper range.
- Report compiles with ELCVIA template, A4, 8–12 pages; every number traceable to a results CSV.
