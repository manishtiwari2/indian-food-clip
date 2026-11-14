# CLAUDE.md — P11 Indian Food Recognition with CLIP (Team 11)

Deep Learning course project. Question: can CLIP recognise Indian dishes zero-shot, how big is the
Indian-vs-Western gap, and how many labelled shots close it? Full plan: `docs/PLAN.md`.

## How to run
```bash
pip install -r requirements.txt
python scripts/prepare_data.py                 # dedup + split Indian set, Food-101 subset
python scripts/train_cnn.py                    # E1 CNN baseline
python scripts/extract_features.py             # cache CLIP image features
python scripts/run_zeroshot.py                 # E2-E4
python scripts/run_fewshot.py                  # E5
python scripts/make_figures.py                 # figures + E6 gap table
```
Any config value can be overridden: `--set paths.indian_raw=/kaggle/input/... cnn.epochs=5`.
On Kaggle use `notebooks/kaggle_pipeline.ipynb`.

## Test command
```bash
python -m pytest -q
```
Tests use synthetic images only (no download, no GPU, no open_clip needed).

## Directory map
- `configs/default.yaml` — every path, seed and hyperparameter. Change values here, not in code.
- `data/class_names.csv` — 80 classes: `folder`, `english` (English description), `hindi_roman`, `hindi_devanagari`.
- `src/foodclip/` — library code: `data` (list/clean/split/load), `prompts`, `cnn`, `features` (open_clip + cache),
  `fewshot` (zero-shot scoring + linear probe), `metrics`, `config`, `io` (seed, device, result rows).
- `tests/` — synthetic-data tests incl. an end-to-end smoke run of all scripts.
- `scripts/` — one entry point per step; they only glue library code together.
- `splits/` — committed split CSVs (relative paths). `results/` — committed metric CSVs.
- `data/raw/`, `features/`, `checkpoints/` — gitignored.
- `report/` — ELCVIA LaTeX report. `docs/` — plan, decisions, team guide, work packages.

## Conventions
- Flow: issue → branch `type/<wp-id>-<slug>` (e.g. `feat/wp3-hindi-prompts`) → PR with `Closes #N` → teammate review (≥15 words) → merge commit. Never push to `main`, never force-push.
- Conventional Commits, plain lowercase sentence: `feat: add devanagari prompt templates`.
- Each person commits with their own GitHub-linked email. Stage files by name, never `git add .` / `-A`.
- Python 3.10+, type hints on public functions, no notebook-only logic (notebooks only call scripts).
- Seeds come from config. Every result row records model, pretrained tag, language, template, seed.

## Boundaries for Claude (must follow)
1. Work only on the issue / work package named in the request. One step at a time; say what changed.
2. **Never invent, estimate or "fill in" results.** Numbers in README, docs or report come only from
   `results/*.csv` produced by a real run. If a number is missing, write `TODO(result)`.
3. Never change the split, seeds, class list, metrics or test set without the project lead's OK —
   it invalidates every earlier result. Propose it in `docs/DECISIONS.md` instead.
4. No GitHub write actions (push, PR, issue, review, comment, merge, release) unless the user approves
   the exact command. Confirm the branch before every commit.
5. **No `Co-Authored-By` lines and no "Generated with" footers** in commits, PRs or issues. Commits show only the human author.
6. Never commit datasets, images, feature caches, checkpoints or secrets (Kaggle key lives in `~/.kaggle/kaggle.json`).
7. Report text written by Claude is a draft. The section owner rewrites it in their own words.
8. Anything outside the current task: flag it (suggest an issue), do not silently fix it.
9. Do not add dependencies without saying why; keep `requirements.txt` the single source.

## Do not touch (without lead's OK)
- `splits/*.csv` once committed, `configs/default.yaml` → `seed`, `split`, `food101` blocks.
- `data/class_names.csv` → `folder` column (must match dataset folders).
- `report/elcvia.sty`, `report/elcvia.cls` (journal template files).

## Known issues
- Indian dataset is Google-scraped: duplicates, watermarks, some wrong labels. Licence "Other" → academic use only; say so in the report.
- `data/class_names.csv` folder names and Hindi spellings are a draft; WP1 owner verifies against the real dataset and a Hindi speaker checks spellings.
- OpenAI CLIP's tokenizer is English-centric; Devanagari prompts are expected to be weak — that is a finding, not a bug.

## Team (Team 11, Project P11)
| Member | GitHub | Work package |
|---|---|---|
| Manish Tiwari | manishtiwari2 | TBD (lead assigns) |
| Aradhya Patel | Aradhya6394 | TBD |
| Divyanshi Rathour | Rathourdivyanshi6398 | TBD |
| Anuradha Tiwari | Anuradhatiwari01 | TBD |
