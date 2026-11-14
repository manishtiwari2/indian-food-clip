# Team guide — P11 Indian Food Recognition with CLIP

Read this once before you start. Rules for Claude are in `CLAUDE.md`; decisions in `docs/DECISIONS.md`;
your tasks in `docs/WORK-PACKAGES.md`.

## 1. One-time setup (local)
1. Clone: `git clone <repo-url>` then `cd indian-food-clip`.
2. Set your commit identity (must be the email linked to your GitHub):
   `git config user.name "Your Name"` and `git config user.email "you@example.com"`.
3. Install: `pip install -r requirements.txt` (Python 3.10+).
4. Check: `python -m pytest -q` → all tests pass.

## 2. Running on Kaggle (GPU)
1. Kaggle → New Notebook → Settings → Accelerator **GPU T4**, Internet **On**.
2. Add data: "+ Add Input" → search `indian-food-images-dataset` (by iamsouravbanerjee).
3. Upload / import `notebooks/kaggle_pipeline.ipynb` and run only the cells for your work package.
4. Download the files it writes in `results/` and `report/figures/`, commit them in your PR.
5. Never upload your `kaggle.json` key anywhere in the repo.

## 3. Daily Git flow (same as CPS)
1. Pick your issue on GitHub (one issue = one branch = one PR).
2. `git switch main` → `git pull` → `git switch -c feat/wp3-hindi-prompts`.
3. Small commits, Conventional Commits, lowercase: `feat: add devanagari prompt templates`.
4. Stage by name: `git add src/foodclip/prompts.py` (never `git add .`).
5. `git push -u origin <branch>` → open PR → body has `Closes #N`, what changed, how you verified.
6. One teammate reviews (≥15 words, read the Commits tab) → merge commit. Nobody pushes to `main`.

## 4. Using Claude safely
- Start every Claude session with: "Read CLAUDE.md. Task: issue #N (WPx)".
- Claude writes code; **you run it and check the output yourself**. If you did not see the number in a
  `results/*.csv` from a real run, it does not go in the report.
- Claude must not change splits, seeds, class list or metrics. If it suggests that, ask the lead.
- Report text from Claude is a draft only — rewrite in your own words.
- Claude never adds "Co-Authored-By" or "Generated with" lines. If you see one, remove it before pushing.

## 5. Definition of done (every work package)
- Code merged via reviewed PR, tests pass.
- Result CSVs in `results/` and figures in `report/figures/` committed.
- Your report section drafted in `report/main.tex` with every number traceable to a CSV.
- Short note in the PR: which Kaggle run produced the results (date, model, settings).

## 6. Who decides what
The project lead decides: scope, splits, models, which results go in the report, deadlines,
who submits the form. Anyone may propose a change in `docs/DECISIONS.md` (status `proposed`).
