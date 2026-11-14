# Decision log

The project lead owns every decision. Claude may only **propose** (status `proposed`).
A decision is final only when the lead marks it `decided` with a date.

| # | Decision | Status | Date | Notes |
|---|---|---|---|---|
| D1 | Dataset: Kaggle `iamsouravbanerjee/indian-food-images-dataset` (80 classes, ~4,000 imgs) | decided | 2026-11-15 | Licence "Other", academic use only |
| D2 | Western comparison: Food-101 test split, 80 random classes × 50 imgs, seed 42 | decided | 2026-11-15 | Same size as Indian set for a fair comparison |
| D3 | Git flow: same as CPS (issue → branch → PR → review → merge commit) | decided | 2026-11-15 | |
| D4 | Claude builds the pipeline; team runs, verifies, writes the report | decided | 2026-11-15 | Boundaries in `CLAUDE.md` |
| D5 | 4 equal work packages; lead assigns people | decided | 2026-11-15 | See `docs/WORK-PACKAGES.md` |
| D6 | Repo name and owner (personal account vs class org) | **open** | | Suggested name `indian-food-clip` |
| D7 | Deadline and internal milestones | **open** | | |
| D8 | WP → person assignment | **open** | | |
| D9 | "English" prompt = English **description** (e.g. "spiced flattened rice"), "Hindi" = dish name romanized ("poha") and in Devanagari ("पोहा") | proposed | | Tests whether CLIP knows the dish *name* or needs a description |
| D10 | Split: dedup with perceptual hash (distance ≤ 4), then 15 test imgs/class, rest train, seed 42 | proposed | | Locked once `splits/indian_split.csv` is committed |
| D11 | CLIP models: OpenAI ViT-B/32, ViT-B/16, LAION XLM-R ViT-B/32 (multilingual) | proposed | | All via `open_clip` |
| D12 | Few-shot: linear probe at 1/2/4/8/16/all shots, seeds 0/1/2 | proposed | | |
| D13 | Optional full fine-tune of ViT-B/16 | **open** | | Only if time remains |
| D14 | Who submits the Google form | **open** | | One entry per team; needs roll numbers |
