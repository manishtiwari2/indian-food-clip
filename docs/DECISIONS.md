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
| D6 | Repo name and owner (personal account vs class org) | decided | 2026-11-19 | Public repo `manishtiwari2/indian-food-clip` |
| D7 | Deadline and internal milestones | **open** | | |
| D8 | WP → person: WP1 Manish, WP2 Anuradha, WP3 Aradhya, WP4 Divyanshi | decided | 2026-11-15 | Issues #1-#4 assigned |
| D9 | "English" prompt = English **description** (e.g. "spiced flattened rice"), "Hindi" = dish name romanized ("poha") and in Devanagari ("पोहा") | decided | 2026-11-19 | Tests whether CLIP knows the dish *name* or needs a description |
| D10 | Split: dedup with perceptual hash (distance ≤ 4); a photo found in two classes is dropped from both; then 15 test imgs/class (30% if class < 30 imgs), rest train, seed 42 | decided | 2026-11-16 | Visual check of 26 removed pairs: all true duplicates. 17 tikka-masala photos were also filed as chicken_tikka. Locked once `splits/indian_split.csv` is committed |
| D11 | CLIP models: OpenAI ViT-B/32, ViT-B/16, LAION XLM-R ViT-B/32 (multilingual) | decided | 2026-11-19 | All via `open_clip` |
| D12 | Few-shot: linear probe with balanced class weights at 1/2/4/8/16/all shots, seeds 0/1/2 | decided | 2026-11-21 | Decided by lead Manish. Changed after the first run: with no class weights, "all" had lower top-1 than k=16 for all 3 models (old "all" top-1: ViT-B-32 0.524, ViT-B-16 0.572, xlm-roberta 0.570). A diagnostic on ViT-B-32 gave 0.615 with balanced weights. Seen on the test set, so this is a post-hoc choice. Old numbers are from Kaggle output, not a committed CSV. `fewshot.class_weight` in the config; one extra unweighted "all" row per model is kept in `results/fewshot.csv` |
| D13 | Optional full fine-tune of ViT-B/16 | **open** | | Only if time remains |
| D14 | Who submits the Google form | **open** | | One entry per team; needs roll numbers |
