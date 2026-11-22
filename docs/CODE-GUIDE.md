# Code guide

A one-page map from the report to the code. Every number in the report comes from a CSV in `results/`.

## How one run flows
1. `scripts/prepare_data.py` lists the Indian images, removes broken files and near-duplicates (pHash), and makes a stratified train/test split. It also picks the Food-101 subset of 80 classes × 50 test images. Output: `splits/*.csv`.
2. `scripts/train_cnn.py` trains a small ResNet from scratch on the Indian train split (E1).
3. `scripts/extract_features.py` runs each CLIP image encoder once and caches the image features in `features/*.npz`.
4. `scripts/run_zeroshot.py` and `scripts/run_fewshot.py` work only on the cached features, so they are fast (E2–E5).
5. `scripts/finetune_clip.py` trains the full ViT-B/16 image encoder on raw images (E7).
6. `scripts/make_figures.py` reads `results/*.csv` and writes the gap table and all report figures (E6).

Every setting (paths, seeds, hyperparameters) is in `configs/default.yaml`. You can change one with `--set key=value`.

## Experiment → code → result → report
| Exp | What | Script | Main library functions | Result files | Report |
|---|---|---|---|---|---|
| Data | Clean, dedup, split | `prepare_data.py` | `data.list_images`, `is_valid_image`, `dedup`, `stratified_split`, `food101_subset` | `splits/*.csv` | Sec. Dataset |
| E1 | CNN from scratch | `train_cnn.py` | `cnn.SmallResNet`, `train_one_epoch`, `predict`, `data.SimpleTransform` | `results/cnn*.csv` | Methods: CNN from scratch; Results: CNN baseline (Fig. 1–2) |
| — | Cache CLIP features | `extract_features.py` | `features.load_clip`, `encode_images`, `save_features` | `features/*.npz` (not committed) | Methods: CLIP zero-shot |
| E2–E3 | Zero-shot, prompt language and templates | `run_zeroshot.py` | `prompts.build_prompts`, `valid_combo`, `features.encode_text`, `fewshot.zero_shot_scores` | `results/zeroshot*.csv` | Results: Zero-shot and prompt language (Fig. 3) |
| E4 | Indian vs Western bias | `run_zeroshot.py` | `prompts.food101_name`, `metrics.per_class_accuracy`, `top_confusions` | `results/zeroshot*.csv` | Results: Indian vs Western bias (Fig. 4–5) |
| E5 | Few-shot linear probe | `run_fewshot.py` | `fewshot.sample_k_shot`, `linear_probe` | `results/fewshot.csv` | Results: How many shots close the gap? (Fig. 6) |
| E6 | Gap table and figures | `make_figures.py` | reads the CSVs | `results/gap.csv`, `report/figures/*.pdf` | All figures; gap table |
| E7 | Full fine-tune of ViT-B/16 | `finetune_clip.py` | `finetune.ClipClassifier`, `data.train_val_split`, `cnn.train_one_epoch` | `results/finetune*.csv` | Methods and Results: Full fine-tuning |

## Key ideas you may be asked about
- **Zero-shot:** each class name goes into prompt templates, for example "a photo of poha, a type of food". The text encoder turns the prompts into vectors. An image gets the class whose text vector has the highest cosine similarity to its image vector (`fewshot.zero_shot_scores`).
- **Linear probe:** the CLIP image encoder stays frozen. A logistic regression is trained on its features, using k images per class and 3 seeds (`fewshot.linear_probe`).
- **Fine-tune:** the classifier head starts as the zero-shot text vectors, so epoch 0 is the zero-shot result. Only the image encoder and the head are trained. The best epoch is picked on a validation part of the train split, never on the test set.
- **Metrics:** top-1, top-5, macro-F1 and per-class accuracy (`metrics.py`).
- **Tests:** `python -m pytest -q` runs the whole pipeline on synthetic images, with no download and no GPU.
