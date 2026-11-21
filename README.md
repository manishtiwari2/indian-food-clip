# Zero-shot vs Few-shot Indian Food Recognition with CLIP

Deep Learning course project — **Team 11, Project P11**.

**Question:** can a vision-language model (CLIP) recognise Indian dishes it rarely saw in training?
How large is the gap to Western food, does the prompt language (English description vs romanised
Hindi vs Devanagari) matter, and how many labelled images per class close the gap?

## Pipeline
| Step | Script | Output |
|---|---|---|
| Data: clean, dedup, split; Food-101 subset | `scripts/prepare_data.py` | `splits/*.csv` |
| E1 CNN from scratch | `scripts/train_cnn.py` | `results/cnn*.csv` |
| CLIP image features (cached) | `scripts/extract_features.py` | `features/*.npz` (not committed) |
| E2–E4 zero-shot, prompt language, bias | `scripts/run_zeroshot.py` | `results/zeroshot*.csv` |
| E5 few-shot linear probe | `scripts/run_fewshot.py` | `results/fewshot.csv` |
| E6 gap + all figures | `scripts/make_figures.py` | `results/gap.csv`, `report/figures/*.pdf` |

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/prepare_data.py --food101 --download-food101 --set paths.indian_raw=<path-to-dataset>
```
GPU runs: `notebooks/kaggle_pipeline.ipynb` on Kaggle (T4).

## Data
- Indian: [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset) — 80 classes, ~4,000 images, scraped from the web; used for academic purposes only.
- Western: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) test split, 80 random classes × 50 images (seed 42).

## Results
Top-1 accuracy (%) on the 1,187 Indian test images, from `results/*.csv`. Food-101 is 80 classes × 50 images.

| Method | ViT-B/32 | ViT-B/16 | XLM-R ViT-B/32 |
|---|---|---|---|
| CNN from scratch (best of 4 runs, `size160`) | 26.4 | – | – |
| CLIP zero-shot, best Indian setting (romanised Hindi) | 43.1 | 46.9 | 43.6 |
| Linear probe, 16 shots per class (mean of 3 seeds) | 58.2 | 60.1 | 59.1 |
| Linear probe, all 2,254 training images (balanced weights) | 61.5 | 63.9 | 62.1 |
| *Food-101 zero-shot (`food` template)* | *85.6* | *90.0* | *82.6* |

The CNN does not use CLIP, so it has one value. Even with all training images, the probe stays below
Food-101 zero-shot for every model (`results/gap.csv`). Details: `results/cnn.csv`, `zeroshot.csv`, `fewshot.csv`.

## Team
Manish Tiwari · Aradhya Patel · Divyanshi Rathour · Anuradha Tiwari

Docs: [plan](docs/PLAN.md) · [decisions](docs/DECISIONS.md) · [team guide](docs/TEAM-GUIDE.md) · [work packages](docs/WORK-PACKAGES.md)

## Reference
Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021. https://arxiv.org/abs/2103.00020
