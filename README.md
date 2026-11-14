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
`TODO(result)` — filled only from `results/*.csv` after the Kaggle runs.

## Team
Manish Tiwari · Aradhya Patel · Divyanshi Rathour · Anuradha Tiwari

Docs: [plan](docs/PLAN.md) · [decisions](docs/DECISIONS.md) · [team guide](docs/TEAM-GUIDE.md) · [work packages](docs/WORK-PACKAGES.md)

## Reference
Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021. https://arxiv.org/abs/2103.00020
