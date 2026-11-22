"""CLIP loading and feature extraction (open_clip). Image features are cached to .npz once.

Why cache: the CLIP image encoder is frozen in E2-E5, so each image's feature never changes. Computing it
once and saving it makes every zero-shot and few-shot run fast (no GPU needed after extraction).
Used by E2-E4 (scripts/run_zeroshot.py), E5 (scripts/run_fewshot.py), feature extraction
(scripts/extract_features.py) and E7 (encode_text for the head of scripts/finetune_clip.py).
Report sections: Methods, "CLIP zero-shot and prompt engineering" and "Few-shot linear probe".
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import resolve
from .data import ImageDataset


def model_tag(model_cfg: dict) -> str:
    """Model config in; short id like "ViT-B-32__openai" (architecture + pretrained weights) out."""
    return f"{model_cfg['name']}__{model_cfg['pretrained']}"


def feature_path(cfg: dict, dataset: str, model_cfg: dict) -> Path:
    """Config, dataset name and model config in; path of that dataset's cached feature file out."""
    return resolve(cfg["paths"]["features"]) / f"{dataset}__{model_tag(model_cfg)}.npz"


def load_clip(model_cfg: dict, device: torch.device):
    """Model config in; (CLIP model in eval mode, image preprocessing, text tokenizer) out."""
    # Imported here so the rest of the package (and the tests) work without open_clip installed.
    import open_clip

    # Build the model with its pretrained weights and the matching image preprocessing.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_cfg["name"], pretrained=model_cfg["pretrained"], device=device)  # "_" = the train transform
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(model_cfg["name"])


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Vectors in, same vectors scaled to length 1 (L2-normalised) out, so dot product = cosine similarity."""
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_images(model, preprocess, paths: Sequence[Path], device: torch.device,
                  batch_size: int = 128, num_workers: int = 2) -> np.ndarray:
    """Image paths in; (n, D) float32 array of L2-normalised CLIP image features out (same order as paths)."""
    # Load images in batches with CLIP's own preprocessing. Labels are dummies (0); they are not used.
    loader = torch.utils.data.DataLoader(ImageDataset(paths, [0] * len(paths), preprocess),
                                         batch_size=batch_size, num_workers=num_workers)
    feats = []
    # Encode each batch. On GPU, autocast uses half precision to save memory and time.
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            f = model.encode_image(x.to(device))  # (batch, D)
        # Back to float32, normalise to length 1, move to CPU memory.
        feats.append(_normalize(f.float()).cpu())
    # Join all batches into one (n, D) array.
    return torch.cat(feats).numpy().astype(np.float32)


@torch.no_grad()
def encode_text(model, tokenizer, prompts_per_class: list[list[str]], device: torch.device) -> np.ndarray:
    """(C, D): mean of the normalised embeddings of each class's prompts, renormalised."""
    out = []
    # For each class: tokenize and encode all its prompts, then normalise each one.
    for prompts in prompts_per_class:
        f = _normalize(model.encode_text(tokenizer(prompts).to(device)).float())  # (n_prompts, D)
        # Average the prompts into one class vector and normalise again (the mean is shorter than 1).
        out.append(_normalize(f.mean(0)))
    # Stack the C class vectors into one (C, D) array.
    return torch.stack(out).cpu().numpy().astype(np.float32)


def save_features(path: Path, feats: np.ndarray, relpaths: Sequence[str]) -> None:
    """Features (n, D) and their image relpaths in; compressed .npz file written."""
    # relpaths are saved too, so load_features can check the order later.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feats=feats, relpath=np.asarray(relpaths, dtype=str))


def load_features(path: Path, relpaths: Sequence[str]) -> np.ndarray:
    """Load cached features and check they are in the same order as the split CSV."""
    data = np.load(path, allow_pickle=False)  # no pickle: safer, and plain arrays do not need it
    # Row i must belong to image i of the split; otherwise labels and features would not match.
    if list(data["relpath"]) != list(relpaths):
        raise ValueError(f"{path} does not match the current split; re-run extract_features.py --force")
    return data["feats"]
