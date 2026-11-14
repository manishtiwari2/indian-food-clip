"""CLIP loading and feature extraction (open_clip). Image features are cached to .npz once."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import resolve
from .data import ImageDataset


def model_tag(model_cfg: dict) -> str:
    return f"{model_cfg['name']}__{model_cfg['pretrained']}"


def feature_path(cfg: dict, dataset: str, model_cfg: dict) -> Path:
    return resolve(cfg["paths"]["features"]) / f"{dataset}__{model_tag(model_cfg)}.npz"


def load_clip(model_cfg: dict, device: torch.device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_cfg["name"], pretrained=model_cfg["pretrained"], device=device)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(model_cfg["name"])


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_images(model, preprocess, paths: Sequence[Path], device: torch.device,
                  batch_size: int = 128, num_workers: int = 2) -> np.ndarray:
    loader = torch.utils.data.DataLoader(ImageDataset(paths, [0] * len(paths), preprocess),
                                         batch_size=batch_size, num_workers=num_workers)
    feats = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            f = model.encode_image(x.to(device))
        feats.append(_normalize(f.float()).cpu())
    return torch.cat(feats).numpy().astype(np.float32)


@torch.no_grad()
def encode_text(model, tokenizer, prompts_per_class: list[list[str]], device: torch.device) -> np.ndarray:
    """(C, D): mean of the normalised embeddings of each class's prompts, renormalised."""
    out = []
    for prompts in prompts_per_class:
        f = _normalize(model.encode_text(tokenizer(prompts).to(device)).float())
        out.append(_normalize(f.mean(0)))
    return torch.stack(out).cpu().numpy().astype(np.float32)


def save_features(path: Path, feats: np.ndarray, relpaths: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feats=feats, relpath=np.asarray(relpaths, dtype=str))


def load_features(path: Path, relpaths: Sequence[str]) -> np.ndarray:
    """Load cached features and check they are in the same order as the split CSV."""
    data = np.load(path, allow_pickle=False)
    if list(data["relpath"]) != list(relpaths):
        raise ValueError(f"{path} does not match the current split; re-run extract_features.py --force")
    return data["feats"]
