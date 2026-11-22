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
    """Model config in; short id like "ViT-B-32__openai" (architecture + pretrained weights) out.

    Why: used in file names so features of different models never overwrite each other."""
    # model_cfg is a dict with keys "name" (architecture) and "pretrained" (weights tag); joined by two underscores.
    return f"{model_cfg['name']}__{model_cfg['pretrained']}"


def feature_path(cfg: dict, dataset: str, model_cfg: dict) -> Path:
    """Config, dataset name and model config in; path of that dataset's cached feature file out."""
    # config.resolve turns the configured features folder into an absolute Path (relative = from the repo root);
    # "/" on a Path appends the file name, e.g. <features dir>/<dataset>__<model tag>.npz.
    return resolve(cfg["paths"]["features"]) / f"{dataset}__{model_tag(model_cfg)}.npz"


def load_clip(model_cfg: dict, device: torch.device):
    """Model config in; (CLIP model in eval mode, image preprocessing, text tokenizer) out.

    The preprocessing resizes/crops/normalises images the way this CLIP model was trained; the tokenizer
    turns text into token ids. eval() switches off training-only behaviour. Weights are downloaded the
    first time. Used by: extract_features.py, run_zeroshot.py, finetune_clip.py."""
    # Imported here so the rest of the package (and the tests) work without open_clip installed.
    import open_clip

    # Build the model with its pretrained weights and the matching image preprocessing.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_cfg["name"], pretrained=model_cfg["pretrained"], device=device)  # "_" = the train transform
    # eval() puts the model in inference mode (affects layers such as dropout/batch norm if present).
    model.eval()
    # Third value: the tokenizer matching this architecture name.
    return model, preprocess, open_clip.get_tokenizer(model_cfg["name"])


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Vectors in, same vectors scaled to length 1 (L2-normalised) out, so dot product = cosine similarity."""
    # x.norm(dim=-1, keepdim=True) = length of each vector along the last axis, shape (..., 1) so it broadcasts
    # when dividing; the result has the same shape as x.
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_images(model, preprocess, paths: Sequence[Path], device: torch.device,
                  batch_size: int = 128, num_workers: int = 2) -> np.ndarray:
    """Image paths in; (n, D) float32 array of L2-normalised CLIP image features out (same order as paths)."""
    # Load images in batches with CLIP's own preprocessing. Labels are dummies (0); they are not used.
    # ImageDataset (data.py) opens image i from disk on demand and returns (transform(image), label), so each
    # item is a (3, H, W) tensor. DataLoader stacks items into batches (batch, 3, H, W); batch_size = images per
    # batch; num_workers = number of background processes that load images in parallel. shuffle is not set,
    # so the order stays the same as paths (needed so features line up with the split CSV).
    loader = torch.utils.data.DataLoader(ImageDataset(paths, [0] * len(paths), preprocess),
                                         batch_size=batch_size, num_workers=num_workers)
    # List of CPU tensors, one (batch, D) tensor per batch.
    feats = []
    # Encode each batch. On GPU, autocast uses half precision to save memory and time.
    for x, _ in loader:
        # torch.autocast runs eligible operations in lower precision; enabled only on a CUDA device.
        # (The @torch.no_grad() decorator on this function turns off gradient tracking: nothing is trained here,
        # so no memory is spent on the backward pass.)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            # x.to(device) moves the batch to the GPU/CPU; encode_image returns one feature vector per image.
            f = model.encode_image(x.to(device))  # (batch, D)
        # Back to float32, normalise to length 1, move to CPU memory.
        feats.append(_normalize(f.float()).cpu())
    # Join all batches into one (n, D) array.
    # torch.cat joins along dim 0 -> (n, D); .numpy() converts to numpy; astype makes sure it is float32.
    return torch.cat(feats).numpy().astype(np.float32)


@torch.no_grad()
def encode_text(model, tokenizer, prompts_per_class: list[list[str]], device: torch.device) -> np.ndarray:
    """(C, D): mean of the normalised embeddings of each class's prompts, renormalised.

    Input: prompts_per_class as built by prompts.build_prompts. Output: one unit-length text vector per
    class (the "zero-shot classifier"). Averaging several prompts is prompt ensembling.
    Used by: run_zeroshot.py and finetune_clip.py (head start values)."""
    # List of (D,) class vectors, one per class (a torch tensor each).
    out = []
    # For each class: tokenize and encode all its prompts, then normalise each one.
    for prompts in prompts_per_class:
        # tokenizer(prompts) -> (n_prompts, context_length) token ids; .to(device) moves them; encode_text gives
        # (n_prompts, D); .float() casts to float32. @torch.no_grad() on the function disables gradient tracking.
        f = _normalize(model.encode_text(tokenizer(prompts).to(device)).float())  # (n_prompts, D)
        # Average the prompts into one class vector and normalise again (the mean is shorter than 1).
        # f.mean(0) averages over the prompts (dim 0) -> (D,).
        out.append(_normalize(f.mean(0)))
    # Stack the C class vectors into one (C, D) array.
    # torch.stack adds a new first axis -> (C, D); .cpu() moves to CPU memory; .numpy() converts to numpy.
    return torch.stack(out).cpu().numpy().astype(np.float32)


def save_features(path: Path, feats: np.ndarray, relpaths: Sequence[str]) -> None:
    """Features (n, D) and their image relpaths in; compressed .npz file written.

    Used by: scripts/extract_features.py, after encode_images."""
    # relpaths are saved too, so load_features can check the order later.
    # Create the folder if missing (parents=True also creates parent folders; exist_ok=True: no error if it exists).
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed writes a zip-compressed .npz with two arrays: "feats" (n, D) and "relpath" (n,) strings.
    np.savez_compressed(path, feats=feats, relpath=np.asarray(relpaths, dtype=str))


def load_features(path: Path, relpaths: Sequence[str]) -> np.ndarray:
    """Load cached features and check they are in the same order as the split CSV.

    Raises ValueError if the saved image list differs from relpaths (stale cache).
    Used by: run_zeroshot.py and run_fewshot.py."""
    data = np.load(path, allow_pickle=False)  # no pickle: safer, and plain arrays do not need it
    # Row i must belong to image i of the split; otherwise labels and features would not match.
    # data["relpath"] is the saved (n,) string array; compared as lists to the current split order.
    if list(data["relpath"]) != list(relpaths):
        raise ValueError(f"{path} does not match the current split; re-run extract_features.py --force")
    # Read and return the (n, D) feature array from the .npz file.
    return data["feats"]
