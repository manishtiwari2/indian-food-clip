"""D13: full fine-tune of a CLIP image encoder. The classifier head starts as the zero-shot classifier.

Used by E7 (scripts/finetune_clip.py, full fine-tune of the ViT-B/16 image encoder).
Report section: Methods, "Full fine-tuning".
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ClipClassifier(nn.Module):
    """CLIP image encoder + linear head initialised from class text embeddings, so epoch 0 = zero-shot.

    How: image -> CLIP visual encoder -> normalised feature (D numbers) -> linear head -> C class scores.
    Why: starting from the text embeddings means fine-tuning begins at the zero-shot accuracy and only
    improves from there, instead of learning a head from random weights.
    Used by: scripts/finetune_clip.py (E7)."""

    def __init__(self, visual: nn.Module, text_feats: np.ndarray, logit_scale: float):
        """Take the CLIP image encoder, (C, D) class text embeddings and CLIP's logit scale; build the model."""
        # nn.Module.__init__ must run first so PyTorch can register the sub-modules assigned below.
        super().__init__()
        # The image encoder, stored as a sub-module so its weights are in model.parameters()
        # (all its weights are trained during fine-tuning).
        self.visual = visual
        # Linear head: D image features in, C class scores out. nn.Linear(in_features, out_features, bias=False)
        # holds a weight of shape (C, D) and no bias vector, like CLIP's cosine score.
        # text_feats.shape[1] = D, len(text_feats) = C.
        self.head = nn.Linear(text_feats.shape[1], len(text_feats), bias=False)
        # Copy text embeddings x logit scale into the head weights. Row c = class c's text vector,
        # so head(image_feature) = logit_scale * cosine similarity = exactly the zero-shot score.
        # Training then starts from zero-shot instead of from random weights.
        # torch.no_grad() turns off gradient tracking; as_tensor converts the NumPy array to a float32 tensor
        # (sharing memory when it can); copy_ overwrites the weight in place.
        with torch.no_grad():  # a plain weight copy, not a step that should be tracked for gradients
            self.head.weight.copy_(torch.as_tensor(text_feats, dtype=torch.float32) * logit_scale)  # (C, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images (B, 3, H, W) in, class logits (B, C) out."""
        # Encode images into CLIP feature vectors.
        f = self.visual(x)  # (B, D)
        # f.norm(dim=-1, keepdim=True) is the L2 length of each row, shape (B, 1); dividing makes every row
        # unit length. L2-normalise so the head computes cosine similarity (times the logit scale).
        # Head: (B, D) @ (D, C) -> (B, C) logits.
        return self.head(f / f.norm(dim=-1, keepdim=True))
