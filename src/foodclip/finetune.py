"""D13: full fine-tune of a CLIP image encoder. The classifier head starts as the zero-shot classifier.

Used by E7 (scripts/finetune_clip.py, full fine-tune of the ViT-B/16 image encoder).
Report section: Methods, "Full fine-tuning".
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ClipClassifier(nn.Module):
    """CLIP image encoder + linear head initialised from class text embeddings, so epoch 0 = zero-shot."""

    def __init__(self, visual: nn.Module, text_feats: np.ndarray, logit_scale: float):
        """Take the CLIP image encoder, (C, D) class text embeddings and CLIP's logit scale; build the model."""
        super().__init__()
        # The image encoder (all its weights are trained during fine-tuning).
        self.visual = visual
        # Linear head: D image features in, C class scores out. No bias, like CLIP's cosine score.
        self.head = nn.Linear(text_feats.shape[1], len(text_feats), bias=False)
        # Copy text embeddings x logit scale into the head weights. Row c = class c's text vector,
        # so head(image_feature) = logit_scale * cosine similarity = exactly the zero-shot score.
        # Training then starts from zero-shot instead of from random weights.
        with torch.no_grad():  # a plain weight copy, not a step that should be tracked for gradients
            self.head.weight.copy_(torch.as_tensor(text_feats, dtype=torch.float32) * logit_scale)  # (C, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images (B, 3, H, W) in, class logits (B, C) out."""
        # Encode images into CLIP feature vectors.
        f = self.visual(x)  # (B, D)
        # L2-normalise so the head computes cosine similarity (times the logit scale).
        return self.head(f / f.norm(dim=-1, keepdim=True))
