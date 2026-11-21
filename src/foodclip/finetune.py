"""D13: full fine-tune of a CLIP image encoder. The classifier head starts as the zero-shot classifier."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ClipClassifier(nn.Module):
    """CLIP image encoder + linear head initialised from class text embeddings, so epoch 0 = zero-shot."""

    def __init__(self, visual: nn.Module, text_feats: np.ndarray, logit_scale: float):
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(text_feats.shape[1], len(text_feats), bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.as_tensor(text_feats, dtype=torch.float32) * logit_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.visual(x)
        return self.head(f / f.norm(dim=-1, keepdim=True))
