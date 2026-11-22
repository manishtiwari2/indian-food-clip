"""E1: small ResNet-style CNN trained from scratch (baseline).

The model is used by E1 (scripts/train_cnn.py). The training loop (train_one_epoch) and predict are
also reused by E7 (scripts/finetune_clip.py). Report sections: Methods, "CNN from scratch" (and
"Full fine-tuning" for the shared training loop).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BasicBlock(nn.Module):
    """Residual block: two 3x3 convs plus a skip connection, so the block learns a change to its input."""

    def __init__(self, cin: int, cout: int, stride: int):
        """cin input channels, cout output channels; stride 2 halves height and width."""
        super().__init__()
        # Main path: conv -> batch norm -> ReLU -> conv -> batch norm. No conv bias because BatchNorm adds one.
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        # Skip path: identity if shapes match, else a 1x1 conv so it has the same shape as the main path.
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map in, feature map out: ReLU(main path + skip path)."""
        return self.act(self.body(x) + self.skip(x))


class SmallResNet(nn.Module):
    """~2.8M parameters. Input 3xHxW (any H, W >= 32)."""

    def __init__(self, n_classes: int, widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        """Build stem, four stages of two BasicBlocks (one stage per width), and a classifier head."""
        super().__init__()
        # Stem: 3x3 conv with stride 2 turns the RGB image into widths[0] channels at half size.
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        # Stages: the first keeps the size, every later stage halves it (stride 2) and uses more channels.
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(cin, w, stride), BasicBlock(w, w, 1)]
            cin = w
        self.features = nn.Sequential(*layers)
        # Head: global average pool to one vector per image, dropout against overfitting, then class scores.
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images (B, 3, H, W) in, class logits (B, n_classes) out."""
        return self.head(self.features(self.stem(x)))


def train_one_epoch(model, loader, optimizer, device, scaler=None) -> tuple[float, float]:
    """Train the model for one pass over the loader; return (mean training loss, training accuracy)."""
    # Training mode: dropout on, batch norm uses batch statistics.
    model.train()
    # Label smoothing 0.1: the target is not 100% on the true class, which reduces overconfidence.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    total, correct, loss_sum = 0, 0, 0.0
    # Mixed precision (AMP) only on GPU and only when a GradScaler is given.
    use_amp = scaler is not None and device.type == "cuda"
    for x, y in loader:
        # Move the batch to the device and clear old gradients.
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # Forward pass and loss (in half precision when AMP is on).
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)  # (B, n_classes) logits
            loss = loss_fn(out, y)
        # Backward pass and weight update. With AMP the scaler scales the loss so small gradients
        # do not become zero in half precision.
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # Keep running totals; loss is a batch mean, so multiply by batch size before summing.
        loss_sum += float(loss.detach()) * len(y)
        correct += int((out.argmax(1) == y).sum())
        total += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on every batch; return (scores (n, C), true labels (n,)) as NumPy arrays."""
    # Eval mode: dropout off, batch norm uses its stored statistics. no_grad saves memory.
    model.eval()
    scores, ys = [], []
    # Collect scores and labels batch by batch, then join them.
    for x, y in loader:
        scores.append(model(x.to(device)).float().cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(scores), np.concatenate(ys)
