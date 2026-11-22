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
        # nn.Module.__init__ must run first so sub-modules assigned below are registered.
        super().__init__()
        # Main path: conv -> batch norm -> ReLU -> conv -> batch norm. No conv bias because BatchNorm adds one.
        # nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias): (B, cin, H, W) -> (B, cout, H/stride, W/stride)
        # for the first conv (3x3, padding 1 keeps size when stride is 1); the second conv keeps (B, cout, H', W').
        # nn.BatchNorm2d(c) normalises each channel over the batch, then learns a scale and shift.
        # nn.ReLU(inplace=True) sets negatives to 0, overwriting its input to save memory.
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        # Skip path: identity if shapes match, else a 1x1 conv so it has the same shape as the main path.
        # nn.Identity returns its input unchanged; the 1x1 conv with the same stride maps (B, cin, H, W) to
        # (B, cout, H/stride, W/stride), then BatchNorm2d normalises it.
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        # Final ReLU applied after the two paths are added.
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map in, feature map out: ReLU(main path + skip path)."""
        # (B, cin, H, W) -> (B, cout, H/stride, W/stride); the + adds the two paths element by element.
        return self.act(self.body(x) + self.skip(x))


class SmallResNet(nn.Module):
    """~2.8M parameters. Input 3xHxW (any H, W >= 32).

    A small ResNet trained from scratch (no pretraining) as the E1 baseline to compare against CLIP.
    Flow: stem -> 4 stages (8 BasicBlocks, more channels and smaller maps each stage) -> global pool ->
    dropout -> linear layer. Used by: scripts/train_cnn.py."""

    def __init__(self, n_classes: int, widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        """Build stem, four stages of two BasicBlocks (one stage per width), and a classifier head."""
        # nn.Module.__init__ must run first so sub-modules assigned below are registered.
        super().__init__()
        # Stem: 3x3 conv with stride 2 turns the RGB image into widths[0] channels at half size:
        # (B, 3, H, W) -> (B, widths[0], H/2, W/2). BatchNorm2d then ReLU follow.
        # Default widths (32, 64, 128, 256) and dropout 0.3: reason not recorded in the repo
        # (the report only lists dropout 0.3 as a setting).
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        # Stages: the first keeps the size, every later stage halves it (stride 2) and uses more channels.
        # layers collects the blocks; cin tracks the channel count entering the next block.
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            # Stage 0 keeps the size; later stages halve H and W. Each stage: first block changes
            # channels cin -> w (and size), second block keeps (B, w, H', W').
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(cin, w, stride), BasicBlock(w, w, 1)]
            cin = w
        # nn.Sequential(*layers) chains the blocks in order: (B, widths[0], H/2, W/2) -> (B, widths[-1], H/16, W/16).
        self.features = nn.Sequential(*layers)
        # Head: global average pool to one vector per image, dropout against overfitting, then class scores.
        # AdaptiveAvgPool2d(1): (B, C, h, w) -> (B, C, 1, 1); Flatten -> (B, C); Dropout(p) randomly zeroes
        # a fraction p of values in training only; Linear(cin, n_classes): (B, C) -> (B, n_classes).
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images (B, 3, H, W) in, class logits (B, n_classes) out."""
        # stem -> features -> head: (B, 3, H, W) -> (B, widths[0], H/2, W/2) -> (B, widths[-1], h, w) -> (B, n_classes).
        return self.head(self.features(self.stem(x)))


def train_one_epoch(model, loader, optimizer, device, scaler=None) -> tuple[float, float]:
    """Train the model for one pass over the loader; return (mean training loss, training accuracy).

    One pass over all training batches = one epoch. Each step: forward, loss, backward (gradients),
    optimizer update. Used by: scripts/train_cnn.py and scripts/finetune_clip.py."""
    # Training mode: dropout on, batch norm uses batch statistics.
    model.train()
    # CrossEntropyLoss takes logits (B, n_classes) and integer labels (B,) and returns a scalar batch mean.
    # Label smoothing 0.1: the target is not 100% on the true class, which reduces overconfidence.
    # The report lists "label smoothing 0.1" as a setting; no further reason is recorded in the repo.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Running counters: samples seen, correct predictions, sum of per-sample loss.
    total, correct, loss_sum = 0, 0, 0.0
    # Mixed precision (AMP) only on GPU and only when a GradScaler is given.
    use_amp = scaler is not None and device.type == "cuda"
    for x, y in loader:
        # Move the batch to the device: x (B, 3, H, W) images, y (B,) labels. non_blocking=True lets the
        # copy overlap with compute when the source memory is pinned.
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Clear old gradients; set_to_none=True sets them to None instead of filling with zeros
        # (less memory traffic, and backward then writes fresh gradients).
        optimizer.zero_grad(set_to_none=True)
        # Forward pass and loss (in half precision when AMP is on).
        # torch.autocast runs eligible ops in float16 on GPU; enabled=False makes it a no-op.
        # loss is a scalar (0-dim tensor).
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)  # (B, n_classes) logits
            loss = loss_fn(out, y)
        # Backward pass and weight update. With AMP the scaler scales the loss so small gradients
        # do not become zero in half precision.
        # GradScaler: scale(loss).backward() computes scaled gradients; step() unscales them and skips the
        # update if they contain inf/NaN; update() adjusts the scale factor for the next step.
        # Plain path: backward() computes gradients, optimizer.step() updates the weights.
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # Keep running totals; loss is a batch mean, so multiply by batch size before summing.
        # detach() drops the graph; float() turns the scalar tensor into a Python number.
        loss_sum += float(loss.detach()) * len(y)
        # argmax(1) = predicted class per row, (B,); == y gives a bool (B,); sum() counts the hits.
        correct += int((out.argmax(1) == y).sum())
        total += len(y)
    # Mean loss per sample and accuracy over the whole epoch.
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on every batch; return (scores (n, C), true labels (n,)) as NumPy arrays.

    No training happens here. The scores go to metrics.summarize. Used by: train_cnn.py, finetune_clip.py."""
    # Eval mode: dropout off, batch norm uses its stored statistics. no_grad saves memory.
    model.eval()
    scores, ys = [], []
    # Collect scores and labels batch by batch, then join them.
    for x, y in loader:
        # x (B, 3, H, W) -> logits (B, C); .float() casts to float32, .cpu().numpy() moves to a NumPy array.
        scores.append(model(x.to(device)).float().cpu().numpy())
        # Labels (B,) as a NumPy array.
        ys.append(np.asarray(y))
    # Join the per-batch arrays: scores -> (n, C), labels -> (n,).
    return np.concatenate(scores), np.concatenate(ys)
