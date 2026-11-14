"""E1: small ResNet-style CNN trained from scratch (baseline)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class SmallResNet(nn.Module):
    """~2.8M parameters. Input 3xHxW (any H, W >= 32)."""

    def __init__(self, n_classes: int, widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(cin, w, stride), BasicBlock(w, w, 1)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.stem(x)))


def train_one_epoch(model, loader, optimizer, device, scaler=None) -> tuple[float, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    total, correct, loss_sum = 0, 0, 0.0
    use_amp = scaler is not None and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            loss = loss_fn(out, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.detach()) * len(y)
        correct += int((out.argmax(1) == y).sum())
        total += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, ys = [], []
    for x, y in loader:
        scores.append(model(x.to(device)).float().cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(scores), np.concatenate(ys)
