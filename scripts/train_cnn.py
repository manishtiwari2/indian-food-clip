"""WP2 / E1: train the CNN baseline from scratch on the Indian train split, evaluate on test.

What it does: trains a small ResNet (no pre-training) on the Indian train images, keeps the epoch with
the best validation accuracy, and reports top-1, top-5 and macro-F1 on the test split.
Experiment: E1 (CNN from scratch), the baseline that CLIP is compared against.
Writes (appends one run each time): results/cnn.csv, results/cnn_history.csv, results/cnn_per_class.csv.
Report sections: Methods "CNN from scratch"; Results "CNN baseline" (Fig. cnn_curves, cnn_worst_classes).

  python scripts/train_cnn.py --run-name baseline
  python scripts/train_cnn.py --run-name size160 --set cnn.image_size=160
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from foodclip.cnn import SmallResNet, predict, train_one_epoch  # noqa: E402
from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import ImageDataset, SimpleTransform, load_dataset, train_val_split  # noqa: E402
from foodclip.io import get_device, run_meta, set_seed, write_rows  # noqa: E402
from foodclip.metrics import per_class_accuracy, summarize, topk_accuracy  # noqa: E402


def main() -> None:
    """Train the CNN for the configured epochs, then test the best-validation checkpoint."""
    # Read options and config; fix the seed so the run can be repeated.
    p = base_parser(__doc__)
    p.add_argument("--run-name", default="baseline")
    args = p.parse_args()
    cfg = config_from_args(args)
    c = cfg["cnn"]
    set_seed(cfg["seed"])
    device = get_device()

    # Load the Indian split and find the row indices of train and test images.
    paths, df, classes = load_dataset(cfg, "indian")
    paths = np.asarray(paths, dtype=object)  # object array so we can index it with an index array
    y = df["y"].to_numpy()  # integer class labels, shape (n_images,)
    train_idx = np.nonzero(df["split"].to_numpy() == "train")[0]
    test_idx = np.nonzero(df["split"].to_numpy() == "test")[0]
    # Hold out part of train as validation; it picks the best epoch, so the test set is never used for choices.
    tr_idx, val_idx = train_val_split(train_idx, y, c["val_fraction"], len(classes), cfg["seed"])

    def loader(idx, train):
        """Build a DataLoader for the given images; augment and shuffle only for training."""
        ds = ImageDataset(paths[idx], y[idx], SimpleTransform(c["image_size"], train, c.get("color_jitter", 0.0)))
        return torch.utils.data.DataLoader(ds, batch_size=c["batch_size"], shuffle=train,
                                           num_workers=c["num_workers"], pin_memory=device.type == "cuda")

    # Build the model, AdamW optimiser, cosine learning-rate schedule and mixed-precision scaler (GPU only).
    train_dl, val_dl, test_dl = loader(tr_idx, True), loader(val_idx, False), loader(test_idx, False)
    model = SmallResNet(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])  # lr falls to ~0 at the last epoch
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Path where the best model weights are saved during training.
    ckpt = resolve(cfg["paths"]["checkpoints"]) / f"cnn_{args.run_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    # Training loop: one pass over the train data per epoch, then measure validation accuracy.
    for epoch in range(1, c["epochs"] + 1):
        loss, train_acc = train_one_epoch(model, train_dl, opt, device, scaler)
        sched.step()
        val_scores, val_y = predict(model, val_dl, device)
        val_acc = topk_accuracy(val_scores, val_y, 1)
        # Keep one history row per epoch for the training-curve figure.
        history.append({"run": args.run_name, "epoch": epoch, "train_loss": loss, "train_acc": train_acc,
                        "val_acc": val_acc})
        print(f"epoch {epoch:3d} loss {loss:.3f} train {train_acc:.3f} val {val_acc:.3f}")
        # Save the weights only when validation accuracy improves (best epoch chosen on validation, not test).
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), ckpt)

    # Reload the best checkpoint and evaluate it once on the test split.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, test_y = predict(model, test_dl, device)  # scores shape (n_test, n_classes)
    # One summary row with the settings, test metrics and run info (GPU, versions) for traceability.
    res = resolve(cfg["paths"]["results"])
    row = {"run": args.run_name, "model": "SmallResNet", "image_size": c["image_size"], "epochs": c["epochs"],
           "lr": c["lr"], "color_jitter": c.get("color_jitter", 0.0), "seed": cfg["seed"], "best_val_acc": best, "n_train": len(tr_idx),
           **summarize(scores, test_y), **run_meta()}
    # Append results so several runs (e.g. different image sizes) stay in the same CSVs.
    write_rows([row], res / "cnn.csv", append=True)
    write_rows(history, res / "cnn_history.csv", append=True)
    pc = per_class_accuracy(test_y, scores.argmax(1), classes).assign(run=args.run_name)  # tag rows with run name
    write_rows(pc.to_dict("records"), res / "cnn_per_class.csv", append=True)
    print({k: row[k] for k in ("top1", "top5", "macro_f1")})


if __name__ == "__main__":
    main()
