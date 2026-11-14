"""WP2 / E1: train the CNN baseline from scratch on the Indian train split, evaluate on test.

  python scripts/train_cnn.py --run-name baseline
  python scripts/train_cnn.py --run-name size160 --set cnn.image_size=160
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from foodclip.cnn import SmallResNet, predict, train_one_epoch  # noqa: E402
from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import ImageDataset, SimpleTransform, load_dataset  # noqa: E402
from foodclip.io import get_device, run_meta, set_seed, write_rows  # noqa: E402
from foodclip.metrics import per_class_accuracy, summarize, topk_accuracy  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--run-name", default="baseline")
    args = p.parse_args()
    cfg = config_from_args(args)
    c = cfg["cnn"]
    set_seed(cfg["seed"])
    device = get_device()

    paths, df, classes = load_dataset(cfg, "indian")
    paths = np.asarray(paths, dtype=object)
    y = df["y"].to_numpy()
    train_idx = np.nonzero(df["split"].to_numpy() == "train")[0]
    test_idx = np.nonzero(df["split"].to_numpy() == "test")[0]
    n_val = int(round(len(train_idx) * c["val_fraction"]))
    stratify = y[train_idx] if n_val >= len(classes) else None
    tr_idx, val_idx = train_test_split(train_idx, test_size=n_val, random_state=cfg["seed"], stratify=stratify)

    def loader(idx, train):
        ds = ImageDataset(paths[idx], y[idx], SimpleTransform(c["image_size"], train))
        return torch.utils.data.DataLoader(ds, batch_size=c["batch_size"], shuffle=train,
                                           num_workers=c["num_workers"], pin_memory=device.type == "cuda")

    train_dl, val_dl, test_dl = loader(tr_idx, True), loader(val_idx, False), loader(test_idx, False)
    model = SmallResNet(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt = resolve(cfg["paths"]["checkpoints"]) / f"cnn_{args.run_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, c["epochs"] + 1):
        loss, train_acc = train_one_epoch(model, train_dl, opt, device, scaler)
        sched.step()
        val_scores, val_y = predict(model, val_dl, device)
        val_acc = topk_accuracy(val_scores, val_y, 1)
        history.append({"run": args.run_name, "epoch": epoch, "train_loss": loss, "train_acc": train_acc,
                        "val_acc": val_acc})
        print(f"epoch {epoch:3d} loss {loss:.3f} train {train_acc:.3f} val {val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, test_y = predict(model, test_dl, device)
    res = resolve(cfg["paths"]["results"])
    row = {"run": args.run_name, "model": "SmallResNet", "image_size": c["image_size"], "epochs": c["epochs"],
           "lr": c["lr"], "seed": cfg["seed"], "best_val_acc": best, "n_train": len(tr_idx),
           **summarize(scores, test_y), **run_meta()}
    write_rows([row], res / "cnn.csv", append=True)
    write_rows(history, res / "cnn_history.csv", append=True)
    pc = per_class_accuracy(test_y, scores.argmax(1), classes).assign(run=args.run_name)
    write_rows(pc.to_dict("records"), res / "cnn_per_class.csv", append=True)
    print({k: row[k] for k in ("top1", "top5", "macro_f1")})


if __name__ == "__main__":
    main()
