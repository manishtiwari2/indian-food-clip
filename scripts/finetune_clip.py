"""D13 / E7: full fine-tune of one CLIP image encoder on the Indian train split, evaluate on test.

The head starts as the zero-shot classifier (config finetune.language + finetune.template), so epoch 0 is
zero-shot. The best epoch is picked on a validation part of the train split, never on the test set.

  python scripts/finetune_clip.py --run-name vitb16

Writes results/finetune.csv, results/finetune_history.csv, results/finetune_per_class.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from run_zeroshot import indian_names  # noqa: E402

from foodclip.cnn import predict, train_one_epoch  # noqa: E402
from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
from foodclip.data import ImageDataset, load_dataset, train_val_split  # noqa: E402
from foodclip.features import encode_text  # noqa: E402
from foodclip.finetune import ClipClassifier  # noqa: E402
from foodclip.io import get_device, run_meta, set_seed, write_rows  # noqa: E402
from foodclip.metrics import per_class_accuracy, summarize, topk_accuracy  # noqa: E402
from foodclip.prompts import build_prompts  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--run-name", default="vitb16")
    args = p.parse_args()
    cfg = config_from_args(args)
    f, m = cfg["finetune"], cfg["finetune"]["model"]
    set_seed(cfg["seed"])
    device = get_device()

    paths, df, classes = load_dataset(cfg, "indian")
    paths = np.asarray(paths, dtype=object)
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    tr_idx, val_idx = train_val_split(np.nonzero(split == "train")[0], y, f["val_fraction"], len(classes), cfg["seed"])
    test_idx = np.nonzero(split == "test")[0]

    clip, train_tf, eval_tf = open_clip.create_model_and_transforms(m["name"], pretrained=m["pretrained"], device=device)
    prompts = build_prompts(indian_names(cfg, classes, f["language"]), f["template"])
    text = encode_text(clip.eval(), open_clip.get_tokenizer(m["name"]), prompts, device)
    model = ClipClassifier(clip.visual, text, float(clip.logit_scale.exp())).to(device)
    del clip  # the text encoder is not trained

    def loader(idx, train):
        ds = ImageDataset(paths[idx], y[idx], train_tf if train else eval_tf)
        return torch.utils.data.DataLoader(ds, batch_size=f["batch_size"], shuffle=train,
                                           num_workers=f["num_workers"], pin_memory=device.type == "cuda")

    train_dl, val_dl, test_dl = loader(tr_idx, True), loader(val_idx, False), loader(test_idx, False)
    opt = torch.optim.AdamW(model.parameters(), lr=f["lr"], weight_decay=f["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=f["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt = resolve(cfg["paths"]["checkpoints"]) / f"finetune_{args.run_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch = [], -1.0, 0
    for epoch in range(f["epochs"] + 1):  # epoch 0 = zero-shot head, no training
        loss = train_acc = float("nan")
        if epoch > 0:
            loss, train_acc = train_one_epoch(model, train_dl, opt, device, scaler)
            sched.step()
        val_acc = topk_accuracy(*predict(model, val_dl, device), 1)
        history.append({"run": args.run_name, "epoch": epoch, "train_loss": loss, "train_acc": train_acc,
                        "val_acc": val_acc})
        print(f"epoch {epoch:3d} loss {loss:.3f} train {train_acc:.3f} val {val_acc:.3f}")
        if val_acc > best:
            best, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, test_y = predict(model, test_dl, device)
    res = resolve(cfg["paths"]["results"])
    row = {"run": args.run_name, "model": m["name"], "pretrained": m["pretrained"], "language": f["language"],
           "template": f["template"], "epochs": f["epochs"], "lr": f["lr"], "batch_size": f["batch_size"],
           "seed": cfg["seed"], "best_epoch": best_epoch, "best_val_acc": best, "n_train": len(tr_idx),
           **summarize(scores, test_y), **run_meta()}
    write_rows([row], res / "finetune.csv", append=True)
    write_rows(history, res / "finetune_history.csv", append=True)
    pc = per_class_accuracy(test_y, scores.argmax(1), classes).assign(run=args.run_name)
    write_rows(pc.to_dict("records"), res / "finetune_per_class.csv", append=True)
    print({k: row[k] for k in ("best_epoch", "top1", "top5", "macro_f1")})


if __name__ == "__main__":
    main()
