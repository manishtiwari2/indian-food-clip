"""D13 / E7: full fine-tune of one CLIP image encoder on the Indian train split, evaluate on test.

The head starts as the zero-shot classifier (config finetune.language + finetune.template), so epoch 0 is
zero-shot. The best epoch is picked on a validation part of the train split, never on the test set.

What it does: unlike the linear probe (E5), all image-encoder weights are trained here. The text encoder is
used once to build the starting head and is then dropped.
Experiment: E7 (full fine-tuning).
Report sections: Methods "Full fine-tuning"; Results "Full fine-tuning".

  python scripts/finetune_clip.py --run-name vitb16

Writes results/finetune.csv, results/finetune_history.csv, results/finetune_per_class.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
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
    """Fine-tune the CLIP image encoder, keep the best-validation epoch, and test it once."""
    # Read options and config; f = fine-tune settings, m = which CLIP model to fine-tune.
    p = base_parser(__doc__)
    p.add_argument("--run-name", default="vitb16")
    args = p.parse_args()
    cfg = config_from_args(args)
    f, m = cfg["finetune"], cfg["finetune"]["model"]
    set_seed(cfg["seed"])
    device = get_device()

    # Load the Indian split; hold out a validation part of train to choose the best epoch (test stays unseen).
    paths, df, classes = load_dataset(cfg, "indian")
    paths = np.asarray(paths, dtype=object)  # object array so we can index it with an index array
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    tr_idx, val_idx = train_val_split(np.nonzero(split == "train")[0], y, f["val_fraction"], len(classes), cfg["seed"])
    test_idx = np.nonzero(split == "test")[0]

    # Load CLIP with its own train (augmenting) and eval image transforms.
    clip, train_tf, eval_tf = open_clip.create_model_and_transforms(m["name"], pretrained=m["pretrained"], device=device)
    # Encode one text embedding per class; these become the starting weights of the classifier head.
    prompts = build_prompts(indian_names(cfg, classes, f["language"]), f["template"])
    text = encode_text(clip.eval(), open_clip.get_tokenizer(m["name"]), prompts, device)  # (n_classes, dim)
    # Image encoder + head from text embeddings, scaled by CLIP's learned temperature, so epoch 0 = zero-shot.
    model = ClipClassifier(clip.visual, text, float(clip.logit_scale.exp())).to(device)
    del clip  # the text encoder is not trained

    def loader(idx, train):
        """Build a DataLoader for the given images; use CLIP's train transform and shuffle only for training."""
        ds = ImageDataset(paths[idx], y[idx], train_tf if train else eval_tf)
        return torch.utils.data.DataLoader(ds, batch_size=f["batch_size"], shuffle=train,
                                           num_workers=f["num_workers"], pin_memory=device.type == "cuda")

    # Data loaders, AdamW over all model weights, cosine learning-rate schedule, mixed precision on GPU.
    train_dl, val_dl, test_dl = loader(tr_idx, True), loader(val_idx, False), loader(test_idx, False)
    opt = torch.optim.AdamW(model.parameters(), lr=f["lr"], weight_decay=f["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=f["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Path where the best model weights are saved during training.
    ckpt = resolve(cfg["paths"]["checkpoints"]) / f"finetune_{args.run_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch = [], -1.0, 0
    # Training loop; epoch 0 only measures the untrained zero-shot head as a starting point.
    for epoch in range(f["epochs"] + 1):  # epoch 0 = zero-shot head, no training
        loss = train_acc = float("nan")  # stays NaN at epoch 0 because nothing is trained
        if epoch > 0:
            loss, train_acc = train_one_epoch(model, train_dl, opt, device, scaler)
            sched.step()
        val_acc = topk_accuracy(*predict(model, val_dl, device), 1)  # predict returns (scores, labels)
        # Keep one history row per epoch.
        history.append({"run": args.run_name, "epoch": epoch, "train_loss": loss, "train_acc": train_acc,
                        "val_acc": val_acc})
        print(f"epoch {epoch:3d} loss {loss:.3f} train {train_acc:.3f} val {val_acc:.3f}")
        # Save weights only when validation accuracy improves (best epoch chosen on validation, not test).
        if val_acc > best:
            best, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt)

    # Reload the best checkpoint and evaluate it once on the test split.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, test_y = predict(model, test_dl, device)  # scores shape (n_test, n_classes)
    # One summary row with the settings, test metrics and run info for traceability.
    res = resolve(cfg["paths"]["results"])
    row = {"run": args.run_name, "model": m["name"], "pretrained": m["pretrained"], "language": f["language"],
           "template": f["template"], "epochs": f["epochs"], "lr": f["lr"], "batch_size": f["batch_size"],
           "seed": cfg["seed"], "best_epoch": best_epoch, "best_val_acc": best, "n_train": len(tr_idx),
           **summarize(scores, test_y), **run_meta()}
    # Append results so several runs stay in the same CSVs.
    write_rows([row], res / "finetune.csv", append=True)
    write_rows(history, res / "finetune_history.csv", append=True)
    pc = per_class_accuracy(test_y, scores.argmax(1), classes).assign(run=args.run_name)  # tag rows with run name
    write_rows(pc.to_dict("records"), res / "finetune_per_class.csv", append=True)
    print({k: row[k] for k in ("best_epoch", "top1", "top5", "macro_f1")})


if __name__ == "__main__":
    main()
