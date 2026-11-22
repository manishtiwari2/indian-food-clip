"""D13 / E7: full fine-tune of one CLIP image encoder on the Indian train split, evaluate on test.

The head starts as the zero-shot classifier (config finetune.language + finetune.template), so epoch 0 is
zero-shot. The best epoch is picked on a validation part of the train split, and the test set is used once
at the end.

Post-hoc choices (D13, as in the report): the head language and template (romanised Hindi names, food template)
were chosen from the zero-shot test results. The hyperparameters are fixed; no hyperparameter search is
recorded in the repo. It is a single run with one seed.

What it does: unlike the linear probe (E5), all image-encoder weights are trained here. The text encoder is
used once to build the starting head and is then dropped.
Experiment: E7 (full fine-tuning).
Report sections: Methods "Full fine-tuning"; Results "Full fine-tuning".

  python scripts/finetune_clip.py --run-name vitb16

Writes results/finetune.csv, results/finetune_history.csv, results/finetune_per_class.csv

How it connects to other steps: it needs the split CSVs from prepare_data.py and the raw images (it reads
images directly, not the cached features, because the encoder weights change every step so cached
features would be stale). It reuses indian_names() from run_zeroshot.py so the class names and prompts match
the zero-shot experiment exactly. make_figures.py does not use this CSV; the report numbers are copied from it.

Why fine-tune at all: the linear probe (E5) only learns a small layer on top of frozen CLIP features.
Fine-tuning also adapts the image encoder itself, which is more powerful but needs more compute and can
overfit when there are few images. Comparing E5 and E7 shows how much that extra freedom buys.

Interview note: "epoch 0 = zero-shot" is a useful sanity check. Epoch-0 validation accuracy should be roughly
in the range of the zero-shot accuracy for the same language and template. It is measured on the 225
validation images, not on the test set, so it is not the same number as the E2-E4 test result.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the library in src/ importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# numpy: index arrays and masks. open_clip: loads CLIP models, image transforms and tokenizers. torch: training.
import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
# indian_names (from scripts/run_zeroshot.py): class names in the chosen language, so prompts match E2-E4.
from run_zeroshot import indian_names  # noqa: E402

# predict: run a model over a DataLoader, returns (scores (n, C), labels (n,)) as NumPy arrays, no training.
# train_one_epoch: one training pass (forward, loss, backward, optimizer step); returns (mean loss, train acc).
from foodclip.cnn import predict, train_one_epoch  # noqa: E402
from foodclip.config import base_parser, config_from_args, resolve  # noqa: E402
# ImageDataset: (image path, label, transform) dataset. load_dataset: split CSV -> (paths, table, classes).
# train_val_split: stratified split of the train rows into train and validation row numbers.
from foodclip.data import ImageDataset, load_dataset, train_val_split  # noqa: E402
# encode_text: class prompts -> (C, D) float32 array, one unit-length text vector per class.
from foodclip.features import encode_text  # noqa: E402
# ClipClassifier: CLIP image encoder + linear head initialised from the text vectors (src/foodclip/finetune.py).
from foodclip.finetune import ClipClassifier  # noqa: E402
from foodclip.io import get_device, run_meta, set_seed, write_rows  # noqa: E402
# topk_accuracy: share of images with the true class in the top k scores. summarize: dict with n_test, top1,
# top5, macro_f1. per_class_accuracy: DataFrame with columns class, n, accuracy (worst class first).
from foodclip.metrics import per_class_accuracy, summarize, topk_accuracy  # noqa: E402
# build_prompts: class names + template name -> list (one entry per class) of prompt-string lists.
from foodclip.prompts import build_prompts  # noqa: E402


def main() -> None:
    """Fine-tune the CLIP image encoder, keep the best-validation epoch, and test it once.

    Steps: (1) load config and data, (2) build the model with a zero-shot head, (3) train epoch by epoch
    while tracking validation accuracy, (4) reload the best epoch, (5) score the test set once and append
    the results to three CSVs. Inputs: config + split CSVs + images. Outputs: the CSVs and a checkpoint.
    """
    # Read options and config; f = fine-tune settings, m = which CLIP model to fine-tune.
    # base_parser adds the shared --config / --set flags; --run-name labels this run in the CSVs
    # so several runs (for example different models) can share the same result files.
    p = base_parser(__doc__)
    p.add_argument("--run-name", default="vitb16")
    # parse_args reads sys.argv into a namespace; config_from_args loads the YAML and applies --set overrides.
    args = p.parse_args()
    cfg = config_from_args(args)
    # f = the finetune config block, m = its model entry (name + pretrained weights tag).
    f, m = cfg["finetune"], cfg["finetune"]["model"]
    # Fix all random number generators so the run can be repeated; the seed comes from the config.
    set_seed(cfg["seed"])
    # Pick the GPU if there is one, otherwise the CPU. In practice this needs a GPU (the reported run used a T4).
    device = get_device()

    # Load the Indian split; hold out a validation part of train to choose the best epoch (test stays unseen).
    # paths = image file paths, df = table with label "y" and "split" columns, classes = class names in order.
    paths, df, classes = load_dataset(cfg, "indian")
    # Shapes: paths, y and split all have shape (n_images,); y = integer class label per image;
    # split = "train" or "test" per image, fixed by the committed split CSV.
    paths = np.asarray(paths, dtype=object)  # object array so we can index it with an index array
    y = df["y"].to_numpy()  # integer class label per image
    split = df["split"].to_numpy()  # "train" or "test" per image, fixed by the committed split CSV
    # Split only the train rows into train/val, stratified by class; test rows are never touched here.
    # np.nonzero(mask)[0] = row numbers where the mask is True; train_val_split returns two row-number arrays.
    tr_idx, val_idx = train_val_split(np.nonzero(split == "train")[0], y, f["val_fraction"], len(classes), cfg["seed"])
    # Indices of the held-out test images; used only once at the very end.
    test_idx = np.nonzero(split == "test")[0]

    # Load CLIP with its own train (augmenting) and eval image transforms.
    # Each CLIP model needs its own resize/normalise recipe, so we take it from open_clip with the weights.
    # create_model_and_transforms returns (model, train transform, eval transform) and downloads the pretrained
    # weights named by m["pretrained"] the first time; device= puts the model on the GPU/CPU.
    clip, train_tf, eval_tf = open_clip.create_model_and_transforms(m["name"], pretrained=m["pretrained"], device=device)
    # Encode one text embedding per class; these become the starting weights of the classifier head.
    # First turn class names into sentences (for example "a photo of poha, a type of food."), then embed them.
    prompts = build_prompts(indian_names(cfg, classes, f["language"]), f["template"])
    # clip.eval() switches off training-only behaviour (returns the same model); get_tokenizer(name) returns the
    # tokenizer that turns text into token ids for this model; encode_text returns a NumPy float32 array.
    text = encode_text(clip.eval(), open_clip.get_tokenizer(m["name"]), prompts, device)  # (n_classes, dim)
    # Image encoder + head from text embeddings, scaled by CLIP's learned temperature, so epoch 0 = zero-shot.
    # clip.logit_scale is CLIP's learned temperature stored as a log value; .exp() turns it into the multiplier
    # applied to cosine similarities; float() makes it a plain Python number. clip.visual = the image encoder only.
    # .to(device) moves the new model (encoder + head) to the GPU/CPU.
    model = ClipClassifier(clip.visual, text, float(clip.logit_scale.exp())).to(device)
    del clip  # the text encoder is not trained
    # (deleting it frees GPU memory; only the image encoder lives on inside `model`)

    def loader(idx, train):
        """Build a DataLoader for the given images; use CLIP's train transform and shuffle only for training.

        idx = positions of the images to use; train = True for the training loader. Returns batches of
        (image tensor, label). Training uses the augmenting transform and random order; val/test use the
        plain eval transform and a fixed order so results are repeatable.
        """
        # Dataset that opens each image and applies the chosen transform on the fly.
        ds = ImageDataset(paths[idx], y[idx], train_tf if train else eval_tf)
        # pin_memory speeds up CPU-to-GPU copies, so it is only turned on for CUDA.
        # DataLoader groups items into batches of images (B, 3, H, W) and labels (B,); shuffle=True mixes the
        # order every epoch; num_workers = parallel loading processes.
        return torch.utils.data.DataLoader(ds, batch_size=f["batch_size"], shuffle=train,
                                           num_workers=f["num_workers"], pin_memory=device.type == "cuda")

    # Data loaders, AdamW over all model weights, cosine learning-rate schedule, mixed precision on GPU.
    train_dl, val_dl, test_dl = loader(tr_idx, True), loader(val_idx, False), loader(test_idx, False)
    # AdamW = Adam with proper weight decay (a penalty on big weights that reduces overfitting).
    # model.parameters() = every trainable weight (image encoder and head); lr and weight_decay come from config.
    opt = torch.optim.AdamW(model.parameters(), lr=f["lr"], weight_decay=f["weight_decay"])
    # Cosine schedule: the learning rate decays from lr to about 0 over the epochs.
    # T_max = number of sched.step() calls for one half cosine wave (here the config epochs).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=f["epochs"])
    # GradScaler is needed for float16 mixed precision (prevents tiny gradients from rounding to zero);
    # it is switched off on CPU.
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Path where the best model weights are saved during training.
    ckpt = resolve(cfg["paths"]["checkpoints"]) / f"finetune_{args.run_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)  # create the checkpoints folder if it is missing
    # history = one dict per epoch; best = best validation accuracy so far (-1 so epoch 0 always beats it).
    history, best, best_epoch = [], -1.0, 0
    # Training loop; epoch 0 only measures the untrained zero-shot head as a starting point.
    for epoch in range(f["epochs"] + 1):  # epoch 0 = zero-shot head, no training
        # Both start as NaN (not-a-number) so the history row shows "no value" at epoch 0.
        loss = train_acc = float("nan")  # stays NaN at epoch 0 because nothing is trained
        if epoch > 0:
            # One full pass over the training data; returns the mean loss and the training accuracy.
            loss, train_acc = train_one_epoch(model, train_dl, opt, device, scaler)
            sched.step()  # move the learning rate one step along the cosine curve
        # Validation accuracy after this epoch. The * unpacks (scores, labels) into topk_accuracy's arguments.
        val_acc = topk_accuracy(*predict(model, val_dl, device), 1)  # predict returns (scores, labels)
        # Keep one history row per epoch.
        history.append({"run": args.run_name, "epoch": epoch, "train_loss": loss, "train_acc": train_acc,
                        "val_acc": val_acc})
        print(f"epoch {epoch:3d} loss {loss:.3f} train {train_acc:.3f} val {val_acc:.3f}")
        # Save weights only when validation accuracy improves (best epoch chosen on validation, not test).
        if val_acc > best:
            best, best_epoch = val_acc, epoch
            # state_dict() = dict of all weight tensors by name; torch.save writes it to the checkpoint file.
            torch.save(model.state_dict(), ckpt)

    # Reload the best checkpoint and evaluate it once on the test split.
    # (The final epoch is not always the best; going back to the best-validation weights avoids overfit epochs.)
    # torch.load reads the saved weight dict (map_location puts the tensors on this device);
    # load_state_dict copies those values into the model's existing weights.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, test_y = predict(model, test_dl, device)  # scores shape (n_test, n_classes)
    # One summary row with the settings, test metrics and run info for traceability.
    # res = absolute path of the results folder.
    res = resolve(cfg["paths"]["results"])
    # **summarize(...) adds n_test, top1, top5, macro_f1; **run_meta() adds run time, torch/Python versions, device.
    row = {"run": args.run_name, "model": m["name"], "pretrained": m["pretrained"], "language": f["language"],
           "template": f["template"], "epochs": f["epochs"], "lr": f["lr"], "batch_size": f["batch_size"],
           "seed": cfg["seed"], "best_epoch": best_epoch, "best_val_acc": best, "n_train": len(tr_idx),
           **summarize(scores, test_y), **run_meta()}
    # Append results so several runs stay in the same CSVs.
    # finetune.csv = one summary row per run; finetune_history.csv = one row per epoch (for learning curves).
    write_rows([row], res / "finetune.csv", append=True)
    write_rows(history, res / "finetune_history.csv", append=True)
    # scores.argmax(1) = predicted class per test image, shape (n_test,). per_class_accuracy returns a DataFrame
    # (columns class, n, accuracy); .assign(run=...) returns a copy with an extra constant column "run".
    pc = per_class_accuracy(test_y, scores.argmax(1), classes).assign(run=args.run_name)  # tag rows with run name
    # finetune_per_class.csv = accuracy of every dish, to see which classes fine-tuning helped.
    # to_dict("records") turns the DataFrame into a list of one dict per row, which write_rows expects.
    write_rows(pc.to_dict("records"), res / "finetune_per_class.csv", append=True)
    # Short console summary of the headline metrics.
    print({k: row[k] for k in ("best_epoch", "top1", "top5", "macro_f1")})


if __name__ == "__main__":
    main()
