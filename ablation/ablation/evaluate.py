"""TTA evaluation on the held-out truth-test split.

Writes:
  - confusion_matrix.svg            row-normalized, editable text
  - per_image_dice.csv              one row per (fold, image)
  - dice_scores.md                  config + per-class + per-image + confusion
  - predictions_h5/*.h5             one mask per (fold, image)
  - predictions_viz/*.jpg           palette-coloured visualisation of each h5
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from ablation.config import RunConfig
from ablation.data import MicroscopyDataset, build_truth_pairs, load_robust_h5
from ablation.inference import (
    confusion_one,
    per_class_dice,
    predict_full_image,
    predict_full_image_tta,
)
from ablation.models import model_factory
from ablation.viz import plot_confusion_matrix, save_colored_jpg


def _save_pred_h5(path: Path, pred_mask: np.ndarray) -> None:
    """Save the predicted argmax mask as a 2-D uint8 h5 (matches the
    convention used by the existing colour-mapper)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("exported_data", data=pred_mask.astype(np.uint8))


def _write_dice_md(
    out_path: Path,
    cfg: RunConfig,
    run_label: str,
    means: np.ndarray,
    stds: np.ndarray,
    per_image_records: list[dict],
    confusion_total: np.ndarray,
) -> None:
    n = cfg.n_classes
    cls_name = cfg.class_names
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [f"# Run: {run_label}      {now}", ""]

    cfg_d = cfg.to_jsonable()
    lines += ["## Config", "", "| param | value |", "|---|---|"]
    for k in sorted(cfg_d.keys()):
        v = cfg_d[k]
        if isinstance(v, (dict, list, tuple)):
            v = json.dumps(v, default=str)
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines += [
        "## Per-class dice (mean ± std across folds × truth-test images)",
        "",
        "| Class | Dice |",
        "|---|---|",
    ]
    for c in range(n):
        lines.append(f"| {cls_name.get(c, f'Class {c}')} ({c}) | {means[c]:.3f} ± {stds[c]:.3f} |")
    lines.append("")

    lines += ["## Per-image dice", "", "| fold | image | " + " | ".join(f"dice_{c}" for c in range(n)) + " |"]
    lines.append("|" + "---|" * (2 + n))
    for r in per_image_records:
        cells = [str(r["fold"]), r["image"]] + [f"{r[f'dice_class_{c}']:.4f}" for c in range(n)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Confusion counts (rows = true, cols = predicted)", "",
              "|       | " + " | ".join(f"pred_{c}" for c in range(n)) + " |"]
    lines.append("|" + "---|" * (n + 1))
    for i in range(n):
        row = [f"true_{i}"] + [str(int(confusion_total[i, j])) for j in range(n)]
        lines.append("| " + " | ".join(row) + " |")

    out_path.write_text("\n".join(lines))


def evaluate_on_truth_test(cfg: RunConfig, checkpoint_paths: list[Path], out_dir: Path, run_label: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_h5_dir = out_dir / "predictions_h5"
    pred_viz_dir = out_dir / "predictions_viz"
    pred_h5_dir.mkdir(exist_ok=True)
    pred_viz_dir.mkdir(exist_ok=True)

    _, test_pairs, _ = build_truth_pairs(cfg)
    if not test_pairs:
        raise RuntimeError("No held-out truth-test files; lower truth_val_count or add truth labels.")
    print(f"🧪 Eval on {len(test_pairs)} truth-test image(s) × {len(checkpoint_paths)} folds.")

    device = cfg.torch_device
    test_dataset = MicroscopyDataset(test_pairs, cfg, is_train=False, mask_loader=load_robust_h5)

    per_image: list[dict] = []
    dice_rows: list[np.ndarray] = []
    confusion_total = np.zeros((cfg.n_classes, cfg.n_classes), dtype=np.int64)

    for fold_idx, ckpt in enumerate(checkpoint_paths):
        model = model_factory(cfg.model_name, cfg.n_classes, cfg.in_channels).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        for img_idx in range(len(test_dataset)):
            img_t, gt = test_dataset[img_idx]
            img_t = img_t.to(device)
            pred = predict_full_image_tta(model, img_t, cfg) if cfg.tta else predict_full_image(model, img_t, cfg)

            dices = per_class_dice(pred, gt.to(pred.device), cfg.n_classes, cfg.ignore_index)
            dice_rows.append(dices)
            confusion_total += confusion_one(pred, gt, cfg.n_classes, cfg.ignore_index)

            img_stem = os.path.splitext(os.path.basename(test_pairs[img_idx][1]))[0]
            tag = f"fold{fold_idx + 1}_{img_stem}"
            pred_np = pred.detach().cpu().numpy()
            _save_pred_h5(pred_h5_dir / f"{tag}.h5", pred_np)

            rec = {
                "fold": fold_idx + 1,
                "checkpoint": os.path.basename(str(ckpt)),
                "image": os.path.basename(test_pairs[img_idx][1]),
            }
            for c in range(cfg.n_classes):
                rec[f"dice_class_{c}"] = float(dices[c])
            per_image.append(rec)

    save_colored_jpg(pred_h5_dir, pred_viz_dir)

    dice_mat = np.array(dice_rows)
    means = dice_mat.mean(axis=0)
    stds = dice_mat.std(axis=0)

    with open(out_dir / "per_image_dice.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    plot_confusion_matrix(confusion_total, cfg.class_names, out_dir / "confusion_matrix.svg")
    _write_dice_md(out_dir / "dice_scores.md", cfg, run_label, means, stds, per_image, confusion_total)

    print(f"\n=== {run_label}: held-out truth-test dice ===")
    for c in range(cfg.n_classes):
        print(f"  {cfg.class_names.get(c, f'Class {c}'):20s}: {means[c]:.3f} ± {stds[c]:.3f}")

    return {
        "per_class_mean": means,
        "per_class_std": stds,
        "confusion_counts": confusion_total,
        "per_image_records": per_image,
    }
