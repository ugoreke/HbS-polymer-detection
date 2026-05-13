"""K-fold training driver. One loop for all four model variants — the
per-model bits (single vs discriminative LR, optional encoder freeze warmup,
mean-of-boosted vs floor-penalty checkpoint selector) are driven from the
RunConfig."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from ablation.config import RunConfig
from ablation.data import (
    MicroscopyDataset,
    build_training_pool,
    build_truth_pairs,
    load_robust_h5,
)
from ablation.inference import per_class_dice, predict_full_image
from ablation.losses import CompositeLoss, build_class_weights
from ablation.models import build_optimizer, is_smp, model_factory, set_encoder_trainable


def _ckpt_score(avg_class_dice: np.ndarray, cfg: RunConfig) -> float:
    boosted = [i for i in cfg.boosted_classes.keys() if i < cfg.n_classes]
    base = float(np.mean([avg_class_dice[i] for i in boosted])) if boosted else float(np.mean(avg_class_dice))
    if cfg.selector == "mean":
        return base
    if cfg.selector == "floor":
        floor = float(np.min(avg_class_dice))
        penalty = cfg.ckpt_floor_penalty_weight * max(0.0, cfg.ckpt_floor_threshold - floor)
        return base - penalty
    raise ValueError(f"Unknown selector: {cfg.selector!r}")


def train_kfold(cfg: RunConfig, checkpoint_dir: Path) -> list[Path]:
    """Train all N folds, return the list of best-checkpoint paths (one per fold)."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    val_pairs, test_pairs, truth_stems = build_truth_pairs(cfg)
    if not val_pairs:
        raise RuntimeError(f"No truth files found in {cfg.truth_dir}.")
    train_pool = build_training_pool(cfg, truth_stems)

    print(f"📊 Truth split: {len(val_pairs)} val, {len(test_pairs)} test.")
    print(f"🧪 Training pool: {len(train_pool)} images (truth basenames excluded).")

    device = cfg.torch_device
    class_weights = build_class_weights(cfg, device)
    val_dataset = MicroscopyDataset(val_pairs, cfg, is_train=False, mask_loader=load_robust_h5)
    val_loader = DataLoader(val_dataset, batch_size=1)

    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.kfold_random_state)
    ckpt_paths: list[Path] = []

    for fold, (train_idx, _) in enumerate(kf.split(train_pool)):
        print(f"\n{'='*20} FOLD {fold + 1}/{cfg.n_folds} {'='*20}")
        train_loader = DataLoader(
            MicroscopyDataset([train_pool[i] for i in train_idx], cfg, is_train=True),
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        model = model_factory(cfg.model_name, cfg.n_classes, cfg.in_channels).to(device)
        optimizer = build_optimizer(model, cfg)
        criterion = CompositeLoss(cfg, class_weights).to(device)

        use_freeze = is_smp(cfg.model_name) and cfg.freeze_epochs > 0
        if use_freeze:
            set_encoder_trainable(model, False)

        ckpt_path = checkpoint_dir / f"unet_fold_{fold + 1}_best.pth"
        best = -1e9 if cfg.selector == "floor" else 0.0

        for epoch in range(cfg.epochs):
            if use_freeze and epoch == cfg.freeze_epochs:
                set_encoder_trainable(model, True)

            model.train()
            epoch_loss = 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            val_dice_sums = np.zeros(cfg.n_classes)
            for v_img, v_mask in val_loader:
                v_img = v_img.to(device)
                v_mask = v_mask.to(device)
                pred = predict_full_image(model, v_img.squeeze(0), cfg)
                val_dice_sums += per_class_dice(pred, v_mask.squeeze(0), cfg.n_classes, cfg.ignore_index)
            avg = val_dice_sums / max(len(val_loader), 1)
            score = _ckpt_score(avg, cfg)

            if (epoch + 1) % 5 == 0:
                parts = [f"Fold {fold + 1} Ep {epoch + 1} | loss {epoch_loss / max(len(train_loader), 1):.4f}"]
                for c in range(cfg.n_classes):
                    parts.append(f"{cfg.class_names.get(c, c)}: {avg[c]:.3f}")
                parts.append(f"score={score:.4f}")
                print(" | ".join(parts))

            if score > best:
                best = score
                torch.save(model.state_dict(), ckpt_path)

        print(f"✅ Fold {fold + 1} best score: {best:.4f}  →  {ckpt_path}")
        ckpt_paths.append(ckpt_path)

    return ckpt_paths
