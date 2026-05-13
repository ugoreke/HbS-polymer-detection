"""Composite loss = WeightedDice + CE + CONFUSION_WEIGHT * directed_confusion
+ FN_WEIGHT * directed_FN + TVERSKY_WEIGHT * Tversky([class 0]).

Identical to the loss used in ``training 2.ipynb`` / the SMP notebooks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ablation.config import RunConfig


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights: torch.Tensor | None = None, smooth: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.weights = weights
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        n_classes = probs.shape[1]
        valid = (targets != self.ignore_index)
        safe = torch.where(valid, targets, torch.zeros_like(targets))
        oh = F.one_hot(safe, num_classes=n_classes).permute(0, 3, 1, 2).float()
        vf = valid.unsqueeze(1).float()
        oh = oh * vf
        probs = probs * vf
        total = 0.0
        for c in range(n_classes):
            p = probs[:, c].reshape(-1)
            t = oh[:, c].reshape(-1)
            inter = (p * t).sum()
            dice = (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            total = total + (1 - dice) * (self.weights[c] if self.weights is not None else 1)
        return total / (self.weights.sum() if self.weights is not None else n_classes)


class TverskyLoss(nn.Module):
    def __init__(self, class_indices: list[int], alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.class_indices = class_indices
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        n_classes = probs.shape[1]
        valid = (targets != self.ignore_index)
        safe = torch.where(valid, targets, torch.zeros_like(targets))
        oh = F.one_hot(safe, num_classes=n_classes).permute(0, 3, 1, 2).float()
        vf = valid.unsqueeze(1).float()
        oh = oh * vf
        probs = probs * vf
        loss = 0.0
        for c in self.class_indices:
            p = probs[:, c].reshape(-1)
            t = oh[:, c].reshape(-1)
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            loss = loss + (1 - tv)
        return loss / max(len(self.class_indices), 1)


def build_class_weights(cfg: RunConfig, device: torch.device) -> torch.Tensor:
    w = torch.ones(cfg.n_classes, dtype=torch.float32, device=device)
    for cls_idx, weight in cfg.boosted_classes.items():
        if cls_idx < cfg.n_classes:
            w[cls_idx] = weight
    return w


class CompositeLoss(nn.Module):
    """Closes over a RunConfig and the resolved class-weight tensor."""

    def __init__(self, cfg: RunConfig, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        self.class_weights = class_weights
        self.dice = WeightedDiceLoss(weights=class_weights, ignore_index=cfg.ignore_index)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg.ignore_index)
        self.tversky = TverskyLoss(
            class_indices=cfg.tversky_classes,
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            ignore_index=cfg.ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        probs = torch.softmax(logits, dim=1)
        valid = (targets != cfg.ignore_index)
        safe = torch.where(valid, targets, torch.zeros_like(targets))
        oh = F.one_hot(safe, num_classes=cfg.n_classes).permute(0, 3, 1, 2).float()
        oh = oh * valid.unsqueeze(1).float()

        confusion = 0.0
        for pred_c, true_c, w in cfg.directed_confusion_penalty:
            confusion = confusion + w * (probs[:, pred_c] * oh[:, true_c]).mean()

        fn_loss = 0.0
        for pred_c, true_c in cfg.directed_fn_penalty:
            fn_loss = fn_loss + (probs[:, pred_c] * oh[:, true_c]).mean()

        return (
            self.dice(logits, targets)
            + self.ce(logits, targets)
            + cfg.confusion_weight * confusion
            + cfg.fn_weight * fn_loss
            + cfg.tversky_weight * self.tversky(logits, targets)
        )
