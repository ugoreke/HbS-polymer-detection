"""Sliding-window inference + 8-way flip/rotation TTA."""
from __future__ import annotations

import numpy as np
import torch

from ablation.config import RunConfig


@torch.no_grad()
def predict_full_image(model, img_t: torch.Tensor, cfg: RunConfig) -> torch.Tensor:
    """Returns the argmax label map (H, W) as a tensor on the model's device."""
    model.eval()
    _, h, w = img_t.shape
    stride = max(int(cfg.tile_size * (1 - cfg.tile_overlap)), 1)
    prob = torch.zeros((cfg.n_classes, h, w), device=img_t.device)
    cnt = torch.zeros((cfg.n_classes, h, w), device=img_t.device)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y0 = max(0, min(y + cfg.tile_size, h) - cfg.tile_size)
            x0 = max(0, min(x + cfg.tile_size, w) - cfg.tile_size)
            crop = img_t[:, y0:y0 + cfg.tile_size, x0:x0 + cfg.tile_size].unsqueeze(0)
            p = torch.softmax(model(crop), dim=1).squeeze(0)
            prob[:, y0:y0 + cfg.tile_size, x0:x0 + cfg.tile_size] += p
            cnt[:, y0:y0 + cfg.tile_size, x0:x0 + cfg.tile_size] += 1
    return torch.argmax(prob / cnt, dim=0)


@torch.no_grad()
def predict_full_image_tta(model, img_t: torch.Tensor, cfg: RunConfig) -> torch.Tensor:
    """Averages softmax over the 8 D4 transforms then argmaxes."""
    model.eval()
    _, h, w = img_t.shape
    stride = max(int(cfg.tile_size * (1 - cfg.tile_overlap)), 1)
    accum = None
    for k in range(4):
        for flip in (False, True):
            x = torch.rot90(img_t, k=k, dims=(1, 2))
            if flip:
                x = torch.flip(x, dims=(2,))
            _, hh, ww = x.shape
            prob = torch.zeros((cfg.n_classes, hh, ww), device=x.device)
            cnt = torch.zeros((cfg.n_classes, hh, ww), device=x.device)
            for y in range(0, hh, stride):
                for xx in range(0, ww, stride):
                    y0 = max(0, min(y + cfg.tile_size, hh) - cfg.tile_size)
                    x0 = max(0, min(xx + cfg.tile_size, ww) - cfg.tile_size)
                    crop = x[:, y0:y0 + cfg.tile_size, x0:x0 + cfg.tile_size].unsqueeze(0)
                    p = torch.softmax(model(crop), dim=1).squeeze(0)
                    prob[:, y0:y0 + cfg.tile_size, x0:x0 + cfg.tile_size] += p
                    cnt[:, y0:y0 + cfg.tile_size, x0:x0 + cfg.tile_size] += 1
            prob = prob / cnt
            if flip:
                prob = torch.flip(prob, dims=(2,))
            prob = torch.rot90(prob, k=-k, dims=(1, 2))
            accum = prob if accum is None else accum + prob
    return torch.argmax(accum, dim=0)


def per_class_dice(pred: torch.Tensor, gt: torch.Tensor, n_classes: int, ignore_index: int) -> np.ndarray:
    """Per-image dice over n_classes, ignoring pixels where gt == ignore_index."""
    valid = (gt != ignore_index).float()
    out = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        p = (pred == c).float() * valid
        t = (gt == c).float() * valid
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            out[c] = 1.0 if t.sum() == 0 else 0.0
        else:
            out[c] = (2.0 * inter / union).item()
    return out


def confusion_one(pred: torch.Tensor, gt: torch.Tensor, n_classes: int, ignore_index: int) -> np.ndarray:
    """Per-image (n, n) integer confusion matrix; rows = true, cols = pred."""
    p = pred.cpu().numpy().ravel() if torch.is_tensor(pred) else pred.ravel()
    g = gt.cpu().numpy().ravel() if torch.is_tensor(gt) else gt.ravel()
    valid = (g >= 0) & (g < n_classes)
    p, g = p[valid], g[valid]
    idx = g * n_classes + p
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
