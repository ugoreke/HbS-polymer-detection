"""Data loading: pseudo-label h5s, truth masks, percentile normalization, and
the class-0-aware ``MicroscopyDataset``. Lifted from ``training 2.ipynb`` with
the global ``cfg`` replaced by an explicit ``RunConfig`` argument."""
from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from ablation.config import RunConfig


def load_robust_h5(filepath: str | Path) -> np.ndarray:
    """Squeezes/unwraps the various shapes Ilastik emits into a 2-D array."""
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())
        key = "exported_data" if "exported_data" in keys else ("data" if "data" in keys else keys[0])
        data = f[key][()]
    data = np.squeeze(data)
    if data.ndim > 2:
        if data.shape[-1] < 5:
            data = data[..., 0]
        elif data.shape[0] < 5:
            data = data[0, ...]
    return data


def load_truth_mask(filepath: str | Path, ignore_index: int = 255) -> np.ndarray:
    """Raw 1-based Ilastik label → 0-based training convention with ``ignore_index``
    on unannotated pixels (raw 0)."""
    raw = load_robust_h5(filepath).astype(np.int64)
    out = np.full_like(raw, ignore_index)
    valid = raw > 0
    out[valid] = raw[valid] - 1
    return out


def normalize_image(img_np: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Percentile-clip + scale to [0, 1]. Single source of truth — must match
    inference-time normalization exactly."""
    p = np.percentile(img_np, percentile)
    denom = p if p > 0 else (img_np.max() if img_np.max() > 0 else 1.0)
    return np.clip(img_np / denom, 0, 1).astype(np.float32)


def _basename_no_ext(path: str | Path) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def _glob_raw(raw_dir: Path, exts: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for e in exts:
        out.extend(glob.glob(str(raw_dir / e)))
    return out


def build_truth_pairs(cfg: RunConfig) -> tuple[list[tuple[str, str]], list[tuple[str, str]], set[str]]:
    """Returns (val_pairs, test_pairs, truth_basenames)."""
    truth_files = sorted(glob.glob(str(cfg.truth_dir / "*.h5")))
    if not truth_files:
        return [], [], set()
    raw_files = _glob_raw(cfg.raw_dir, cfg.raw_exts)
    raw_by_stem = {_basename_no_ext(p): p for p in raw_files}

    pairs: list[tuple[str, str]] = []
    truth_stems: set[str] = set()
    for tf in truth_files:
        stem = _basename_no_ext(tf).replace("_segmentation", "")
        if stem not in raw_by_stem:
            print(f"⚠️  Truth file {os.path.basename(tf)} has no matching raw image (stem {stem!r}) — skipping.")
            continue
        pairs.append((raw_by_stem[stem], tf))
        truth_stems.add(stem)

    n_val = min(cfg.truth_val_count, len(pairs))
    return pairs[:n_val], pairs[n_val:], truth_stems


def build_training_pool(cfg: RunConfig, truth_stems: set[str]) -> list[tuple[str, str]]:
    raw_files = sorted(_glob_raw(cfg.raw_dir, cfg.raw_exts))
    seg_files = sorted(glob.glob(str(cfg.seg_dir / "*.h5")))
    if len(raw_files) != len(seg_files) or not raw_files:
        raise RuntimeError(
            f"File mismatch: {len(raw_files)} raw images vs {len(seg_files)} segmentations "
            f"under {cfg.raw_dir} / {cfg.seg_dir}."
        )
    pool = [(r, s) for r, s in zip(raw_files, seg_files) if _basename_no_ext(r) not in truth_stems]
    return pool


class MicroscopyDataset(Dataset):
    """Class-0-aware crop sampler. Training: returns ``steps_per_epoch * batch_size``
    crops per epoch; validation: returns full-size images one at a time."""

    def __init__(
        self,
        file_pairs: list[tuple[str, str]],
        cfg: RunConfig,
        is_train: bool = True,
        mask_loader: Callable[[str | Path], np.ndarray] = load_robust_h5,
    ) -> None:
        self.cfg = cfg
        self.is_train = is_train
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.class0_centers: list[tuple[np.ndarray, np.ndarray]] = []

        half = cfg.tile_size // 2
        target_cls = cfg.class0_target
        for img_path, mask_path in file_pairs:
            img_arr = normalize_image(
                np.array(Image.open(img_path).convert("L"), dtype=np.float32),
                percentile=cfg.norm_percentile,
            )
            if mask_loader is load_truth_mask:
                mask_arr = load_truth_mask(mask_path, ignore_index=cfg.ignore_index).astype(np.int64)
            else:
                mask_arr = mask_loader(mask_path).astype(np.int64)
            self.images.append(img_arr)
            self.masks.append(mask_arr)
            if is_train:
                h, w = mask_arr.shape
                hits = (mask_arr == target_cls)
                if h > cfg.tile_size and w > cfg.tile_size:
                    hits[:half, :] = False
                    hits[h - half:, :] = False
                    hits[:, :half] = False
                    hits[:, w - half:] = False
                rows, cols = np.where(hits)
                self.class0_centers.append((rows, cols))
            else:
                self.class0_centers.append((np.array([]), np.array([])))

    def __len__(self) -> int:
        return self.cfg.batch_size * self.cfg.steps_per_epoch if self.is_train else len(self.images)

    def __getitem__(self, idx):
        cfg = self.cfg
        if self.is_train:
            img_idx = random.randint(0, len(self.images) - 1)
            image, mask = self.images[img_idx], self.masks[img_idx]
            h, w = image.shape
            half = cfg.tile_size // 2
            rows, cols = self.class0_centers[img_idx]
            use_biased = len(rows) > 0 and random.random() < cfg.class0_crop_prob
            if use_biased:
                k = random.randint(0, len(rows) - 1)
                cy, cx = int(rows[k]), int(cols[k])
                top = max(0, min(cy - half, h - cfg.tile_size))
                left = max(0, min(cx - half, w - cfg.tile_size))
            else:
                top = random.randint(0, h - cfg.tile_size)
                left = random.randint(0, w - cfg.tile_size)
            img_crop = image[top:top + cfg.tile_size, left:left + cfg.tile_size]
            mask_crop = mask[top:top + cfg.tile_size, left:left + cfg.tile_size]
            img_t = torch.from_numpy(img_crop).float().unsqueeze(0)
            mask_t = torch.from_numpy(mask_crop).float().unsqueeze(0)
            if random.random() > 0.5:
                img_t, mask_t = TF.hflip(img_t), TF.hflip(mask_t)
            if random.random() > 0.5:
                img_t, mask_t = TF.vflip(img_t), TF.vflip(mask_t)
            rot = random.choice([0, 90, 180, 270])
            if rot > 0:
                img_t = TF.rotate(img_t, rot)
                mask_t = TF.rotate(mask_t, rot, interpolation=transforms.InterpolationMode.NEAREST)
            return img_t, mask_t.long().squeeze(0)
        else:
            image, mask = self.images[idx], self.masks[idx]
            return torch.from_numpy(image).float().unsqueeze(0), torch.from_numpy(mask).long()
