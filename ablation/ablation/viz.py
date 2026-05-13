"""Viz helpers: SVG confusion matrix (editable text) + palette-colored mask JPGs."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Same palette as `save_colored_jpg` in training 2.ipynb.
PALETTE = [
    0, 0, 0,
    255, 224, 0,
    0, 130, 206,
    251, 0, 70,
    0, 230, 35,
    255, 0, 255,
    0, 255, 255,
    128, 0, 0,
] + [0, 0, 0] * 248


def _load_h5_array(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
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


def save_colored_jpg(input_path: str | Path, output_dir: str | Path) -> None:
    """Convert an h5 mask (or every h5 in a directory) to palette-coloured JPGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    files = list(input_path.glob("*.h5")) if input_path.is_dir() else [input_path]
    if not files:
        print(f"⚠️ No .h5 files in {input_path}")
        return
    for h5_path in files:
        data = _load_h5_array(h5_path)
        if data.max() > 255:
            data = data % 255
        img = Image.fromarray(data.astype(np.uint8), mode="P")
        img.putpalette(PALETTE)
        out = output_dir / (h5_path.stem + ".jpg")
        img.convert("RGB").save(out, quality=90)


def plot_confusion_matrix(
    cm_counts: np.ndarray,
    class_names: dict[int, str] | list[str],
    save_path: str | Path,
) -> None:
    """Row-normalized confusion matrix saved as SVG with editable text
    (svg.fonttype = 'none' so Illustrator keeps it as live text)."""
    mpl.rcParams["svg.fonttype"] = "none"
    n = cm_counts.shape[0]
    row_sums = cm_counts.sum(axis=1, keepdims=True).clip(min=1)
    cm = cm_counts / row_sums
    labels = (
        [class_names[i] for i in range(n)] if isinstance(class_names, dict)
        else list(class_names)
    )

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.text(j, i, f"{v * 100:.1f}%", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Fraction of true-class pixels")
    fig.tight_layout()
    fig.savefig(save_path, format="svg")
    plt.close(fig)
