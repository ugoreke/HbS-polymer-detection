"""RunConfig: dataclass that replaces the ad-hoc Config class in the
training notebooks. One ``RunConfig`` fully specifies a single ablation run
(architecture + loss + optimizer + selector + paths)."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch


def _autodetect_base_dir() -> Path:
    if sys.platform.startswith("win"):
        return Path(r"E:\utku g leica\sickling")
    return Path("/Users/utkugoreke/anaconda_projects/sickling")


def _autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class RunConfig:
    # --- Architecture ---
    model_name: str = "unet_vanilla"
    n_classes: int = 4
    in_channels: int = 1

    # --- Paths (auto-resolved relative to base_dir) ---
    base_dir: Path = field(default_factory=_autodetect_base_dir)
    raw_subdir: str = "trainingImages"
    seg_subdir: str = "h5_fixed_for_training"
    truth_subdir: str = "dense_segmentations/truth"
    raw_exts: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    truth_val_count: int = 2

    # --- Tiling / sampling ---
    tile_size: int = 256
    batch_size: int = 16
    steps_per_epoch: int = 100
    n_folds: int = 5
    kfold_random_state: int = 42
    class0_crop_prob: float = 0.5
    class0_target: int = 0
    ignore_index: int = 255
    norm_percentile: float = 99.0

    # --- Optimizer / schedule ---
    epochs: int = 50
    lr: float = 1.0e-4
    encoder_lr_mult: float = 1.0     # 0.1 for SMP discriminative-LR runs
    freeze_epochs: int = 0           # >0 for encoder-freeze warmup (SMP only)

    # --- Class weights ---
    boosted_classes: dict[int, float] = field(default_factory=lambda: {0: 5.0, 3: 3.0})

    # --- Composite loss knobs ---
    # Each entry is (pred_class, true_class, weight). Note the loss applies
    # `weight * (probs[:, pred] * one_hot[:, true]).mean()`.
    directed_confusion_penalty: list[tuple[int, int, float]] = field(
        default_factory=lambda: [(0, 1, 2.0), (0, 2, 1.0), (0, 3, 1.0)]
    )
    directed_fn_penalty: list[tuple[int, int]] = field(
        default_factory=lambda: [(1, 0), (2, 0), (3, 0)]
    )
    confusion_weight: float = 0.3
    fn_weight: float = 0.1
    tversky_classes: list[int] = field(default_factory=lambda: [0])
    tversky_weight: float = 0.3
    tversky_alpha: float = 0.4
    tversky_beta: float = 0.6

    # --- Checkpoint selector ---
    selector: str = "mean"                  # "mean" or "floor"
    ckpt_floor_threshold: float = 0.70
    ckpt_floor_penalty_weight: float = 0.5

    # --- Eval ---
    tta: bool = True
    tile_overlap: float = 0.5

    # --- Class display names ---
    class_names: dict[int, str] = field(
        default_factory=lambda: {0: "Polymer", 1: "Background", 2: "Cell", 3: "Cell boundary"}
    )

    # --- Device (string so JSON-serialisable) ---
    device: str = field(default_factory=_autodetect_device)

    # ---------- helpers ----------
    @property
    def raw_dir(self) -> Path:
        return self.base_dir / self.raw_subdir

    @property
    def seg_dir(self) -> Path:
        return self.base_dir / self.seg_subdir

    @property
    def truth_dir(self) -> Path:
        return self.base_dir / self.truth_subdir

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["base_dir"] = str(self.base_dir)
        # tuple → list for JSON
        d["raw_exts"] = list(self.raw_exts)
        # int keys → str for JSON
        d["boosted_classes"] = {str(k): v for k, v in self.boosted_classes.items()}
        d["class_names"] = {str(k): v for k, v in self.class_names.items()}
        return d

    @classmethod
    def from_params(cls, model_name: str, params: dict[str, Any] | "RunConfig") -> "RunConfig":
        if isinstance(params, cls):
            return params
        kwargs: dict[str, Any] = dict(params or {})
        kwargs["model_name"] = model_name
        # boosted_classes int-keying after JSON round-trips
        bc = kwargs.get("boosted_classes")
        if isinstance(bc, dict):
            kwargs["boosted_classes"] = {int(k): float(v) for k, v in bc.items()}
        if "base_dir" in kwargs and not isinstance(kwargs["base_dir"], Path):
            kwargs["base_dir"] = Path(kwargs["base_dir"])
        return cls(**kwargs)
