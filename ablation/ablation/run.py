"""Orchestrator: ``run_ablation`` creates a versioned run folder, trains
k-folds, runs TTA evaluation, and writes all artifacts."""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

from ablation.config import RunConfig
from ablation.evaluate import evaluate_on_truth_test
from ablation.train import train_kfold


def _next_run_dir(out_root: Path, model_name: str) -> tuple[Path, int]:
    """Returns (folder_path, version_number) using ``{out_root}_{model}_v{N}``
    naming and never overwriting an existing folder."""
    n = 1
    while True:
        candidate = Path(f"{out_root}_{model_name}_v{n}")
        if not candidate.exists():
            return candidate, n
        n += 1


def run_ablation(
    model_name: str,
    params: dict[str, Any] | RunConfig,
    notebook_path: str | Path,
    out_root: str | Path = "evaluation_truth",
) -> Path:
    """Train + evaluate one ablation run.

    Returns the path to the created run folder.
    """
    cfg = RunConfig.from_params(model_name, params)
    out_root_path = Path(out_root)
    run_dir, version = _next_run_dir(out_root_path, model_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"📁 Run folder: {run_dir}")

    (run_dir / "run_config.json").write_text(json.dumps(cfg.to_jsonable(), indent=2, default=str))

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    nb_src = Path(notebook_path)
    if nb_src.exists():
        nb_dst = run_dir / f"ablation_{model_name}_v{version}_{timestamp}.ipynb"
        shutil.copy2(nb_src, nb_dst)
    else:
        print(f"⚠️  Notebook {nb_src} not found — skipping notebook snapshot.")

    ckpt_dir = run_dir / "checkpoints"
    checkpoint_paths = train_kfold(cfg, ckpt_dir)

    run_label = f"{model_name} v{version}"
    result = evaluate_on_truth_test(cfg, checkpoint_paths, run_dir, run_label)
    polymer = result["per_class_mean"][0]
    polymer_std = result["per_class_std"][0]
    print(f"\n🎯 {run_label}: polymer dice {polymer:.3f} ± {polymer_std:.3f}  →  {run_dir}")
    return run_dir
