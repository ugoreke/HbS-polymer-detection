"""Ablation harness for polymer/cell segmentation.

One function (``run_ablation``) drives a full k-fold train + TTA eval and
writes a self-contained ``evaluation_truth_{model}_v{N}/`` folder.
"""
from ablation.config import RunConfig
from ablation.run import run_ablation

__all__ = ["RunConfig", "run_ablation"]
