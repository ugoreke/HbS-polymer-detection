# Ablation harness — polymer / cell segmentation

Reproduces the loss × architecture sweep behind the polymer-segmentation
figure. One notebook (`ablation.ipynb`) is a thin orchestrator: each cell is
a single `run_ablation(...)` call. The heavy lifting lives in the `ablation/`
Python package.

Each run trains all `n_folds` (default 5) on the pseudo-label pool, runs
8-way TTA on the held-out truth-test images, and writes a self-contained
versioned folder so re-running never overwrites previous results.

## What gets produced

Every call to `run_ablation(...)` creates:

```
evaluation_truth_{model_name}_v{N}/
  ablation_{model_name}_v{N}_{YYYYMMDD-HHMMSS}.ipynb   # snapshot of the notebook
  run_config.json                                       # resolved RunConfig
  confusion_matrix.svg                                  # row-normalized, editable text
  dice_scores.md                                        # config + per-class + per-image + confusion
  per_image_dice.csv
  predictions_h5/    # one .h5 per (fold × truth-test image)
  predictions_viz/   # palette-coloured .jpg for each .h5
  checkpoints/       # unet_fold_*_best.pth, scoped to this run
```

`{N}` auto-increments — re-running `unet_vanilla` produces
`..._v2`, `..._v3`, etc.

## Model factory keys

| key                       | architecture                                                              |
|---------------------------|---------------------------------------------------------------------------|
| `unet_vanilla`            | from-scratch 1-channel U-Net (4 down/up, BN+ReLU)                         |
| `smp_unet_resnet34`       | `smp.Unet(encoder='resnet34', encoder_weights='imagenet')`                |
| `smp_unet_effb0_scse`     | `smp.Unet(encoder='efficientnet-b0', weights='imagenet', attn='scse')`    |
| `smp_unet_effb7_scse`     | `smp.Unet(encoder='efficientnet-b7', weights='imagenet', attn='scse')`    |

Vanilla U-Net uses a single LR; the SMP variants use a discriminative LR
(encoder at `lr * encoder_lr_mult`) and optionally a `freeze_epochs` warmup.

## Checkpoint selectors

- `selector="mean"` — mean dice over `boosted_classes`.
- `selector="floor"` — same base, minus
  `ckpt_floor_penalty_weight * max(0, ckpt_floor_threshold - min(class_dice))`.
  Defaults reproduce the EfficientNet runs: `threshold=0.70, weight=0.5`.

## Pre-filled runs A–F

The notebook ships one commented-out cell per run from
`evaluationSummary.md`. Uncomment whichever you want to reproduce.

| Run | model_name              | Tversky (α, β, w) | Other knobs                                                  |
|-----|-------------------------|-------------------|--------------------------------------------------------------|
| A   | `unet_vanilla`          | 0.3 / 0.7 / 1.0   | FN_W=0.3, CONF_W=0.2, symmetric (0,*) penalty, selector=mean |
| B   | `unet_vanilla`          | 0.4 / 0.6 / 0.3   | FN_W=0.1, CONF_W=0.3, asym (0,1)=2.0, selector=mean          |
| C   | `smp_unet_resnet34`     | 0.4 / 0.6 / 0.3   | LR=5e-5, freeze_epochs=5, selector=mean                      |
| D   | `smp_unet_effb0_scse`   | 0.4 / 0.6 / 0.3   | LR=5e-5, encoder_lr_mult=0.1, selector=floor                 |
| E   | `smp_unet_effb0_scse`   | 0.4 / 0.7 / 0.5   | same as D + recall push                                      |
| F   | `smp_unet_effb7_scse`   | 0.4 / 0.7 / 0.5   | same as E                                                    |

Run B is the figure model: polymer dice 0.606 ± 0.033 on 10 held-out
measurements (5 folds × 2 truth-test images).

## Layout

```
ablation/
├── ablation.ipynb           # orchestrator notebook (cells A–F pre-filled, commented)
└── ablation/                # the package
    ├── __init__.py          # re-exports RunConfig, run_ablation
    ├── config.py            # @dataclass RunConfig
    ├── data.py              # MicroscopyDataset, load_robust_h5, load_truth_mask,
    │                        # normalize_image, build_truth_pairs
    ├── models.py            # model_factory, build_optimizer, freeze/unfreeze
    ├── losses.py            # WeightedDiceLoss, TverskyLoss, CompositeLoss
    ├── train.py             # train_kfold(cfg, ckpt_dir) -> list[Path]
    ├── inference.py         # predict_full_image, predict_full_image_tta
    ├── evaluate.py          # writes svg / md / csv / h5 / jpg artifacts
    ├── viz.py               # plot_confusion_matrix (svg, editable text), palette JPGs
    └── run.py               # run_ablation(...) — the one function the notebook calls
```

## How to run

```bash
pip install torch torchvision segmentation-models-pytorch \
            scikit-learn scikit-image h5py matplotlib pillow tqdm
cd ablation
jupyter notebook ablation.ipynb
```

The notebook's first cell does `%load_ext autoreload`, so edits to the
package are picked up without restarting the kernel.

The default `RunConfig.base_dir` autodetects macOS vs Windows; override it
on the `params=` dict if your training data lives elsewhere:

```python
run_ablation(
    model_name='unet_vanilla',
    params=dict(base_dir='/path/to/sickling/data', ...),
    notebook_path=NOTEBOOK_PATH,
)
```

Required directory layout under `base_dir`:

- `trainingImages/` — raw `.jpg`/`.png`/`.tif` images
- `h5_fixed_for_training/` — pseudo-label `.h5` masks (0..N-1 valid, 255 = ignore)
- `dense_segmentations/truth/` — hand-labelled `_segmentation.h5` files
  (raw 1-based Ilastik export; the first `truth_val_count` files sorted →
  per-epoch checkpoint selection, the rest → held-out test).

## Pipeline invariants (held constant across runs)

These match the `evaluationSummary.md` "held constant" list:

- 4-class semantics: `0=Polymer, 1=Background, 2=Cell, 3=Cell boundary`
- `IGNORE_INDEX=255`, `NORM_PERCENTILE=99`, `TILE_SIZE=256`, `N_FOLDS=5`,
  `KFold(random_state=42)`
- Class-0-aware crop sampling (`class0_crop_prob=0.5`)
- 8-way flip/rotation TTA at evaluation
- Composite loss:
  `WeightedDice + CE + CONFUSION_WEIGHT * directed_confusion
   + FN_WEIGHT * directed_FN + TVERSKY_WEIGHT * Tversky([class 0])`
- Truth-image basenames are excluded from the training pool entirely.
