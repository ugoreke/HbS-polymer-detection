# Polymer + cell segmentation — demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugoreke/HbS-polymer-detection/blob/main/demo/demo.ipynb)

Loads a single microscopy image (`sample.jpg`), runs the frozen U-Net
checkpoint, segments cell instances via watershed, and renders one figure:
the raw image on top, the overlay (orange polymer tint + per-cell instance
boundaries) on the bottom.

## What's in this folder

```
demo/
├── demo.ipynb        # 6-cell notebook (setup, imports, predict, overlay, plot)
├── sample.jpg        # the input image rendered by the notebook
├── sickling/
│   ├── config.py             # dataclasses
│   ├── io/images.py          # load_raw_greyscale, normalize_image, find_raw_image
│   ├── stage1_unet/          # UNet + load_unet + predict_label_map
│   └── stage2_instances/     # mask_to_instances_with_reasons (marker-seeded watershed)
└── README.md
```

The bundled `sickling/` subset means the notebook runs without needing the
full project on the Python path — `sys.path.insert(0, '.')` in cell 1 is
enough.

## Model checkpoint

The U-Net checkpoint is 124 MB and lives outside the repo:

  https://drive.google.com/file/d/123OgOWBpMXkRRBDnfOsmkR1_-_MVgpF6/view?usp=sharing

The first cell in `demo.ipynb`:

- **On Colab.** Detects the Colab runtime, `pip install`s `gdown` + a couple
  of small deps, clones the bundled `sickling/` and `sample.jpg` into the
  Colab working directory, then downloads the model into
  `unet_fold_2_best.pth`. Drive's "anyone with the link can view" setting
  is what `gdown` needs — no extra changes required.
- **Locally.** Defaults `MODEL_PATH` to the parent project location
  (`/Users/utkugoreke/anaconda_projects/sickling/rbc-class/models/unet_fold_2_best.pth`).
  If that file doesn't exist, it falls back to `./unet_fold_2_best.pth` so
  you can drop the checkpoint next to the notebook and run.

Edit `MODEL_PATH` directly if you keep the checkpoint somewhere else.

## How to run

### Colab (one click)

Click the badge at the top. The setup cell handles installs + downloads.

### Local

```bash
pip install torch torchvision scikit-image h5py matplotlib pillow scipy pyyaml pydantic pydantic-settings
cd demo
jupyter notebook demo.ipynb
```

Either point the bundled `MODEL_PATH` at your local checkpoint, or download
the Drive file and drop it as `demo/unet_fold_2_best.pth`.

## What the overlay encodes

- **Greyscale background** — the percentile-normalized raw image
  (`normalize_image` with the same 99th-percentile clip used at training).
- **Orange tint, alpha 0.5** — pixels predicted as `class 0 = Polymer` by
  the U-Net.
- **Pinkish-red 2-px outlines** — boundaries of cell instances surviving
  the Stage 2 watershed filter (`mask_to_instances_with_reasons` applies
  area + edge-touching filters from `InstancesConfig`).

The notebook prints `cell instances kept / proposed` for QA.

## Class palette (for reference)

`0 = Polymer (thin, faint, rare)`, `1 = Background`, `2 = Cell body`,
`3 = Cell boundary`. Only class 0 is overlaid in the demo; the cell vs.
boundary distinction is collapsed into the instance outline.
