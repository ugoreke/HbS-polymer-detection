# HbS-polymer-detection

Polymer + cell segmentation for sickle-cell microscopy. Two deliverables:

- **[`ablation/`](ablation/)** — ablation harness (`ablation.ipynb` + supporting
  `ablation/` package). One `run_ablation(...)` call per cell trains all
  k-folds, runs 8-way TTA on the held-out truth-test images, and writes a
  versioned `evaluation_truth_{model}_v{N}/` folder with svg confusion matrix,
  per-class + per-image dice, and visualised predictions. Pre-filled (but
  commented) cells reproduce Runs A–F from the project's ablation log.

- **[`demo/`](demo/)** — minimal end-to-end demo:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugoreke/HbS-polymer-detection/blob/main/demo/demo.ipynb)
  Loads `sample.jpg`, runs the frozen U-Net, segments cell instances, and
  renders one figure (raw on top, overlay on bottom). The 124 MB checkpoint
  is auto-downloaded from Google Drive on Colab.

## Quick start

Pick the deliverable you want and follow its README:

- Reproducing the ablation table → [`ablation/README.md`](ablation/README.md)
- Just see it run on one image → [`demo/README.md`](demo/README.md)

## Model checkpoint

The U-Net weights live on Drive (too large for GitHub):
https://drive.google.com/file/d/123OgOWBpMXkRRBDnfOsmkR1_-_MVgpF6/view?usp=sharing

Sharing is set to "anyone with the link can view", which is what `gdown` (the
download path the Colab demo uses) needs.

## Classes

`0 = Polymer (thin, faint, rare)`, `1 = Background`, `2 = Cell body`,
`3 = Cell boundary`.
