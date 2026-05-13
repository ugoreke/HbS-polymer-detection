"""Model factory + optimizer builder.

All four model variants flow through the same training loop in ``train.py``;
the only per-model knobs are (a) which architecture is instantiated here and
(b) how the optimizer is built (single LR vs discriminative LR with optional
encoder freeze)."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from ablation.config import RunConfig


# ----- Vanilla U-Net (lifted from training 2.ipynb) -----
class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class VanillaUNet(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 4) -> None:
        super().__init__()
        self.inc = _DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up1 = _DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up2 = _DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up3 = _DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up4 = _DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.conv_up1(torch.cat([x4, self.up1(self.down4(x4))], dim=1))
        x = self.conv_up2(torch.cat([x3, self.up2(x)], dim=1))
        x = self.conv_up3(torch.cat([x2, self.up3(x)], dim=1))
        x = self.conv_up4(torch.cat([x1, self.up4(x)], dim=1))
        return self.outc(x)


_SMP_ENCODERS = {
    "smp_unet_resnet34":   ("resnet34",        None),
    "smp_unet_effb0_scse": ("efficientnet-b0", "scse"),
    "smp_unet_effb7_scse": ("efficientnet-b7", "scse"),
}


def model_factory(name: str, n_classes: int, in_channels: int = 1) -> nn.Module:
    if name == "unet_vanilla":
        return VanillaUNet(n_channels=in_channels, n_classes=n_classes)
    if name in _SMP_ENCODERS:
        import segmentation_models_pytorch as smp  # local import keeps SMP optional
        encoder, attn = _SMP_ENCODERS[name]
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=n_classes,
            decoder_attention_type=attn,
        )
    raise ValueError(f"Unknown model_name={name!r}. Valid keys: "
                     f"{['unet_vanilla', *_SMP_ENCODERS.keys()]}")


def is_smp(name: str) -> bool:
    return name in _SMP_ENCODERS


def _encoder_params(model: nn.Module) -> tuple[Iterable[torch.nn.Parameter], Iterable[torch.nn.Parameter]]:
    enc = list(model.encoder.parameters())  # type: ignore[attr-defined]
    other = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return enc, other


def build_optimizer(model: nn.Module, cfg: RunConfig) -> optim.Optimizer:
    """Vanilla U-Net: single LR. SMP models: discriminative LR (encoder at
    ``lr * encoder_lr_mult``)."""
    if not is_smp(cfg.model_name):
        return optim.Adam(model.parameters(), lr=cfg.lr)
    enc, other = _encoder_params(model)
    return optim.Adam([
        {"params": enc,   "lr": cfg.lr * cfg.encoder_lr_mult},
        {"params": other, "lr": cfg.lr},
    ])


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze the SMP encoder for warmup. No-op for vanilla U-Net."""
    if not hasattr(model, "encoder"):
        return
    for p in model.encoder.parameters():  # type: ignore[attr-defined]
        p.requires_grad = trainable
