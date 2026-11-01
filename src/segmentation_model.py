"""Pretrained 2-D vessel segmentation: model architecture and inference.

The U-Net layers and state-dict keys match the released segmentation weights.
Inference uses per-image min/max normalization, bilinear resizing with
``align_corners=False``, and sigmoid probabilities. No segmentation training or
dataset generation is included in this module.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_image_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv2D(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up2D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2)
        self.conv = DoubleConv2D(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]

        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2DSoftSeg(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, out_ch=1):
        super().__init__()
        self.inc = DoubleConv2D(in_ch, base_ch)
        self.down1 = Down2D(base_ch, base_ch * 2)
        self.down2 = Down2D(base_ch * 2, base_ch * 4)
        self.down3 = Down2D(base_ch * 4, base_ch * 8)
        self.up1 = Up2D(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up2D(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3 = Up2D(base_ch * 2, base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def load_soft_seg_model(ckpt_path: str, device: str = "cuda", base_ch: int = 16):
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)

    model = UNet2DSoftSeg(in_ch=1, base_ch=int(base_ch), out_ch=1).to(dev)

    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    # strip common prefixes if present
    clean_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        clean_state[nk] = v

    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model, dev


@torch.no_grad()
def predict_soft_masks_from_xrays(
    model,
    model_device,
    xrays_hw,
    seg_input_hw=(256, 256),
):
    xs = np.stack(
        [normalize_image_01(x).astype(np.float32) for x in xrays_hw],
        axis=0,
    )

    xt = torch.from_numpy(xs[:, None]).to(model_device, dtype=torch.float32)

    if tuple(xt.shape[-2:]) != tuple(seg_input_hw):
        xt = F.interpolate(
            xt,
            size=tuple(seg_input_hw),
            mode="bilinear",
            align_corners=False,
        )

    logits = model(xt)
    probs = torch.sigmoid(logits)[:, 0]

    return probs.detach().cpu().numpy().astype(np.float32)
