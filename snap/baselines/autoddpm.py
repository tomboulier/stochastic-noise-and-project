# -*- coding: utf-8 -*-
"""AutoDDPM (Bercea et al., ICML 2023): mask, stitch, resample, on the pre-trained model.

    1  mask       x_rec = DDPM reconstruction of x at t_mask (anoddpm.ddpm_chain, Gaussian noise);
                  anomaly map = |x - x_rec| averaged over the channels, Gaussian blur (sigma 4 px);
                  mask = map > threshold, dilated by a (2 * 5 + 1)-pixel max filter.  The threshold is the
                  quantile q = 0.95 of all the map values of healthy (in-domain) VALIDATION images reconstructed
                  at t_mask (`autoddpm_threshold`; paper: 32 images).
    2  stitch     x_s = mask * x_rec + (1 - mask) * x
    3  resample   x_r = DDPM chain of x_s from t_resample; output mask * x_r + (1 - mask) * x.

Network calls per image: (t_mask + 1) + (t_resample + 1).  Paper grid (Fig. 6): t_mask in {200, 300, 400},
t_resample = 50; faces: t_mask in {100, 200}, t_resample = 50.

    python scripts/run_baseline.py --method autoddpm --model weights/chestmnist_224.pt --config configs/chestmnist_224.json \\
        --input data/chestmnist224/cardiomegaly_test.npy --n 64 --group Cardiomegaly \\
        --healthy data/chestmnist224/healthy_val.npy --n_healthy 32
"""

import math

import torch
import torch.nn.functional as F

from .anoddpm import ddpm_chain

__all__ = ["autoddpm", "autoddpm_threshold", "anomaly_map", "smooth"]


def smooth(c, sigma_px):
    """Separable Gaussian blur of maps (N, 1, H, W), radius ceil(3 sigma), reflected borders."""
    r = max(1, int(math.ceil(3 * sigma_px)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=c.device, dtype=torch.float32) / sigma_px) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    c = F.conv2d(F.pad(c, (r, r, 0, 0), mode="reflect"), k)
    return F.conv2d(F.pad(c, (0, 0, r, r), mode="reflect"), k.transpose(2, 3))


def anomaly_map(x, x_rec, sigma_px=4.0):
    """|x - x_rec| averaged over the channels, blurred: (N, 1, H, W)."""
    return smooth((x - x_rec).abs().mean(1, keepdim=True), sigma_px)


@torch.no_grad()
def autoddpm_threshold(model, healthy, t_mask, quantile=0.95, generator=None, batch=16, sigma_px=4.0,
                       max_values=5_000_000):
    """Quantile of all the anomaly-map values of healthy images (N, C, H, W) reconstructed at t_mask (in batches of
    `batch`, noise from `generator`).  Above `max_values` values, a fixed random subset (CPU generator, seed 0)."""
    device = model.device
    vals = []
    for s in range(0, len(healthy), batch):
        xb = healthy[s:s + batch].to(device)
        xr = ddpm_chain(model, xb, t_mask, generator)
        vals.append(anomaly_map(xb, xr, sigma_px).flatten().cpu())
    v = torch.cat(vals)
    if v.numel() > max_values:
        v = v[torch.randperm(v.numel(), generator=torch.Generator().manual_seed(0))[:max_values]]
    return float(torch.quantile(v, quantile))


@torch.no_grad()
def autoddpm(model, x, t_mask, t_resample, threshold, generator=None, sigma_px=4.0, dilation_px=5):
    """x (B, C, H, W) in [-1, 1] -> (y, nfe, mask (B, 1, H, W))."""
    x_rec = ddpm_chain(model, x, t_mask, generator)
    m = (anomaly_map(x, x_rec, sigma_px) > threshold).float()
    if dilation_px > 0:
        k = 2 * int(dilation_px) + 1
        m = F.max_pool2d(m, k, stride=1, padding=int(dilation_px))
    xs = m * x_rec + (1 - m) * x
    xr = ddpm_chain(model, xs, t_resample, generator)
    y = m * xr + (1 - m) * x
    return y, (int(t_mask) + 1) + (int(t_resample) + 1), m
