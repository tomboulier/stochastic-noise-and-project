# -*- coding: utf-8 -*-
"""Multi-octave 2-D simplex noise in torch, for AnoDDPM (Wyatt et al., CVPR Workshops 2022).

AnoDDPM replaces the Gaussian noise of the DDPM with simplex noise (gradient noise with a spatial structure, several
octaves): the model is trained to predict this noise, and the partial diffusion and every step of the DDPM chain
draw it.  The original code draws it with OpenSimplex (numpy, 3-D with a random offset in the third coordinate);
here, Gustavson's 2-D simplex, vectorised, with a random offset per image, channel and octave (the role of their
random third coordinate); each field is then standardised (mean 0, variance 1 per image and channel) to keep the
statistics of the DDPM's epsilon.  AnoDDPM's defaults: 6 octaves, persistence 0.8, starting frequency 1/64 at
256 px, here relative to the size: the first octave has a wavelength of H / 4 pixels.

Random draws, in this order, from `generator` (a CUDA generator only seeds a fresh CPU generator with one randint):
a permutation of 256 gradients, then one (N * C, 2) uniform offset per octave.
"""

import math

import torch

__all__ = ["simplex_noise", "simplex2d"]

_F2 = 0.5 * (math.sqrt(3.0) - 1.0)
_G2 = (3.0 - math.sqrt(3.0)) / 6.0
_GRAD = torch.tensor([[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float32)


def _hash(i, j, perm):
    """Gradient index (0..7) of the integer vertex (i, j), through a permutation table of 256."""
    return perm[(perm[i & 255] + j) & 255] & 7


def simplex2d(x, y, perm):
    """x, y tensors of the same shape -> 2-D simplex noise, roughly in [-1, 1]."""
    s = (x + y) * _F2
    i = torch.floor(x + s)
    j = torch.floor(y + s)
    t = (i + j) * _G2
    x0 = x - (i - t)
    y0 = y - (j - t)
    i1 = (x0 > y0).to(x.dtype)
    j1 = 1.0 - i1
    x1, y1 = x0 - i1 + _G2, y0 - j1 + _G2
    x2, y2 = x0 - 1.0 + 2.0 * _G2, y0 - 1.0 + 2.0 * _G2
    ii, jj = i.long(), j.long()
    g = _GRAD.to(x.device)
    out = torch.zeros_like(x)
    for (dx, dy, ci, cj) in ((x0, y0, ii, jj), (x1, y1, ii + i1.long(), jj + j1.long()), (x2, y2, ii + 1, jj + 1)):
        gr = g[_hash(ci, cj, perm)]
        tt = (0.5 - dx * dx - dy * dy).clamp_min(0.0)
        out = out + tt ** 4 * (gr[..., 0] * dx + gr[..., 1] * dy)
    return 70.0 * out


def simplex_noise(shape, device, generator=None, octaves=6, persistence=0.8, wavelength=None, standardize=True):
    """shape (N, C, H, W) -> multi-octave simplex noise, one independent field per image and channel, float32.

    wavelength   of the first octave, in pixels (default H / 4: 64 at 256 px, as in AnoDDPM)
    standardize  mean 0 and standard deviation 1 per image and channel
    """
    n, c, h, w = shape
    wavelength = float(wavelength or h / 4.0)
    if generator is not None and generator.device.type != "cpu":    # CPU draws, seeded from the given generator
        generator = torch.Generator().manual_seed(
            int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator, device=generator.device)))
    perm = torch.randperm(256, generator=generator, device="cpu").to(device)
    perm = torch.cat([perm, perm])
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    total = torch.zeros((n * c, h, w), device=device)
    amp = 1.0
    for o in range(int(octaves)):
        f = (2.0 ** o) / wavelength
        off = torch.rand((n * c, 2), generator=generator, device="cpu").to(device) * 1000.0     # offset per field
        X = xx[None] * f + off[:, 0, None, None]
        Y = yy[None] * f + off[:, 1, None, None]
        total = total + amp * simplex2d(X, Y, perm)
        amp *= float(persistence)
    total = total.view(n, c, h, w)
    if standardize:
        m = total.mean(dim=(2, 3), keepdim=True)
        s = total.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        total = (total - m) / s
    return total
