# -*- coding: utf-8 -*-
"""The degradations of the face-restoration experiment (paper, Sec. 5 and Table 1).

    blur        PIL GaussianBlur, radius in pixels at 256 px (scaled with the width)
    jpeg        JPEG compression (PIL), quality q
    pixelation  down-sampling by block averaging (PIL BOX) by a factor f, then nearest-neighbour up-sampling
    noise       additive Gaussian noise of standard deviation s on the image in [0, 1], then clipped to [0, 1];
                numpy generator of seed 1000 + i, i = index of the photograph in its set (test then validation)

    moderate    blur radius 3, JPEG quality 8, pixelation x8, noise 0.1
    strong      blur radius 6, JPEG quality 4, pixelation x12, noise 0.2

Every degradation is deterministic given the image and its index.
"""

import io

import numpy as np

__all__ = ["DEGRADATIONS", "LEVELS", "degrade", "describe", "COLUMN_NAMES"]

DEGRADATIONS = ("blur", "jpeg", "pixelation", "noise")
LEVELS = {"moderate": {"blur": 3.0, "jpeg": 8, "pixelation": 8, "noise": 0.1},
          "strong": {"blur": 6.0, "jpeg": 4, "pixelation": 12, "noise": 0.2}}
COLUMN_NAMES = {"blur": "blur", "jpeg": "JPEG", "pixelation": "pixelation", "noise": "noise"}
PARAMETER = {"blur": "radius_px", "jpeg": "quality", "pixelation": "factor", "noise": "std"}


def degrade(u8, kind, value, index=0):
    """One image (C, H, W) uint8, C = 3 (or 1) -> the degraded image (C, H, W) uint8.

    kind   blur | jpeg | pixelation | noise | clean;  value  its parameter;  index  seeds the noise (1000 + index)
    """
    from PIL import Image, ImageFilter
    u8 = np.asarray(u8, np.uint8)
    rgb = u8.shape[0] == 3
    im = Image.fromarray(np.ascontiguousarray(u8.transpose(1, 2, 0) if rgb else u8[0]))
    W, H = im.size
    if kind == "clean":
        out = im
    elif kind == "blur":
        out = im.filter(ImageFilter.GaussianBlur(radius=float(value) * W / 256))
    elif kind == "jpeg":
        b = io.BytesIO()
        im.save(b, format="JPEG", quality=int(value))
        out = Image.open(io.BytesIO(b.getvalue())).convert("RGB" if rgb else "L")
    elif kind == "pixelation":
        f = max(1, int(value))
        out = im.resize((max(1, W // f), max(1, H // f)), Image.BOX).resize((W, H), Image.NEAREST)
    elif kind == "noise":
        rng = np.random.default_rng(1000 + int(index))
        shape = (H, W, 3) if rgb else (H, W)
        x = np.asarray(im, np.float32) / 255 + rng.normal(0, float(value), shape).astype(np.float32)
        out = Image.fromarray((np.clip(x, 0, 1) * 255).round().astype(np.uint8))
    else:
        raise ValueError(f"unknown degradation {kind!r} (one of {', '.join(DEGRADATIONS)}, clean)")
    a = np.asarray(out, np.uint8)
    return (a.transpose(2, 0, 1) if rgb else a[None]).copy()


def describe(kind, value, long=False):
    """'r = 3 px', 'q = 8', 'x8', 's = 0.1' (long: 'blur of radius 3 pixels', 'JPEG quality 8', ...)."""
    v = format(float(value), "g")
    if long:
        return {"blur": f"blur of radius {v} pixels", "jpeg": f"JPEG quality {v}", "pixelation": f"pixelation $\\times${v}",
                "noise": f"noise {v}"}[kind]
    return {"blur": f"r = {v} px", "jpeg": f"q = {v}", "pixelation": f"x{v}", "noise": f"s = {v}"}[kind]
