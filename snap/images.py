# -*- coding: utf-8 -*-
"""Reading and writing images.  Nothing is ever overwritten: every run writes into a new folder."""

import os
import time

import numpy as np
import torch

__all__ = ["load_images", "first", "to_uint8", "from_uint8", "save_png", "save_npy", "new_run_dir"]

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _to_nchw(a):
    a = np.asarray(a)
    if a.ndim == 3:                                      # (N, H, W)
        return a[:, None]
    if a.ndim == 4 and a.shape[-1] in (1, 3, 4) and a.shape[1] not in (1, 3, 4):
        return a.transpose(0, 3, 1, 2)                   # (N, H, W, C)
    if a.ndim == 4:
        return a                                         # (N, C, H, W)
    raise ValueError(f"cannot read an image array of shape {a.shape}")


def load_uint8(path, channels=None):
    """-> (uint8 array (N, C, H, W), names).  A folder of images (sorted by name), or a .npy / .npz
    array of uint8 images (a .npz: key "images", else its first array; names from key "names" if any)."""
    if os.path.isdir(path):
        from PIL import Image
        files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXT))
        if not files:
            raise FileNotFoundError(f"no image in {path}")
        mode = {1: "L", 3: "RGB"}.get(channels)
        arr = []
        for f in files:
            with Image.open(os.path.join(path, f)) as im:
                arr.append(np.asarray(im.convert(mode) if mode else im))
        return _to_nchw(np.stack(arr)), [os.path.splitext(f)[0] for f in files]
    names = None
    if path.endswith(".npz"):
        z = np.load(path)
        a = z["images"] if "images" in z.files else z[z.files[0]]
        if "names" in z.files:
            names = [str(n) for n in z["names"]]
    else:
        a = np.load(path)
    a = _to_nchw(a)
    return a, names or [f"{i:05d}" for i in range(len(a))]


def load_images(path, channels=None, size=None, resize=False):
    """-> (x float32 (N, C, H, W) in [-1, 1] on CPU, names).

    channels  1 or 3: convert (folder) or adapt (array: gray -> RGB by repetition, RGB -> luminance).
    size      expected side; a mismatch is an error unless resize=True (bicubic, antialiased).
    """
    a, names = load_uint8(path, channels)
    if a.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8 images, got {a.dtype}")
    if channels is not None and a.shape[1] != channels:
        if a.shape[1] == 1 and channels == 3:
            a = np.repeat(a, 3, 1)
        elif a.shape[1] in (3, 4) and channels == 1:                 # luminance (ITU-R 601)
            a = np.round(0.299 * a[:, 0] + 0.587 * a[:, 1] + 0.114 * a[:, 2])[:, None].astype(np.uint8)
        else:
            a = a[:, :channels]
    x = from_uint8(a)
    if size is not None and tuple(x.shape[-2:]) != (int(size), int(size)):
        if not resize:
            raise ValueError(f"{path}: images are {tuple(x.shape[-2:])}, the model expects {size} x {size} "
                             "(pass --resize to resize them)")
        x = torch.nn.functional.interpolate(x, size=(int(size), int(size)), mode="bicubic",
                                            antialias=True, align_corners=False).clamp(-1, 1)
    return x, names


def first(x, names, n=0, start=0):
    """Images start .. start + n - 1, in file order (n <= 0: all from start).  The paper's image sets are
    the first images of arrays written by scripts/export_chestmnist.py, already shuffled with seed 0."""
    stop = len(x) if n is None or int(n) <= 0 else int(start) + int(n)
    return x[int(start):stop], list(names[int(start):stop])


def from_uint8(a):
    return torch.from_numpy(np.asarray(a).astype(np.float32) / 127.5 - 1.0)


def to_uint8(x):
    """[-1, 1] tensor -> uint8 numpy array, same shape."""
    return ((x.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()


def save_png(u8, path):
    """One image (C, H, W) uint8 to a new PNG file."""
    from PIL import Image
    a = u8[0] if u8.shape[0] == 1 else u8.transpose(1, 2, 0)
    with open(path, "xb") as f:
        Image.fromarray(np.ascontiguousarray(a)).save(f, format="PNG")


def save_npy(a, path):
    with open(path, "xb") as f:
        np.save(f, a)


def new_run_dir(root, name):
    """root/<YYYYmmdd-HHMMSS>_<name>, created; _2, _3... if it already exists.  Never an existing folder."""
    os.makedirs(root, exist_ok=True)
    base = os.path.join(root, f"{time.strftime('%Y%m%d-%H%M%S')}_{name}")
    path, i = base, 1
    while True:
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            i += 1
            path = f"{base}_{i}"
