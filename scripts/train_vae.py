# -*- coding: utf-8 -*-
"""Train the pseudo-healthy VAE baseline (snap/baselines/vae.py) from scratch on in-domain images, one model per beta.

    # ChestMNIST 224 px (Fig. 6): all the healthy training radiographs, in the order of scripts/export_chestmnist.py
    # (exported WITHOUT --max_healthy_train)
    python scripts/train_vae.py --images data/chestmnist224/healthy_train.npy --betas 1,0.1,0.01 --steps 20000 --batch 64

    # FFHQ 256 px (faces): the first 8000 photographs of a folder of FFHQ 256 images, sorted by file name
    python scripts/train_vae.py --images data/ffhq256 --n 8000 --betas 1,0.1 --steps 15000 --batch 64

Loss: sum of squares over the pixels + beta KL, per image, averaged over the batch; Adam (lr 1e-3), cosine schedule;
batches drawn with replacement by a CPU generator seeded with --seed; the network is initialised after
torch.manual_seed(--seed).  The whole training set is moved to the device.  Writes a new folder
<out>/<date-time>_vae/: vae_b<beta>.pt (state dicts, read by scripts/run_baseline.py --method vae), vae.json
(settings and training log).  Nothing is overwritten.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from snap.images import IMAGE_EXT, load_uint8, from_uint8, new_run_dir  # noqa: E402
from snap.baselines.vae import train_vae  # noqa: E402


def read_images(path, n, size):
    """uint8 (N, C, H, W): an array (its first n images), or a folder (its first n image files sorted by name,
    converted to RGB and resized to size x size with Lanczos when needed)."""
    if not os.path.isdir(path):
        a, _ = load_uint8(path)
        return a[:n] if n and n > 0 else a
    from PIL import Image
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXT))
    files = files[:n] if n and n > 0 else files
    out = []
    for f in files:
        with Image.open(os.path.join(path, f)) as im:
            im = im.convert("RGB")
            if size and im.size != (size, size):
                im = im.resize((size, size), Image.LANCZOS)
            out.append(np.asarray(im, np.uint8).transpose(2, 0, 1))
    return np.stack(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="uint8 array (.npy / .npz) or folder of in-domain images")
    p.add_argument("--n", type=int, default=0, help="first n images (0: all; FFHQ: 8000)")
    p.add_argument("--size", type=int, default=256, help="folder images are resized to this side when needed")
    p.add_argument("--betas", default="1,0.1,0.01", help="KL weights, one model each")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                    # noqa: BLE001
        pass
    betas = [float(b) for b in a.betas.split(",") if b]
    x = from_uint8(read_images(a.images, a.n, a.size))
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[data] {tuple(x.shape)} from {a.images} | betas {betas}, {a.steps} steps, batch {a.batch} | {device}")
    out = new_run_dir(a.out, "vae")
    log, t0 = [], time.time()
    for beta in betas:
        net = train_vae(x, beta, a.steps, a.batch, a.lr, a.latent, a.seed, device, log)
        with open(os.path.join(out, f"vae_b{beta:g}.pt"), "xb") as f:
            torch.save(net.state_dict(), f)
    with open(os.path.join(out, "vae.json"), "x") as f:
        json.dump({"images": a.images, "n_images": int(len(x)), "image_size": int(x.shape[-1]),
                   "channels": int(x.shape[1]), "latent": a.latent, "betas": betas, "steps": a.steps,
                   "batch": a.batch, "lr": a.lr, "seed": a.seed, "device": str(device), "torch": torch.__version__,
                   "seconds": round(time.time() - t0, 1), "log": log}, f, indent=1)
    print(f"[done] -> {out}")
    return out


if __name__ == "__main__":
    main()
