# -*- coding: utf-8 -*-
"""Pseudo-healthy reconstruction with a variational autoencoder (the non-diffusion baseline of Fig. 6).

A convolutional VAE is trained from scratch on in-domain images (ChestMNIST: the healthy training radiographs at
224 px; faces: FFHQ photographs at 256 px); the output for an image is the decoding of its posterior mean.

Architecture: the side is halved by stride-2 4 x 4 convolutions while it is even and > 7 (224 -> 7, five levels;
256 -> 4, six levels; 64 -> 4, four levels), channels 32, 64, 128, 256, 256, ..., each followed by batch norm and
LeakyReLU(0.2); dense mean and log-variance (clamped to [-12, 8]) of a latent of 256; a symmetric decoder of
transposed convolutions ending with tanh.  Loss: sum of squares over the pixels + beta KL, per image, averaged over
the batch; Adam (lr 1e-3), cosine schedule, batches drawn with replacement from a CPU generator.

Paper: beta in {0.01, 0.1, 1} (ChestMNIST, 20000 steps, batch 64); faces: beta in {0.1, 1} (8000 FFHQ photographs,
15000 steps, batch 64).  Training: scripts/train_vae.py; projection: scripts/run_baseline.py --method vae.

    python scripts/run_baseline.py --method vae --vae runs/vae/vae_b1.pt,runs/vae/vae_b0.1.pt,runs/vae/vae_b0.01.pt \\
        --input data/chestmnist224/cardiomegaly_test.npy --n 64 --group Cardiomegaly
"""

import json
import os
import time

import torch
import torch.nn as nn

__all__ = ["PseudoHealthyVAE", "train_vae", "reconstruct", "load_vae"]


class PseudoHealthyVAE(nn.Module):
    """The VAE; parameter names and creation order are those of the paper's checkpoints."""

    def __init__(self, latent=256, image_size=64, channels=1):
        super().__init__()
        n, s = 0, int(image_size)
        while s > 7 and s % 2 == 0:
            s //= 2
            n += 1
        c = (int(channels), 32, 64, 128, 256) + (256,) * max(0, n - 4)
        c = c[:n + 1]
        self.ch, self.side, self.latent = c[-1], s, int(latent)
        self.image_size, self.channels = int(image_size), int(channels)
        enc = []
        for i in range(n):
            enc += [nn.Conv2d(c[i], c[i + 1], 4, 2, 1), nn.BatchNorm2d(c[i + 1]), nn.LeakyReLU(0.2)]
        self.enc = nn.Sequential(*enc)
        self.mu = nn.Linear(self.ch * s * s, self.latent)
        self.lv = nn.Linear(self.ch * s * s, self.latent)
        self.fc = nn.Linear(self.latent, self.ch * s * s)
        dec = []
        for i in range(n, 1, -1):
            dec += [nn.ConvTranspose2d(c[i], c[i - 1], 4, 2, 1), nn.BatchNorm2d(c[i - 1]), nn.LeakyReLU(0.2)]
        dec += [nn.ConvTranspose2d(c[1], int(channels), 4, 2, 1), nn.Tanh()]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.mu(h), self.lv(h).clamp(-12, 8)

    def decode(self, z):
        return self.dec(self.fc(z).view(-1, self.ch, self.side, self.side))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.decode(z), mu, lv


def train_vae(x, beta, steps=20000, batch=64, lr=1e-3, latent=256, seed=0, device=None, log=None, log_every=500):
    """x (N, C, H, W) in [-1, 1] (CPU; moved to the device whole) -> the trained VAE in eval mode.
    log: list receiving {"beta", "step", "rec", "kl"} every `log_every` steps."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    net = PseudoHealthyVAE(latent, x.shape[-1], x.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    X = x.to(device)
    t0 = time.time()
    for it in range(1, steps + 1):
        net.train()
        idx = torch.randint(0, len(X), (batch,), generator=g).to(device)
        xb = X[idx]
        xh, mu, lv = net(xb)
        rec = (xh - xb).pow(2).flatten(1).sum(1).mean()
        kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
        loss = rec + beta * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == steps:
            if log is not None:
                log.append({"beta": beta, "step": it, "rec": rec.item(), "kl": kl.item()})
            print(f"[vae beta={beta:g}] step {it}/{steps} | rec {rec.item():.1f} | kl {kl.item():.1f} | "
                  f"{time.time() - t0:.0f} s")
    return net.eval()


@torch.no_grad()
def reconstruct(net, x, batch=16, device=None):
    """Decoded posterior means of x (N, C, H, W) in [-1, 1], in batches -> float32 CPU tensor (not clipped)."""
    device = device or next(net.parameters()).device
    out = []
    for s in range(0, len(x), batch):
        mu, _ = net.encode(x[s:s + batch].to(device))
        out.append(net.decode(mu).float().cpu())
    return torch.cat(out, 0)


def load_vae(path, image_size, channels, latent=None, device=None):
    """A VAE state dict (scripts/train_vae.py, or the paper's vae_b<beta>.pt).  The latent size is read from
    vae.json next to the file when present, else `latent` (default 256)."""
    meta = os.path.join(os.path.dirname(os.path.abspath(path)), "vae.json")
    if latent is None and os.path.exists(meta):
        with open(meta) as f:
            latent = json.load(f).get("latent")
    net = PseudoHealthyVAE(int(latent or 256), image_size, channels)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                                   # torch < 1.13
        state = torch.load(path, map_location="cpu")
    net.load_state_dict(state)
    return net.to(device or "cpu").eval()
