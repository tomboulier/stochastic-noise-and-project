# -*- coding: utf-8 -*-
"""SNAP: stochastic noise-and-project cycles (paper, Sec. 3).

One SNAP cycle at level sigma (timestep tau):

    xi ~ N(0, I);   x_tau = sqrt(alpha_bar_tau) x + sqrt(1 - alpha_bar_tau) xi
    x  <- C_sigma(x, xi) = DDIM integration (eta = 0) from tau to 0, on the timesteps
          tau, tau - s, ..., s, 0 (s = stride = 10), x0 clipped to [-1, 1] at every step.

SNAP iterates this cycle with a fresh noise at each iteration: `snap(model, x, sigma, n_cycles)`.
`run_cycles` is the loop shared with P-SNAP (methods/psnap.py).
"""

import time

import numpy as np
import torch

__all__ = ["SNAPCycle", "snap", "run_cycles", "to_uint8"]


def to_uint8(x):
    """[-1, 1] tensor -> uint8 numpy array of the same shape."""
    return ((x.detach().float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()


def mse_to(x, x0):
    """Per-image MSE between x and x0, images in [0, 1] (mean over pixels and channels)."""
    return ((x - x0) / 2).pow(2).flatten(1).mean(1).cpu().numpy()


class SNAPCycle:
    """One SNAP cycle at level `sigma` (or timestep `tau`).

    nfe      network calls per cycle (tau / stride + 1)
    tau      the first DDIM timestep (tau rounded down to a multiple of the stride)
    sigma    its noise level
    """

    name = "snap"

    def __init__(self, model, tau=None, sigma=None, stride=10, clip=True):
        self.model, self.stride, self.clip = model, int(stride), bool(clip)
        tau = model.resolve_tau(tau, sigma, self.stride)
        if not 0 < tau < model.T:
            raise ValueError(f"tau = {tau} outside (0, {model.T})")
        self.tau = tau - tau % self.stride
        if self.tau == 0:
            raise ValueError(f"tau = {tau} is below the DDIM stride {self.stride}")
        self.timesteps = list(range(self.tau, -1, -self.stride))
        self.sigma = model.sigma(self.tau)
        self.nfe = len(self.timesteps)
        self.nfe_fp32 = 0

    def __repr__(self):
        return f"{type(self).__name__}(sigma={self.sigma:.3f}, tau={self.tau}, nfe={self.nfe})"

    @property
    def calls(self):
        """Network calls per cycle, all precisions."""
        return self.nfe + self.nfe_fp32

    def add_noise(self, x, xi):
        sa, sb = self.model.coef(self.tau)
        return sa * x + sb * xi

    def ddim_step(self, x, t, t_prev):
        """One deterministic DDIM step from t to t_prev (t_prev < 0: to x0)."""
        m = self.model
        eps = m.eps(x, t)
        sa, sb = m.coef(t)
        x0 = (x - sb * eps) / sa
        if self.clip:
            x0 = x0.clamp(-1.0, 1.0)
        if t_prev < 0:
            return x0
        if m.eps_from_clipped_x0:                     # guided-diffusion's DDIM
            eps = (x - sa * x0) / sb
        sa_p, sb_p = m.coef(t_prev)
        return sa_p * x0 + sb_p * eps

    def denoise(self, x_tau):
        """C_sigma: the DDIM integration from tau to 0."""
        ts = self.timesteps
        x = x_tau
        for i, t in enumerate(ts):
            x = self.ddim_step(x, t, ts[i + 1] if i + 1 < len(ts) else -1)
        return x

    @torch.no_grad()
    def __call__(self, x, generator=None, noise=None, info=None):
        """x (B, C, H, W) in [-1, 1] -> the image after one cycle.  `noise` imposes xi."""
        xi = torch.randn(x.shape, device=x.device, generator=generator) if noise is None else noise
        return self.denoise(self.add_noise(x, xi))


@torch.no_grad()
def run_cycles(cycle, x, n_cycles, batch=64, generator=None, save_cycles=(), monitors=None,
               monitor_cycles=None, log_every=0, log_name=None):
    """n_cycles cycles of `cycle` on every image of x (N, C, H, W) in [-1, 1].

    One generator for the whole run; at each cycle the images are processed in batches of
    `batch`, and each batch draws its own noise: the result depends on (seed, batch, device).
    monitors        {name: f}, f(x on the device) -> numpy (N, ...): evaluated at `monitor_cycles`
                    (default: every cycle), e.g. a classifier's probabilities.
    -> dict with x (final, CPU), x0 (uint8), frames {k: uint8}, mse (K+1, N) to x0 in [0, 1],
       monitors {name: (M, N, ...)}, monitor_cycles, nfe (N,), cycles_run (N,).
    """
    device = cycle.model.device
    x = x.to(device).float().clone()
    x0 = x.clone()
    N, K = len(x), int(n_cycles)
    keep = {int(k) for k in save_cycles if 0 <= int(k) <= K}
    monitors = dict(monitors or {})
    mon_k = sorted({int(k) for k in (range(K + 1) if monitor_cycles is None else monitor_cycles) if 0 <= int(k) <= K})
    mon = {n: [] for n in monitors}
    frames, mse = {}, np.zeros((K + 1, N), np.float32)
    t0 = time.time()
    for k in range(K + 1):
        mse[k] = mse_to(x, x0)
        if k in keep:
            frames[k] = to_uint8(x)
        if k in mon_k:
            for n, f in monitors.items():
                mon[n].append(np.asarray(f(x)))
        if k == K:
            break
        xn = x.clone()
        for b in range(0, N, batch):
            xn[b:b + batch] = cycle(x[b:b + batch], generator)
        x = xn
        if log_every and (k + 1) % log_every == 0:
            print(f"[{log_name or cycle.name}] cycle {k + 1}/{K} | mean MSE {mse_to(x, x0).mean():.2e} "
                  f"({time.time() - t0:.0f} s)")
    return {"x": x.cpu(), "x0": to_uint8(x0), "frames": frames, "mse": mse,
            "monitors": {n: np.stack(v) for n, v in mon.items()}, "monitor_cycles": np.array(mon_k, np.int32),
            "nfe": np.full(N, K * cycle.calls, np.float64), "cycles_run": np.full(N, K, np.int32)}


def snap(model, x, sigma=None, n_cycles=50, tau=None, stride=10, batch=64, generator=None, **kw):
    """SNAP: n_cycles SNAP cycles at a fixed level.  Returns the dict of `run_cycles`."""
    return run_cycles(SNAPCycle(model, tau, sigma, stride), x, n_cycles, batch, generator, **kw)
