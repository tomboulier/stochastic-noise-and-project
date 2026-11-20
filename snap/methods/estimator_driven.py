# -*- coding: utf-8 -*-
"""SNAPED and P-SNAPED: estimator-driven noise levels and stopping (paper, Sec. 3.2, Algorithm 2).

A small classifier s (CNN, snap/estimators.py), fitted on N images of the source domain and on
images of the model's domain, gives a score that grows with the distance to the domain.  Two
calibration medians: s_h (held-out in-domain images; the stopping threshold) and s_s (the N
source images).  For every image and every cycle k:

    1  s_k <- s(x_k)
    2  if s_k < s_h: stop                                    (the image reads as in-domain)
    3  lambda_k <- clip((s_k - s_h) / (s_s - s_h), 0, 1)
    4  tau_k <- tau_min + lambda_k (tau_max - tau_min)       (rounded to a multiple of the stride)
    5  x_{k+1} <- one cycle at the level sigma_k of index tau_k:
                  SNAP cycle (SNAPED) or P-SNAP cycle, Algorithm 1 (P-SNAPED)

Paper settings: sigma_k in [0.34, 2.04] (tau 100 to 400), K_max = 500, s_h = median of the
held-out in-domain scores.  An image that stopped stays frozen.
"""

import time

import numpy as np
import torch

from .psnap import PSNAPCycle
from .snap import SNAPCycle, mse_to, to_uint8

__all__ = ["estimator_driven", "snaped", "psnaped"]


@torch.no_grad()
def estimator_driven(model, x, estimator, cycle="psnap", tau_min=100, tau_max=400, max_cycles=500,
                     s_healthy=None, s_source=None, stop_quantile=0.5, gamma=4.67, h_rel=0.01, stride=10,
                     batch=64, generator=None, save_cycles=(), monitors=None, log_every=0):
    """Algorithm 2.  cycle: "snap" (SNAPED) or "psnap" (P-SNAPED).

    estimator       an Estimator (snap/estimators.py): score(x) -> numpy (B,), calibrated
    s_healthy, s_source
                    calibration levels; default: from the estimator (quantile `stop_quantile` of its
                    held-out in-domain scores, median of its source scores)
    monitors        {name: f}, f(x) -> numpy (N, ...), evaluated at EVERY cycle (frozen images included)
    -> dict with x, x0, frames, mse (K+1, N), monitors {name: (K+1, N, ...)}, monitor_cycles,
       scores (K+1, N), taus (K, N) (0 once stopped), cycles_run (N,), nfe (N,) (cycles and estimator),
       stopped (N,), s_healthy, s_source, last_cycle.
    """
    if cycle not in ("snap", "psnap"):
        raise ValueError("cycle: 'snap' (SNAPED) or 'psnap' (P-SNAPED)")
    device = model.device
    s_healthy = float(estimator.healthy_threshold(stop_quantile) if s_healthy is None else s_healthy)
    s_source = float(estimator.source_level() if s_source is None else s_source)
    tau_min, tau_max = int(tau_min), int(tau_max)
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)
    x = x.to(device).float().clone()
    x0 = x.clone()
    N, K = len(x), int(max_cycles)
    scores = np.full((K + 1, N), np.nan, np.float32)
    mse = np.zeros((K + 1, N), np.float32)
    taus = np.zeros((K, N), np.int16)
    active = np.ones(N, bool)
    n_cyc = np.zeros(N, np.int32)
    nfe = np.zeros(N, np.float64)
    keep = sorted({int(k) for k in save_cycles if 0 <= int(k) <= K})
    monitors = dict(monitors or {})
    mon = {n: [] for n in monitors}
    frames, cycles = {}, {}
    last = K
    t0 = time.time()
    for k in range(K + 1):
        for n, f in monitors.items():
            mon[n].append(np.asarray(f(x)))
        scores[k] = estimator.score(x)                                    # line 1
        mse[k] = mse_to(x, x0)
        if k in keep:
            frames[k] = to_uint8(x)
        if k == K:
            break
        active &= scores[k] > s_healthy                                   # line 2
        nfe[active] += estimator.cost                                     # the estimator read for this decision
        if not active.any():
            last = k
            break
        lam = np.clip((scores[k] - s_healthy) / max(s_source - s_healthy, 1e-6), 0, 1)     # line 3
        t = (np.round((tau_min + lam * (tau_max - tau_min)) / stride) * stride).astype(int)  # line 4
        t[~active] = 0
        taus[k] = t
        xn = x.clone()
        for tv in sorted(set(t[active].tolist())):                        # line 5, grouped by level
            if tv not in cycles:
                cycles[tv] = (PSNAPCycle(model, tau=tv, gamma=gamma, stride=stride, h_rel=h_rel) if cycle == "psnap"
                              else SNAPCycle(model, tau=tv, stride=stride))
            idx = np.where(active & (t == tv))[0]
            it = torch.from_numpy(idx).to(device)
            for b in range(0, len(idx), batch):
                ib = it[b:b + batch]
                xn[ib] = cycles[tv](x[ib], generator)
            nfe[idx] += cycles[tv].calls
        n_cyc[active] += 1
        x = xn
        if log_every and (k + 1) % log_every == 0:
            sig = [model.sigma(v) for v in t[active]]
            print(f"[{'P-SNAPED' if cycle == 'psnap' else 'SNAPED'}] cycle {k + 1}: {int(active.sum())}/{N} running, "
                  f"median sigma {np.median(sig):.2f} ({time.time() - t0:.0f} s)")
    if last < K:                                                          # all stopped: the rest is frozen
        scores[last + 1:] = scores[last]
        mse[last + 1:] = mse[last]
        for n in mon:
            mon[n] += [mon[n][-1]] * (K - last)
        for kk in keep:
            if kk > last:
                frames[kk] = to_uint8(x)
    return {"x": x.cpu(), "x0": to_uint8(x0), "frames": frames, "mse": mse, "scores": scores, "taus": taus,
            "monitors": {n: np.stack(v) for n, v in mon.items()}, "monitor_cycles": np.arange(K + 1, dtype=np.int32),
            "cycles_run": n_cyc, "nfe": nfe, "stopped": scores[min(last, K)] <= s_healthy,
            "s_healthy": s_healthy, "s_source": s_source, "last_cycle": int(last)}


def snaped(model, x, estimator, **kw):
    """SNAPED: Algorithm 2 with SNAP cycles."""
    return estimator_driven(model, x, estimator, cycle="snap", **kw)


def psnaped(model, x, estimator, gamma=4.67, **kw):
    """P-SNAPED: Algorithm 2 with P-SNAP cycles (Algorithm 1)."""
    return estimator_driven(model, x, estimator, cycle="psnap", gamma=gamma, **kw)
