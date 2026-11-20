# -*- coding: utf-8 -*-
"""DIIP, Diffusion Image Prior (Chihaoui & Favaro, ICCV 2025), reimplemented for a frozen pre-trained model (no
official code was available).  Choices, from the sections of their paper:

  - optimised variable: the initial noise z ~ N(0, I) (Algo. 2, line 1; Eq. 5); the model is frozen (Sec. 3.3);
  - generator g(z): deterministic DDIM from t = 999 to 0 by steps of 100 (Appendix A), 10 network calls, x0 clipped
    to [-1, 1] and the step taken with the model's DDIM convention (as in a SNAP cycle); the gradient flows through
    the whole chain (Algo. 2, line 9), recomputed step by step (non-reentrant checkpointing, autocast kept);
  - loss ||g(z) - y||^2 (Eq. 5), mean over the pixels of each image; Adam, learning rate 0.0015 (Sec. 4);
  - early stopping (Algo. 1, Sec. 3.4), after k_min = 100 iterations:
      "loss"       (high-frequency degradations: JPEG, noise) stop as soon as the relative decrease of the loss
                   (E_{k-1} - E_k) / E_{k-1} falls below eps = 0.001, return g(z_k);
      "sharpness"  (low-frequency degradations: blur, pixelation) variance of the Laplacian of the luminance of
                   g(z_k); stop when it decreases, return the last peak g(z_{k-1});
    an image that has not stopped after k_max = 300 iterations returns g(z_{k_max}) ("capped").

Images are processed in a stream of `batch` slots: when one stops, the next one takes its slot.  Adam is written
out with a step counter PER IMAGE (the formulas of torch.optim.Adam, betas 0.9 / 0.999, eps 1e-8), so the result
of an image depends only on its initial noise and its input.  Cost: per iteration, 10 forward calls and the
backward pass through them (with their recomputation).

guided-diffusion's attention block calls its own `checkpoint(..., True)`, a reentrant recomputation that loses the
autocast state in the backward pass: it is replaced by a direct call in the network's module (same forward result),
and every `use_checkpoint` flag of the network is turned off.  This modifies the loaded network in place.

    python scripts/run_baseline.py --method diip --model weights/ffhq.pt --config configs/ffhq_256_adm.json \\
        --recipe faces --input <level>/jpeg.npz --degradation jpeg --row_offset 132 --n 100
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

__all__ = ["DIIP", "laplacian_variance", "disable_adm_checkpointing"]


def laplacian_variance(x):
    """x (B, C, H, W) in [-1, 1] -> variance of the Laplacian of the luminance (ITU-R 601; the image itself when
    C = 1), in [0, 1] units, (B,)."""
    lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None] if x.shape[1] == 3 else x[:, :1]
    g = lum * 0.5 + 0.5
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device, dtype=g.dtype)[None, None]
    return F.conv2d(g, k).flatten(1).var(1)


def disable_adm_checkpointing(net):
    """Turn off guided-diffusion's internal (reentrant) gradient checkpointing; returns the number of changes."""
    n = 0
    for m in net.modules():
        if getattr(m, "use_checkpoint", False):
            m.use_checkpoint = False
            n += 1
    mod = sys.modules.get(type(net).__module__)
    if mod is not None and hasattr(mod, "checkpoint") and "guided_diffusion" in mod.__name__:
        mod.checkpoint = lambda func, inputs, params, flag: func(*inputs)
        n += 1
    return n


class DIIP:
    """DIIP on a DiffusionModel.  restore(y, criterion, z0) -> (images, stopping iterations, capped)."""

    def __init__(self, model, dt=100, t_max=999, lr=1.5e-3, eps=1e-3, k_min=100, k_max=300, recompute=True):
        self.model = model
        self.timesteps = list(range(int(t_max), -1, -int(dt)))        # 999, 899, ..., 99: 10 calls
        self.lr, self.eps, self.k_min, self.k_max = float(lr), float(eps), int(k_min), int(k_max)
        self.recompute = bool(recompute)
        self.n_calls = len(self.timesteps)
        for p in model.net.parameters():
            p.requires_grad_(False)
        self.n_disabled = disable_adm_checkpointing(model.net)

    def _step(self, x, i):
        m = self.model
        t = self.timesteps[i]
        sa, sb = m.coef(t)
        eps = m.eps(x, t)
        x0 = ((x - sb * eps) / sa).clamp(-1.0, 1.0)
        if i + 1 == len(self.timesteps):
            return x0
        if m.eps_from_clipped_x0:
            eps = (x - sa * x0) / sb
        sa_p, sb_p = m.coef(self.timesteps[i + 1])
        return sa_p * x0 + sb_p * eps

    def generate(self, z):
        """g(z): the deterministic DDIM chain, differentiable."""
        x = z
        for i in range(len(self.timesteps)):
            if self.recompute and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(self._step, x, i, use_reentrant=False)
            else:
                x = self._step(x, i)
        return x

    def restore(self, y, criterion, z0, batch=4, log=None):
        """y (N, C, H, W) in [-1, 1] (CPU); criterion "loss" or "sharpness"; z0 (N, C, H, W) initial noises.
        -> (images float32 CPU (N, C, H, W) in [-1, 1], stopping iteration (N,), capped (N,) bool)."""
        if criterion not in ("loss", "sharpness"):
            raise ValueError("criterion: 'loss' or 'sharpness'")
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        dev = self.model.device
        N = len(y)
        out = torch.zeros_like(y)
        k_stop = np.full(N, -1, dtype=int)
        capped = np.zeros(N, bool)
        queue = list(range(N))
        slots = []                                                    # image indices of the active rows
        z = m = v = yy = None
        t = torch.zeros(0, device=dev)
        e_prev = torch.zeros(0, device=dev)
        lv_prev = torch.zeros(0, device=dev)
        x_prev = None
        rnd = 0

        def add(i):
            nonlocal z, m, v, yy, t, e_prev, lv_prev, x_prev
            zi = z0[i:i + 1].to(dev).float()
            yi = y[i:i + 1].to(dev).float()
            zero = torch.zeros_like(zi)
            z = zi if z is None else torch.cat([z, zi])
            m = zero if m is None else torch.cat([m, zero])
            v = zero.clone() if v is None else torch.cat([v, zero])
            yy = yi if yy is None else torch.cat([yy, yi])
            t = torch.cat([t, torch.zeros(1, device=dev)])
            e_prev = torch.cat([e_prev, torch.full((1,), float("nan"), device=dev)])
            lv_prev = torch.cat([lv_prev, torch.full((1,), float("nan"), device=dev)])
            x_prev = torch.zeros_like(yi.cpu()) if x_prev is None else torch.cat([x_prev, torch.zeros_like(yi.cpu())])
            slots.append(i)

        def keep_rows(keep):
            nonlocal z, m, v, yy, t, e_prev, lv_prev, x_prev, slots
            kk = torch.as_tensor(np.where(keep)[0], device=dev, dtype=torch.long)
            z, m, v, yy, t, e_prev, lv_prev = (q[kk] for q in (z, m, v, yy, t, e_prev, lv_prev))
            x_prev = x_prev[np.where(keep)[0]]
            slots = [s for s, k_ in zip(slots, keep) if k_]

        while queue or slots:
            while queue and len(slots) < batch:
                add(queue.pop(0))
            rnd += 1
            zl = z.detach().requires_grad_(True)
            x = self.generate(zl)
            e = (x - yy).pow(2).flatten(1).mean(1)
            e.sum().backward()
            with torch.no_grad():                                     # the operations of torch.optim.Adam
                g = zl.grad
                t = t + 1
                m = torch.lerp(m, g, 1 - b1)
                v = (v * b2).addcmul_(g, g, value=1 - b2)
                tt = t.double()                                       # bias corrections in double, as torch does
                step = (self.lr / (1 - b1 ** tt)).float().view(-1, 1, 1, 1)
                bc2s = ((1 - b2 ** tt) ** 0.5).float().view(-1, 1, 1, 1)
                z = z + (-step * m) / (v.sqrt() / bc2s + eps_a)
            xd, ed = x.detach(), e.detach()
            kcur = t.long().cpu().numpy()                            # iteration reached by each row
            stop = np.zeros(len(slots), bool)
            image_stop = [None] * len(slots)
            if criterion == "loss":
                drop = ((e_prev - ed) / e_prev.clamp_min(1e-12)).cpu().numpy()
                ok = (kcur >= self.k_min) & np.isfinite(drop) & (drop < self.eps)
                for r in np.where(ok)[0]:
                    stop[r], image_stop[r] = True, (xd[r].cpu(), int(kcur[r]))
                e_prev = ed
            else:
                lv = laplacian_variance(xd)
                ok = (kcur > self.k_min) & (lv < lv_prev).cpu().numpy()
                for r in np.where(ok)[0]:
                    stop[r], image_stop[r] = True, (x_prev[r].clone(), int(kcur[r]) - 1)
                lv_prev, x_prev = lv, xd.cpu()
            for r in range(len(slots)):
                if not stop[r] and kcur[r] >= self.k_max:
                    stop[r], image_stop[r] = True, (xd[r].cpu(), int(kcur[r]))
                    capped[slots[r]] = True
                if stop[r]:
                    out[slots[r]], k_stop[slots[r]] = image_stop[r]
            if stop.any():
                keep_rows(~stop)
            if log is not None and (rnd % 50 == 0 or rnd == 1):
                log(f"    round {rnd:5d} | done {int((k_stop >= 0).sum())}/{N} | active {len(slots)} | "
                    f"mean loss {float(ed.mean()):.5f}")
        return out, k_stop, capped
