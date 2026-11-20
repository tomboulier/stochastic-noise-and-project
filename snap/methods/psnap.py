# -*- coding: utf-8 -*-
"""P-SNAP: projected stochastic noise-and-project cycles (paper, Sec. 3.1, Algorithm 1).

Notation: x the current image, z = x + sigma xi the noised image (x_tau = sqrt(alpha_bar) z),
x0hat(x_tau) = (x_tau - sqrt(1 - alpha_bar) eps(x_tau, tau)) / sqrt(alpha_bar) the one-step
(Tweedie) denoiser, D(z) = x0hat(sqrt(alpha_bar) z) and J = dD/dz its Jacobian.

Algorithm 1 (one P-SNAP cycle):

    1  xi ~ N(0, I);  z <- x + sigma xi;  Delta <- C_sigma(x, xi) - x        (the SNAP displacement)
    2  h <- r ||z|| / ||Delta||
    3  J Delta <- [x0hat(z + h Delta) - x0hat(z - h Delta)] / 2h            (central difference, float32)
    4  u <- clip(gamma (Delta - J Delta), +-|Delta|)                       pixel by pixel
    5  return x + u

(I - J) Delta keeps the normal part of the displacement and removes the random step along the
manifold; the gain gamma >= 1 lets the pixels whose displacement is mostly normal move as with
SNAP (they reach the bound), while the others stay attenuated.  gamma = 1 is the projected
displacement up to the clipping.  The paper uses gamma = 4.67 on ChestMNIST, gamma = 3 on FFHQ.
x0hat is evaluated in float32 and not clipped (clipping would zero J on saturated pixels).
"""

import math

import torch

from .snap import SNAPCycle, run_cycles

__all__ = ["PSNAPCycle", "psnap"]


def _norm(v):
    return v.flatten(1).norm(dim=1)


def _per_image(s, like):
    return s.view(-1, *([1] * (like.dim() - 1)))


class PSNAPCycle(SNAPCycle):
    """One P-SNAP cycle (Algorithm 1).

    gamma      gain (>= 1)
    h_rel      relative finite-difference step r (paper: 0.01 on ChestMNIST, 0.003 on FFHQ)
    nfe        network calls of the DDIM chain; nfe_fp32 = 2 one-step float32 calls for J Delta.
    """

    name = "psnap"

    def __init__(self, model, tau=None, sigma=None, gamma=4.67, stride=10, h_rel=0.01, clip=True):
        super().__init__(model, tau, sigma, stride, clip)
        self.gamma, self.h_rel = float(gamma), float(h_rel)
        self.sqrt_ab = math.sqrt(model.ab_value(self.tau))
        self.nfe_fp32 = 2

    def __repr__(self):
        return (f"PSNAPCycle(sigma={self.sigma:.3f}, tau={self.tau}, gamma={self.gamma:g}, r={self.h_rel:g}, "
                f"nfe={self.nfe}+{self.nfe_fp32} fp32)")

    def x0hat(self, x_t):
        """One-step denoiser at level tau, float32, not clipped."""
        x_t = x_t.float()
        eps = self.model.eps(x_t, self.tau, fp32=True)
        sa, sb = self.model.coef(self.tau)
        return (x_t - sb * eps) / sa

    def jvp(self, x_tau, v):
        """J v at z = x_tau / sqrt(alpha_bar), central difference with h = r ||z|| / ||v||."""
        z = x_tau / self.sqrt_ab
        h = self.h_rel * _norm(z) / _norm(v).clamp_min(1e-12)
        hh = _per_image(h, v)
        dz = hh * self.sqrt_ab * v
        return (self.x0hat(x_tau + dz) - self.x0hat(x_tau - dz)) / (2.0 * hh)

    @torch.no_grad()
    def __call__(self, x, generator=None, noise=None, info=None):
        xi = torch.randn(x.shape, device=x.device, generator=generator) if noise is None else noise
        x_tau = self.add_noise(x, xi)
        delta = self.denoise(x_tau) - x                       # line 1: the SNAP displacement, same xi
        j_delta = self.jvp(x_tau, delta)                      # lines 2-3
        bound = delta.abs()
        u = torch.maximum(torch.minimum(self.gamma * (delta - j_delta), bound), -bound)   # line 4
        if info is not None:
            rd = math.sqrt(x[0].numel())
            info["step_snap"] = (_norm(delta) / rd).cpu().numpy()
            info["step_psnap"] = (_norm(u) / rd).cpu().numpy()
            info["fraction_bounded"] = ((self.gamma * (delta - j_delta)).abs() >= bound).float().flatten(1).mean(1).cpu().numpy()
        return x + u                                          # line 5


def psnap(model, x, sigma=None, n_cycles=50, gamma=4.67, tau=None, stride=10, h_rel=0.01, batch=64,
          generator=None, **kw):
    """P-SNAP: n_cycles P-SNAP cycles at a fixed level.  Returns the dict of `run_cycles`."""
    return run_cycles(PSNAPCycle(model, tau, sigma, gamma, stride, h_rel), x, n_cycles, batch, generator, **kw)
