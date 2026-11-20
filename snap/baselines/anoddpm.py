# -*- coding: utf-8 -*-
"""AnoDDPM (Wyatt et al., CVPR Workshops 2022): partial diffusion of the input to the timestep t, then the full
ancestral DDPM chain t, t-1, ..., 0 (one network call per timestep, t + 1 in all), every step drawing noise of the
same kind as the partial diffusion: simplex noise (simplex.py) with a model fine-tuned to predict it
(scripts/finetune_simplex_ddpm.py), as in the paper, or Gaussian noise with the original model.

`ddpm_chain` is also AutoDDPM's reconstruction and resampling (autoddpm.py).  The ancestral step follows the
sampler convention of the model (DiffusionModel.eps_from_clipped_x0), as the paper's runs did:

    diffusers convention (ChestMNIST U-Net)   x_{s-1} = sqrt(ab_{s-1}) x0 + sqrt(1 - ab_{s-1} - v) eps + sqrt(v) z,
                                              v = (1 - ab_{s-1}) / (1 - ab_s) (1 - ab_s / ab_{s-1}) (DDIM with
                                              eta = 1, Song et al. 2021, Eq. 12), schedule scalars as the model
                                              gives them (float32 tensors);
    guided-diffusion convention (FFHQ ADM)    GaussianDiffusion.p_sample: posterior mean c1 x0 + c2 x_s with float64
                                              coefficients, log-variance learned by the network (LEARNED_RANGE) when
                                              it outputs one, else the clipped posterior log-variance.

x0 is clipped to [-1, 1] at every step, and the last step (s = 0) returns it.  Paper grid (Fig. 6): t in
{100, 200, 250, 300, 400}, simplex noise; faces: t in {50, 100, 200}, Gaussian noise (not in Table 1).

    python scripts/run_baseline.py --method anoddpm --model weights/chestmnist_224_simplex.pt \\
        --config configs/chestmnist_224.json --input data/chestmnist224/cardiomegaly_test.npy --n 64 --group Cardiomegaly
"""

import numpy as np
import torch

from .simplex import simplex_noise

__all__ = ["anoddpm", "ddpm_chain", "schedule_betas"]


def schedule_betas(model):
    """float64 betas of the model's schedule: the linear DDPM betas (np.linspace, scaled to T steps as in
    guided-diffusion) when the model's alpha_bar is exactly that schedule, else 1 - alpha_bar_t / alpha_bar_{t-1}."""
    ac = np.asarray(model.alphas_cumprod, np.float64)
    T = len(ac)
    scale = 1000.0 / T
    b = np.linspace(scale * 0.0001, scale * 0.02, T, dtype=np.float64)
    if np.array_equal(np.cumprod(1.0 - b), ac):
        return b
    return 1.0 - ac / np.append(1.0, ac[:-1])


class _Posterior:
    """Coefficients of q(x_{s-1} | x_s, x0) in float64, as guided-diffusion's GaussianDiffusion computes them."""

    def __init__(self, model):
        b = schedule_betas(model)
        ab = np.asarray(model.alphas_cumprod, np.float64)
        ab_prev = np.append(1.0, ab[:-1])
        self.betas = b
        post_var = b * (1.0 - ab_prev) / (1.0 - ab)
        self.post_logvar = np.log(np.append(post_var[1], post_var[1:]))
        self.c1 = b * np.sqrt(ab_prev) / (1.0 - ab)
        self.c2 = (1.0 - ab_prev) * np.sqrt(1.0 - b) / (1.0 - ab)


def _ab(model, t, device):
    """alpha_bar_t as the model's scalars: a float32 0-dim tensor (diffusers) or a Python float (guided-diffusion)."""
    a = model.ab_value(t)
    return torch.tensor(a, dtype=torch.float32, device=device) if model.scalars == "float32" else a


def _sqrt(v):
    return v.sqrt() if torch.is_tensor(v) else v ** 0.5


def _pos(v):
    return v.clamp_min(0) if torch.is_tensor(v) else max(v, 0.0)


@torch.no_grad()
def ddpm_chain(model, x, t, generator=None, noise="gaussian", octaves=6, persistence=0.8, clip=True, x_noise=None):
    """Partial diffusion of x (B, C, H, W) in [-1, 1] to the timestep t, then the ancestral chain t, ..., 0.

    noise     "gaussian" or "simplex", for the partial diffusion and every step
    x_noise   imposes the noise of the partial diffusion (tests)
    -> the image at step 0 (t + 1 network calls)."""
    t = int(t)
    shape = tuple(x.shape)

    def draw():
        if noise == "simplex":
            return simplex_noise(shape, x.device, generator, octaves, persistence)
        return torch.randn(shape, device=x.device, generator=generator)

    xi = draw() if x_noise is None else x_noise
    sa, sb = model.coef(t)
    xt = sa * x + sb * xi
    gd = model.eps_from_clipped_x0
    post = _Posterior(model) if gd else None
    if gd and model.prediction_type != "epsilon":
        raise ValueError("the guided-diffusion ancestral step needs an epsilon-prediction network")
    C = x.shape[1]
    for s in range(t, -1, -1):
        var_out = None
        if gd:
            out = model.forward(xt, s)
            e = out[:, :C]
            if out.shape[1] > C:
                var_out = out[:, C:]
        else:
            e = model.eps(xt, s)
        if gd:
            sa, sb = model.coef(s)
            x0 = (xt - sb * e) / sa
        else:
            a_t = _ab(model, s, x.device)
            x0 = (xt - _sqrt(1 - a_t) * e) / _sqrt(a_t)
        if clip:
            x0 = x0.clamp(-1, 1)
        if s == 0:
            return x0
        if gd:                                           # GaussianDiffusion.p_sample
            mean = float(post.c1[s]) * x0 + float(post.c2[s]) * xt
            if var_out is not None:                      # LEARNED_RANGE
                min_log = float(post.post_logvar[s])
                max_log = float(np.log(post.betas[s]))
                frac = (var_out + 1.0) / 2.0
                logv = frac * max_log + (1.0 - frac) * min_log
            else:
                logv = torch.full_like(xt, float(post.post_logvar[s]))
            z = draw()
            xt = mean + torch.exp(0.5 * logv) * z
        else:                                            # DDIM with eta = 1
            a_p = _ab(model, s - 1, x.device)
            var = (1 - a_p) / (1 - a_t) * (1 - a_t / a_p)
            std = _sqrt(var)
            xt = _sqrt(a_p) * x0 + _sqrt(_pos(1 - a_p - var)) * e + std * draw()
    return xt


def anoddpm(model, x, t, generator=None, noise="simplex", octaves=6, persistence=0.8):
    """AnoDDPM: x (B, C, H, W) in [-1, 1] -> (y, nfe = t + 1)."""
    return ddpm_chain(model, x, t, generator, noise, octaves, persistence), int(t) + 1
