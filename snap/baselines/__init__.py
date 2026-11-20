# -*- coding: utf-8 -*-
"""The competing methods of the paper (Sec. 5: Fig. 6 on ChestMNIST, Table 1 on degraded faces), run on the SAME
pre-trained diffusion model as SNAP, through `snap.model.DiffusionModel` (the diffusers U-Net of ChestMNIST or the
guided-diffusion ADM of FFHQ), so that only the method changes.

    SDEdit     sdedit.py     noise to tau, deterministic DDIM back to 0: exactly one SNAP cycle
    AnoDDPM    anoddpm.py    partial diffusion to t, then the full ancestral DDPM chain t, t-1, ..., 0; simplex noise
                             (simplex.py) with a model fine-tuned on it (scripts/finetune_simplex_ddpm.py), or Gaussian
    AutoDDPM   autoddpm.py   mask (DDPM reconstruction at t_mask, smoothed anomaly map above a quantile of the maps of
                             healthy validation images, dilated), stitch, resample at t_resample, keep the input
                             outside the mask
    VAE        vae.py        pseudo-healthy reconstruction (decoded posterior mean) of a VAE trained on in-domain
                             images (scripts/train_vae.py)
    DIIP       diip.py       Diffusion Image Prior: the initial noise of a differentiated 10-step DDIM is optimised to
                             fit the degraded image, with early stopping

Command line: scripts/run_baseline.py (one output per setting, snap/runs.py SettingsRun).

Seeds
-----
Every setting of a run draws its noise from its own generator, seeded with `derived_seed(seed, *keys)`:

    derived_seed(seed, *keys) = (seed * 1000003 + int(md5("|".join(keys))[:8], 16)) mod 2^31

with the keys of the paper's runs (`setting_keys`), so that a recipe reproduces them:

    recipe "chestmnist" (Fig. 6)   one generator per (setting, image group); the images of a group go through it in
                                   batches of 16 in file order.
        SDEdit     ("sdedit_tau<tau>", group)
        AnoDDPM    ("anoddpm_s_t<t>", group) with simplex noise, ("anoddpm_g_t<t>", group) with Gaussian noise
        AutoDDPM   ("autoddpm_tm<t_mask>_tr<t_resample>", group); its threshold has its own generator ("seuil", t_mask)
        group      "Cardiomegaly", "Effusion", "Edema" (64 test images each), or "sain_reel" (the 64 healthy test images)
    recipe "faces" (Table 1)        one generator per setting, through which ALL the degraded photographs of a run go
                                   in batches of 16 (the 4 degradations x (100 test + 32 validation photographs), in
                                   the order blur, JPEG, pixelation, noise), AutoDDPM's threshold first.
        SDEdit     ("sdedit", tau)        AnoDDPM ("anoddpm", t)        AutoDDPM ("autoddpm", t_mask)
        DIIP       one initial noise per image, CPU generator ("diip", degradation key, row in that order)

The key strings are those of the original experiments and are kept verbatim (`HEALTHY_GROUP`, `THRESHOLD_KEY`,
`DEGRADATION_KEYS`): they only feed md5.
"""

import hashlib

import numpy as np
import torch

__all__ = ["derived_seed", "setting_keys", "diip_keys", "make_generator", "mse_uint8", "RECIPES", "HEALTHY_GROUP",
           "THRESHOLD_KEY", "DEGRADATION_KEYS", "DIIP_CRITERION", "sdedit", "anoddpm", "ddpm_chain", "simplex_noise",
           "autoddpm", "autoddpm_threshold", "anomaly_map", "PseudoHealthyVAE", "train_vae", "reconstruct",
           "load_vae", "DIIP"]

RECIPES = ("chestmnist", "faces")
HEALTHY_GROUP = "sain_reel"           # seed key of the healthy test radiographs in the ChestMNIST runs
THRESHOLD_KEY = "seuil"               # seed key of AutoDDPM's threshold in the ChestMNIST runs
DEGRADATION_KEYS = {"blur": "flou", "jpeg": "jpeg", "pixelation": "pixel", "noise": "bruit"}   # faces runs
DIIP_CRITERION = {"blur": "sharpness", "pixelation": "sharpness", "jpeg": "loss", "noise": "loss"}   # DIIP, Sec. 4


def derived_seed(seed, *keys):
    """The per-setting seed of the paper's runs: (seed * 1000003 + first 32 bits of md5("k1|k2|...")) mod 2^31."""
    h = hashlib.md5("|".join(map(str, keys)).encode()).hexdigest()
    return (int(seed) * 1_000_003 + int(h[:8], 16)) % (2 ** 31)


def setting_keys(recipe, method, level, group=None, resample_tau=50, noise="simplex"):
    """The keys hashed by `derived_seed` for one setting of a run.

    recipe        "chestmnist" or "faces" (see the module docstring)
    method        "sdedit", "anoddpm", "autoddpm" or "autoddpm_threshold"
    level         tau (SDEdit), t (AnoDDPM) or t_mask (AutoDDPM), an integer timestep
    group         ChestMNIST recipe: the image group ("Cardiomegaly", "Effusion", "Edema", HEALTHY_GROUP)
    """
    level = int(level)
    if recipe == "chestmnist":
        if method == "sdedit":
            return (f"sdedit_tau{level}", group)
        if method == "anoddpm":
            return (f"anoddpm_{'s' if noise == 'simplex' else 'g'}_t{level}", group)
        if method == "autoddpm":
            return (f"autoddpm_tm{level}_tr{int(resample_tau)}", group)
        if method == "autoddpm_threshold":
            return (THRESHOLD_KEY, level)
    elif recipe == "faces":
        if method in ("sdedit", "anoddpm", "autoddpm"):
            return (method, level)
        if method == "autoddpm_threshold":
            return ("autoddpm", level)                   # the same generator as the images, used first
    raise ValueError(f"no seed keys for recipe {recipe!r}, method {method!r}")


def diip_keys(key, row):
    """Keys of the initial noise of DIIP for one image: ("diip", degradation key or group, row)."""
    return ("diip", key, int(row))


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def mse_uint8(a_u8, b_u8):
    """Per-image MSE between uint8 images (N, C, H, W), both read in [0, 1] in float64 (as for the paper's numbers)."""
    a = np.asarray(a_u8) / 255.0
    b = np.asarray(b_u8) / 255.0
    return ((a - b) ** 2).mean(axis=tuple(range(1, a.ndim)))


from .sdedit import sdedit  # noqa: E402
from .simplex import simplex_noise  # noqa: E402
from .anoddpm import anoddpm, ddpm_chain  # noqa: E402
from .autoddpm import autoddpm, autoddpm_threshold, anomaly_map  # noqa: E402
from .vae import PseudoHealthyVAE, train_vae, reconstruct, load_vae  # noqa: E402
from .diip import DIIP  # noqa: E402
