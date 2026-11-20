# -*- coding: utf-8 -*-
"""SDEdit (Meng et al., ICLR 2022) with the deterministic DDIM sampler: noise the input to the timestep tau, then
integrate DDIM (eta = 0, stride 10, x0 clipped to [-1, 1]) back to 0.  This is exactly one SNAP cycle
(snap/methods/snap.py): same noising, same steps, same convention of the model (diffusers or guided-diffusion).

Paper grid (Fig. 6, ChestMNIST): tau in {50, 100, 150, 200, 300, 400, 600}; faces (Table 1): tau in
{50, 100, 150, 200, 300, 400} at the moderate level, {50, ..., 400, 500} at the strong level.

    python scripts/run_baseline.py --method sdedit --model weights/chestmnist_224.pt --config configs/chestmnist_224.json \\
        --input data/chestmnist224/cardiomegaly_test.npy --n 64 --group Cardiomegaly
"""

from ..methods.snap import SNAPCycle

__all__ = ["sdedit"]


def sdedit(model, x, tau=None, sigma=None, stride=10, generator=None, noise=None):
    """x (B, C, H, W) in [-1, 1] -> (y, nfe).  tau is rounded down to a multiple of the stride; nfe = tau / stride + 1.
    `noise` imposes the Gaussian noise (otherwise drawn from `generator`, shape of x, on x's device)."""
    cycle = SNAPCycle(model, tau, sigma, stride)
    return cycle(x, generator, noise), cycle.nfe
