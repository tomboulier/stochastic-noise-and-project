# -*- coding: utf-8 -*-
"""Self-checks of the competing methods (snap/baselines) on a tiny random diffusers UNet2DModel, on CPU, in a few
minutes.

    python tests/test_baselines.py [--out test_outputs]     # writes only into a new folder <out>/<date-time>_baselines/

1. seeds: the per-setting derivation of the paper's runs;
2. SDEdit is one SNAP cycle; the DDPM chain at t = 0 is the clipped one-step denoiser; simplex noise is standardised;
3. AutoDDPM leaves the input unchanged outside its mask;
4. VAE: training, reconstruction, save / load;
5. DIIP: early stopping, capped images, outputs in [-1, 1];
6. scripts/run_baseline.py and scripts/train_vae.py: SettingsRun folders, MSE to x0.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(1, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from snap import DiffusionModel, SNAPCycle, new_run_dir  # noqa: E402
from snap.model import linear_schedule  # noqa: E402
from snap.runs import load_run  # noqa: E402
from snap.baselines import (derived_seed, setting_keys, sdedit, ddpm_chain, simplex_noise, autoddpm,  # noqa: E402
                            autoddpm_threshold, train_vae, reconstruct, load_vae, DIIP, mse_uint8)
from test_snap import tiny_net, check  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="test_outputs")
    a = p.parse_args(argv)
    out = new_run_dir(a.out, "baselines")
    net = tiny_net()
    m = DiffusionModel(net, linear_schedule(), half=False)
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(6, 1, 16, 16, generator=g) * 2 - 1) * 0.8

    print("1. seeds")
    check(derived_seed(0, *setting_keys("chestmnist", "sdedit", 50, "Cardiomegaly")) == 1383064105,
          "derived_seed(0, 'sdedit_tau50', 'Cardiomegaly') = 1383064105")
    check(setting_keys("faces", "autoddpm_threshold", 100) == setting_keys("faces", "autoddpm", 100),
          "faces recipe: AutoDDPM's threshold and images share one generator")

    print("2. SDEdit, DDPM chain, simplex noise")
    with torch.no_grad():
        xi = torch.randn(x.shape, generator=g)
        y, nfe = sdedit(m, x, 300, noise=xi)
        check(torch.equal(y, SNAPCycle(m, tau=300)(x, noise=xi)) and nfe == 31, "SDEdit at tau 300 = one SNAP cycle, 31 calls")
        y0 = ddpm_chain(m, x, 0, x_noise=xi)
        sa, sb = m.coef(0)
        ref = ((sa * x + sb * xi - sb * m.eps(sa * x + sb * xi, 0)) / sa).clamp(-1, 1)
        check(torch.equal(y0, ref), "DDPM chain at t = 0: the clipped one-step denoiser")
        s = simplex_noise((4, 1, 32, 32), "cpu", torch.Generator().manual_seed(1))
        mu, sd = s.mean((2, 3)).abs().max().item(), (s.std((2, 3)) - 1).abs().max().item()
        check(mu < 1e-5 and sd < 1e-5, f"simplex noise standardised per image (|mean| {mu:.0e}, |std - 1| {sd:.0e})")

        print("3. AutoDDPM")
        thr = autoddpm_threshold(m, x[:4], 20, 0.95, torch.Generator().manual_seed(2), batch=2)
        y, nfe, mask = autoddpm(m, x, 20, 10, thr, torch.Generator().manual_seed(3))
        outside = (mask == 0).expand_as(x)
        check(thr > 0 and nfe == 32 and torch.equal(y[outside], x[outside]),
              f"threshold {thr:.3f}, {nfe} calls, input unchanged outside the mask ({float(mask.mean()):.0%} masked)")

    print("4. VAE")
    vae = train_vae(x, 0.1, steps=5, batch=4, latent=8, seed=0, device="cpu", log_every=5)
    r = reconstruct(vae, x, batch=4)
    path = os.path.join(out, "vae_b0.1.pt")
    with open(path, "xb") as f:
        torch.save(vae.state_dict(), f)
    back = load_vae(path, 16, 1, latent=8)
    check(r.shape == x.shape and torch.equal(reconstruct(back, x, batch=4), r), "train, reconstruct, save / load")

    print("5. DIIP")
    dp = DIIP(m, k_min=2, k_max=5, lr=0.3, eps=0.5)
    z0 = torch.randn((3, 1, 16, 16), generator=torch.Generator().manual_seed(4))
    y, k, capped = dp.restore(x[:3], "loss", z0, batch=2)
    check(bool((k >= 2).all() and (k <= 5).all()) and bool(np.all(capped == (k == 5))) and float(y.abs().max()) <= 1,
          f"stopping iterations {k.tolist()}, capped {capped.tolist()}")
    y2, k2, _ = DIIP(m, k_min=2, k_max=5, lr=0.3, eps=0.5).restore(x[:3], "loss", z0, batch=1)
    check(np.array_equal(k, k2) and float((y - y2).abs().max()) < 1e-3,
          f"one image per batch: the same stops, the same images up to floating point (max gap "
          f"{float((y - y2).abs().max()):.0e})")

    print("6. command lines")
    net.save_pretrained(os.path.join(out, "unet"))
    u8 = ((x[:, 0].numpy() + 1) * 127.5).round().astype(np.uint8)
    for name, arr in (("images.npy", u8), ("healthy.npy", u8[::-1].copy())):
        with open(os.path.join(out, name), "xb") as f:
            np.save(f, arr)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
    import run_baseline
    import train_vae as train_vae_cli
    common = ["--model", os.path.join(out, "unet"), "--input", os.path.join(out, "images.npy"), "--device", "cpu",
              "--out", out, "--group", "Cardiomegaly"]
    d = run_baseline.main(common + ["--method", "sdedit", "--taus", "50,100"])
    run = load_run(d)
    check(run.labels == ["tau=50", "tau=100"] and run["outputs"].shape == (2, 6, 1, 16, 16)
          and np.allclose(run["mse"][1], mse_uint8(run["outputs"][1], run["x0"])) and run["nfe"].tolist() == [6, 11],
          "run_baseline sdedit: 2 settings, MSE to x0, network calls")
    d = run_baseline.main(common + ["--method", "anoddpm", "--taus", "5", "--batch", "4"])
    check(load_run(d)["outputs"].shape == (1, 6, 1, 16, 16), "run_baseline anoddpm (simplex noise)")
    d = run_baseline.main(common + ["--method", "autoddpm", "--mask_taus", "10", "--resample_tau", "5",
                                    "--healthy", os.path.join(out, "healthy.npy"), "--n_healthy", "4"])
    run = load_run(d)
    check("mon_mask_fraction" in run and run.meta["settings"][0]["threshold"] > 0, "run_baseline autoddpm")
    vd = train_vae_cli.main(["--images", os.path.join(out, "images.npy"), "--betas", "1", "--steps", "3", "--batch", "4",
                             "--latent", "8", "--device", "cpu", "--out", out])
    d = run_baseline.main(["--method", "vae", "--vae", os.path.join(vd, "vae_b1.pt"), "--input",
                           os.path.join(out, "images.npy"), "--device", "cpu", "--out", out])
    check(load_run(d).labels == ["beta=1"], "train_vae + run_baseline vae (latent read from vae.json)")
    d = run_baseline.main(common + ["--method", "diip", "--k_min", "2", "--k_max", "3", "--lr", "0.3", "--n", "2"])
    run = load_run(d)
    check(run["mon_iterations"].shape == (1, 2) and run.labels == ["early_stop"], "run_baseline diip")

    print(f"\n{'all checks passed' if not check.failed else f'{check.failed} check(s) FAILED'} (outputs in {out})")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
