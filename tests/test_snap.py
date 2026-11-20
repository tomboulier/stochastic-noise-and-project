# -*- coding: utf-8 -*-
"""Self-checks on a tiny random diffusers UNet2DModel, on CPU, in about a minute.

    python tests/test_snap.py            # writes only into a new folder test_outputs/<date-time>_selftest/

1. sigma <-> tau;
2. the SNAP cycle equals diffusers' DDIMScheduler (eta = 0, clipped x0) step by step;
3. P-SNAP (Algorithm 1): finite-difference J v against autograd, |u_i| <= |Delta_i|, gamma = 0 / J = 0 limits;
4. a diffusers folder, a state dict + JSON and a `file.py:function` give the same network;
5. estimators, SNAPED and P-SNAPED: in-domain images stop at once, levels in range, frozen after the stop;
6. the run format: save_run / load_run.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from snap import (DiffusionModel, load_model, SNAPCycle, PSNAPCycle, snap, psnap, snaped, psnaped,  # noqa: E402
                  train_cnn, fit_lda, load_estimator, new_run_dir, save_run, load_run)
from snap.evaluation import auroc, youden_threshold  # noqa: E402
from snap.model import linear_schedule  # noqa: E402

CFG = {"sample_size": 16, "in_channels": 1, "out_channels": 1, "layers_per_block": 1,
       "block_out_channels": [16, 32], "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
       "up_block_types": ["AttnUpBlock2D", "UpBlock2D"], "norm_num_groups": 8}


def tiny_net(seed=0):
    from diffusers import UNet2DModel
    torch.manual_seed(seed)
    net = UNet2DModel(**CFG)
    with torch.no_grad():                    # a random network is chaotic: shrink its output (a real denoiser contracts)
        net.conv_out.weight.mul_(0.01)
        net.conv_out.bias.mul_(0.01)
    return net.eval().requires_grad_(False)


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        check.failed += 1


check.failed = 0


def main():
    torch.set_grad_enabled(False)
    out = new_run_dir("test_outputs", "selftest")
    net = tiny_net()
    m = DiffusionModel(net, linear_schedule(), half=False)
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(6, 1, 16, 16, generator=g) * 2 - 1) * 0.8

    print("1. sigma <-> tau")
    check(all(m.tau_for_sigma(m.sigma(t)) == t for t in (10, 50, 100, 150, 200, 300, 400, 990)), "tau_for_sigma(sigma(tau)) == tau")
    check([round(m.sigma(t), 2) for t in (100, 150, 200, 300, 400)] == [0.34, 0.52, 0.72, 1.24, 2.04],
          "sigma = 0.34, 0.52, 0.72, 1.24, 2.04 at tau = 100 ... 400 (DDPM linear schedule)")

    print("2. SNAP cycle == diffusers DDIM")
    from diffusers import DDIMScheduler
    sch = DDIMScheduler(num_train_timesteps=1000)
    sch.set_timesteps(100)
    for tau in (50, 300):
        xi = torch.randn(x.shape, generator=g)
        mine = SNAPCycle(m, tau=tau)(x, noise=xi)
        ab = sch.alphas_cumprod[tau]
        z = ab.sqrt() * x + (1 - ab).sqrt() * xi
        for t in range(tau, -1, -10):
            z = sch.step(net(z, t).sample, t, z, eta=0.0).prev_sample
        check(float((mine - z).abs().max()) < 1e-4, f"tau {tau}: max |SNAP - diffusers DDIM| = {float((mine - z).abs().max()):.1e}")

    print("3. P-SNAP (Algorithm 1)")
    pc = PSNAPCycle(m, sigma=1.24, gamma=4.67)
    xi = torch.randn(x.shape, generator=g)
    x_tau = pc.add_noise(x, xi)
    v, w = torch.randn(x.shape, generator=g), torch.randn(x.shape, generator=g)
    jv = pc.jvp(x_tau, v)
    with torch.enable_grad():                # dot-product test: <w, J v> (finite differences) = <J^T w, v> (autograd)
        z = (x_tau / pc.sqrt_ab).clone().requires_grad_(True)
        jtw, = torch.autograd.grad(pc.x0hat(pc.sqrt_ab * z), z, grad_outputs=w)
    lhs, rhs = (w * jv).flatten(1).sum(1), (jtw * v).flatten(1).sum(1)
    rel = float(((lhs - rhs).abs() / rhs.abs().clamp_min(1e-3)).max())
    check(rel < 1e-2, f"finite-difference J v vs autograd J^T (dot-product test): relative error {rel:.1e}")
    info = {}
    y = pc(x, noise=xi, info=info)
    delta = SNAPCycle(m, sigma=1.24)(x, noise=xi) - x
    check(bool(((y - x).abs() <= delta.abs() + 1e-6).all()), f"|u_i| <= |Delta_i| ({np.mean(info['fraction_bounded']):.0%} of pixels at the bound)")
    p0 = PSNAPCycle(m, sigma=1.24, gamma=1e6)
    p0.jvp = lambda x_tau, v: torch.zeros_like(v)      # J = 0 and a huge gain: u = Delta, i.e. SNAP
    gap = float((p0(x, noise=xi) - SNAPCycle(m, sigma=1.24)(x, noise=xi)).abs().max())   # x + (y - x) vs y: rounding
    check(gap < 1e-6, f"J = 0, gain -> inf: P-SNAP = SNAP (max gap {gap:.0e}, rounding of x + (y - x))")
    r = psnap(m, x, sigma=0.52, n_cycles=3, gamma=4.67, batch=4, generator=torch.Generator().manual_seed(1), save_cycles=(0, 3))
    check(sorted(r["frames"]) == [0, 3] and r["nfe"][0] == 3 * (16 + 2) and r["mse"].shape == (4, 6),
          f"psnap(): frames, {r['nfe'][0]:.0f} calls (3 x (16 + 2)), MSE at every cycle")

    print("4. loading")
    net.save_pretrained(os.path.join(out, "unet_folder"))
    m2 = load_model(os.path.join(out, "unet_folder"), device="cpu", half=False)
    with open(os.path.join(out, "weights.pt"), "xb") as f:
        torch.save(net.state_dict(), f)
    with open(os.path.join(out, "arch.json"), "x") as f:
        json.dump({"unet": CFG}, f)
    m3 = load_model(os.path.join(out, "weights.pt"), os.path.join(out, "arch.json"), device="cpu", half=False)
    with open(os.path.join(out, "my_model.py"), "x") as f:
        f.write("import os, sys\nsys.path.insert(0, %r)\nfrom snap import DiffusionModel\nfrom snap.model import linear_schedule\n"
                "import test_snap\n\n\ndef build(device, half=None):\n"
                "    return DiffusionModel(test_snap.tiny_net().to(device), linear_schedule(), half=half)\n"
                % os.path.dirname(os.path.abspath(__file__)))
    m4 = load_model(os.path.join(out, "my_model.py") + ":build", device="cpu", half=False)
    e1, e2, e3, e4 = (mm.eps(x, 100) for mm in (m, m2, m3, m4))
    check(bool(torch.equal(e1, e3)), "state dict + JSON: the same epsilon, bit for bit")
    check(bool(torch.equal(e1, e4)), "my_model.py:build (your own code): the same epsilon")
    check(float((e1 - e2).abs().max()) < 1e-6, f"diffusers folder: the same epsilon (max gap {float((e1 - e2).abs().max()):.0e})")
    m64 = DiffusionModel(net, linear_schedule(), half=False, scalars="float64")
    gap = float((SNAPCycle(m64, sigma=0.52)(x, noise=xi) - SNAPCycle(m, sigma=0.52)(x, noise=xi)).abs().max())
    check(gap < 1e-5, f"scalars float64 (guided-diffusion convention) vs float32: max gap {gap:.0e}")

    print("5. estimators, SNAPED and P-SNAPED")
    gd = torch.Generator().manual_seed(2)
    healthy = (torch.rand(400, 1, 16, 16, generator=gd) * 0.6 - 0.8)       # dark images: the model's "domain"
    source = (torch.rand(40, 1, 16, 16, generator=gd) * 0.6 - 0.2)         # brighter: the source domain
    q = lambda t: ((t + 1) * 127.5).round() / 127.5 - 1
    healthy, source = q(healthy), q(source)
    with torch.enable_grad():
        cnn = train_cnn(source[:20], healthy[:200], steps=60, seeds=(0,), widths=(8, 8, 16, 16), device="cpu")
    cnn.calibrate(source[:20], healthy[200:300])
    a_cnn = auroc(cnn.score_all(source[20:]), cnn.score_all(healthy[300:]))
    check(a_cnn > 0.9, f"CNN AUROC on held-out synthetic data {a_cnn:.3f}")
    lda, rows = fit_lda(m, source[:20], healthy[:200], blocks=["down1", "mid"], pools=(1, 2), taus_f=(0, 20),
                        n_f=2, gammas=(0.1, 1.0), folds=4, log=lambda s: None)
    lda.calibrate(source[:20], healthy[200:300])
    a_lda = auroc(lda.score_all(source[20:]), lda.score_all(healthy[300:]))
    check(a_lda > 0.9 and len(rows) == 16, f"LDA AUROC {a_lda:.3f}, {len(rows)} CV rows")
    for est, name in ((cnn, "cnn"), (lda, "lda")):
        est.save(os.path.join(out, f"{name}.pt"))
        back = load_estimator(os.path.join(out, f"{name}.pt"), m, device="cpu")
        check(np.allclose(back.score_all(source), est.score_all(source), atol=1e-5), f"{name}: save / load keeps the scores")
    xin = torch.cat([healthy[300:303], source[20:23]])
    for fn, label in ((snaped, "SNAPED"), (psnaped, "P-SNAPED")):
        r = fn(m, xin, cnn, tau_min=100, tau_max=400, max_cycles=4, stop_quantile=0.99, batch=4,
               generator=torch.Generator().manual_seed(3), save_cycles=(0, 4), monitors={"mean": lambda z: z.mean((1, 2, 3)).numpy()})
        inside = r["scores"][0] <= r["s_healthy"]
        t = r["taus"][0][~inside]
        check(bool((r["cycles_run"][inside] == 0).all()) and bool(((t >= 100) & (t <= 400) & (t % 10 == 0)).all())
              and bool(np.array_equal(r["frames"][0][inside], r["frames"][4][inside])) and r["monitors"]["mean"].shape == (5, 6),
              f"{label}: {int(inside.sum())} in-domain images untouched, first levels tau {t.tolist()}, monitors every cycle")

    print("6. run format and evaluation helpers")
    res = snap(m, x, sigma=0.52, n_cycles=2, batch=4, generator=torch.Generator().manual_seed(4), save_cycles=(0, 1, 2),
               monitors={"mean": lambda z: z.mean((1, 2, 3)).numpy()})
    d = save_run(new_run_dir(out, "run"), res, {"method": "snap", "sigma": 0.52}, [f"im{i}" for i in range(6)])
    run = load_run(d)
    check(run.method == "snap" and run.frame(2, 3).shape == (1, 16, 16) and run.monitor("mean", 1).shape == (6,)
          and os.path.exists(os.path.join(d, "final", "im5.png")), "save_run / load_run")
    thr, j = youden_threshold([0.9, 0.8, 0.3], [0.1, 0.2, 0.35])
    check(abs(thr - 0.25) < 1e-12 and abs(j - 2 / 3) < 1e-12, f"Youden threshold {thr} (J = {j:.3f}, first maximum)")

    print(f"\n{'all checks passed' if not check.failed else f'{check.failed} check(s) FAILED'} (outputs in {out})")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
