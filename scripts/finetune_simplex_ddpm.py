# -*- coding: utf-8 -*-
"""Fine-tune a healthy diffusers DDPM to predict SIMPLEX noise, for the AnoDDPM baseline (snap/baselines/anoddpm.py).

AnoDDPM (Wyatt et al., CVPR Workshops 2022) trains its DDPM on healthy images with simplex noise instead of Gaussian
noise.  As in the paper's runs, we start from the healthy Gaussian DDPM (same training images) and fine-tune it on
simplex noise with the standard epsilon-prediction objective:

    eps = simplex noise (6 octaves, persistence 0.8, snap/baselines/simplex.py);  t ~ U{0, ..., 999}
    x_t = DDPMScheduler(1000 steps, linear betas 1e-4 .. 0.02).add_noise(x, eps, t);  loss = MSE(eps_theta(x_t, t), eps)

AdamW (lr 2e-5), batch 16, EMA 0.999 of the weights (the EMA weights are the output), fp16 autocast + GradScaler on
CUDA.  Paper: 40000 steps requested, at most 9 hours (--max_hours 9).

    python scripts/finetune_simplex_ddpm.py --model weights/chestmnist_224.pt --config configs/chestmnist_224.json \\
        --images data/chestmnist224/healthy_train.npy --steps 40000 --max_hours 9

Writes a new folder <out>/<date-time>_simplex_ddpm/: resume.pt (training state every 1000 steps, the only file that
is replaced, atomically, by a more advanced one), log.json, and the final EMA weights in a NEW file
ema_<steps>.pt (a state dict for --model ... --config ... of scripts/run_baseline.py).  --resume <folder> continues a
run.  Differences from the paper's run: the data loader's shuffling is seeded (--seed), the noise generator is a CPU
generator seeded with --seed + the step reached, as there.
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from snap import load_model, new_run_dir  # noqa: E402
from snap.images import load_uint8  # noqa: E402
from snap.baselines.simplex import simplex_noise  # noqa: E402


class _Uint8Images(torch.utils.data.Dataset):
    """uint8 (N, C, H, W) -> float32 (C, H, W) in [-1, 1], converted on the fly."""

    def __init__(self, a):
        self.a = a

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return torch.from_numpy(self.a[i].astype(np.float32) / 127.5 - 1.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="the healthy Gaussian DDPM (diffusers), as in scripts/project.py")
    p.add_argument("--config", default=None)
    p.add_argument("--images", required=True, help="healthy training images: uint8 array or folder")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--octaves", type=int, default=6)
    p.add_argument("--persistence", type=float, default=0.8)
    p.add_argument("--max_hours", type=float, default=0.0, help="stop after this time (0: no limit)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="", help="folder of a previous run to continue")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                    # noqa: BLE001
        pass
    from diffusers import DDPMScheduler
    model = load_model(a.model, a.config, a.device, half=False)
    device = model.device
    net = model.net
    net.requires_grad_(True)
    ema = copy.deepcopy(net).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    fp16 = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    sched = DDPMScheduler(num_train_timesteps=1000)
    out = a.resume or new_run_dir(a.out, "simplex_ddpm")
    state_f = os.path.join(out, "resume.pt")
    step0, log = 0, []
    if a.resume and os.path.exists(state_f):
        e = torch.load(state_f, map_location="cpu", weights_only=False)
        net.load_state_dict(e["model"])
        ema.load_state_dict(e["ema"])
        opt.load_state_dict(e["opt"])
        if fp16 and e.get("scaler"):
            scaler.load_state_dict(e["scaler"])
        step0, log = int(e["step"]), list(e.get("log", []))
        print(f"[resume] step {step0}")
    images, _ = load_uint8(a.images, model.channels)
    loader = torch.utils.data.DataLoader(_Uint8Images(images), batch_size=a.batch, shuffle=True, drop_last=True,
                                         num_workers=a.workers, pin_memory=fp16, persistent_workers=a.workers > 0,
                                         generator=torch.Generator().manual_seed(a.seed + step0))
    gen = torch.Generator().manual_seed(a.seed + step0)
    print(f"[data] {len(images)} images | {len(loader)} batches per pass | {a.steps} steps, lr {a.lr}, EMA {a.ema} -> {out}")

    def save_state(step):
        tmp = os.path.join(out, f"resume_{step}.pt.tmp")
        with open(tmp, "xb") as f:
            torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict() if fp16 else None, "step": step, "log": log, "args": vars(a)}, f)
        os.replace(tmp, state_f)

    t0, step, total, n = time.time(), step0, 0.0, 0
    done = step >= a.steps
    while not done:
        for x in loader:
            net.train()
            x = x.to(device, non_blocking=True)
            b = x.shape[0]
            eps = simplex_noise(tuple(x.shape), device, gen, a.octaves, a.persistence)
            t = torch.randint(0, 1000, (b,), generator=gen).to(device)
            xt = sched.add_noise(x, eps, t)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = F.mse_loss(net(xt, t).sample.float(), eps)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(a.ema).add_(pn.detach(), alpha=1 - a.ema)
            step += 1
            total += loss.item()
            n += 1
            if step % 200 == 0 or step == a.steps:
                h = (time.time() - t0) / 3600
                log.append({"step": step, "loss": total / max(1, n), "hours": h})
                print(f"  step {step}/{a.steps} | loss {total / max(1, n):.4f} | {h:.2f} h")
                total, n = 0.0, 0
            if step % 1000 == 0:
                save_state(step)
            if step >= a.steps or (a.max_hours and (time.time() - t0) / 3600 >= a.max_hours):
                done = True
                break
    save_state(step)
    final = os.path.join(out, f"ema_{step}.pt")
    with open(final, "xb") as f:
        torch.save(ema.state_dict(), f)
    with open(os.path.join(out, f"log_{step}.json"), "x") as f:
        json.dump({"args": vars(a), "step": step, "weights": final, "log": log}, f, indent=1)
    print(f"[done] EMA weights -> {final} ({step} steps)")
    return final


if __name__ == "__main__":
    main()
