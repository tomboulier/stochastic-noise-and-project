# -*- coding: utf-8 -*-
"""Project images onto the domain of a pre-trained diffusion model with SNAP, P-SNAP, SNAPED or P-SNAPED.

    # SNAP and P-SNAP at a fixed level (P-SNAP: Algorithm 1, gain gamma)
    python scripts/project.py --model weights/chestmnist_224.pt --config configs/chestmnist_224.json \\
        --input data/chestmnist224/cardiomegaly_test.npy --n 64 --method snap  --sigma 1.24 --cycles 600
    python scripts/project.py ... --method psnap --sigma 1.24 --gamma 4.67 --cycles 600

    # SNAPED and P-SNAPED (Algorithm 2; train the estimator first: scripts/train_estimator.py)
    python scripts/project.py ... --method psnaped --estimator estimators/cnn_cardiomegaly.pt --gamma 4.67

    # read a classifier at every cycle while projecting (stored in the run, used by the paper's figures)
    python scripts/project.py ... --classifier weights/medmnist --heads Cardiomegaly

Writes a new folder <out>/<date-time>_<method>/ (snap/runs.py): run.json, trajectory.npz (inputs, images
at --save_cycles, MSE to the input at every cycle, monitors, network calls, driven: scores and levels),
final/<name>.png.  Nothing is overwritten.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from snap import load_model, load_images, first, new_run_dir, save_run, load_estimator  # noqa: E402
from snap.methods import SNAPCycle, PSNAPCycle, run_cycles, estimator_driven  # noqa: E402

DEFAULT_SAVE = "0,1,2,3,5,7,10,15,20,30,50,70,100,150,200,300,400,500,600"


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="diffusers folder / Hub id, weight file (with --config), or file.py:build")
    p.add_argument("--config", default=None, help="model JSON (configs/)")
    p.add_argument("--input", required=True, help="folder of images, or a .npy / .npz array of uint8 images")
    p.add_argument("--method", choices=("snap", "psnap", "snaped", "psnaped"), required=True)
    p.add_argument("--n", type=int, default=0, help="number of images, from --start in file order (0: all)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--resize", action="store_true", help="resize the inputs to the model's size")
    g = p.add_argument_group("fixed level (snap, psnap)")
    g.add_argument("--sigma", type=float, default=1.24, help="noise level sigma (paper: 0.52, 0.72, 1.24, 2.04)")
    g.add_argument("--tau", type=int, default=None, help="timestep instead of --sigma")
    g.add_argument("--cycles", type=int, default=600)
    g = p.add_argument_group("P-SNAP (Algorithm 1)")
    g.add_argument("--gamma", type=float, default=4.67, help="gain (paper: 4.67 ChestMNIST, 3 FFHQ)")
    g.add_argument("--h_rel", type=float, default=0.01, help="relative finite-difference step r (paper: 0.01 ChestMNIST, 0.003 FFHQ)")
    g = p.add_argument_group("estimator-driven (snaped, psnaped; Algorithm 2)")
    g.add_argument("--estimator", default=None, help="file written by scripts/train_estimator.py")
    g.add_argument("--sigma_min", type=float, default=0.34)
    g.add_argument("--sigma_max", type=float, default=2.04)
    g.add_argument("--tau_min", type=int, default=None, help="timestep instead of --sigma_min")
    g.add_argument("--tau_max", type=int, default=None, help="timestep instead of --sigma_max")
    g.add_argument("--max_cycles", type=int, default=500)
    g.add_argument("--stop_quantile", type=float, default=0.5, help="quantile of held-out in-domain scores")
    g = p.add_argument_group("monitors (measurements during the run; the methods do not use them)")
    g.add_argument("--classifier", default=None, help="MedMNIST weight file(s) / folder, or file.py:build")
    g.add_argument("--heads", default="", help="classes read at every cycle, e.g. Cardiomegaly,Effusion,Edema")
    g.add_argument("--monitor_cycles", default="all", help="'all' or a list (fixed level only)")
    g = p.add_argument_group("run")
    g.add_argument("--stride", type=int, default=10, help="DDIM step (timesteps)")
    g.add_argument("--batch", type=int, default=64, help="images per network call (the noise depends on it)")
    g.add_argument("--seed", type=int, default=0, help="seed of the run's generator")
    g.add_argument("--save_cycles", default=DEFAULT_SAVE, help="cycles whose images are kept")
    g.add_argument("--device", default=None)
    g.add_argument("--fp32", action="store_true", help="full precision for the DDIM chain (default: fp16 on CUDA)")
    g.add_argument("--label", default="", help="suffix of the output folder")
    g.add_argument("--out", default="outputs")
    return p.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                    # noqa: BLE001
        pass
    a = parse(argv)
    model = load_model(a.model, a.config, a.device, half=(False if a.fp32 else None))
    print(f"[model] {model}")
    x, names = load_images(a.input, channels=model.channels, size=model.image_size, resize=a.resize)
    x, names = first(x, names, a.n, a.start)
    print(f"[input] {len(x)} images from {a.input} (from #{a.start})")
    gen = torch.Generator(device=model.device).manual_seed(a.seed)
    save_k = [int(k) for k in a.save_cycles.split(",") if k != ""]
    monitors = {}
    if a.classifier:
        from snap.evaluation import load_classifier
        clf = load_classifier(a.classifier, model.device)
        heads = [h for h in a.heads.split(",") if h]
        monitors = {f"p_{h}": clf.head(h) for h in heads} if heads else {"p": clf.probs}
    meta = {"method": a.method, "model": a.model, "model_repr": repr(model), "input": a.input, "start": a.start,
            "names": names, "seed": a.seed, "batch": a.batch, "stride": a.stride, "fp16": bool(model.half),
            "device": str(model.device), "torch": torch.__version__, "monitors": sorted(monitors),
            "classifier": a.classifier}
    t0 = time.time()
    if a.method in ("snap", "psnap"):
        sig = None if a.tau is not None else a.sigma
        cyc = (SNAPCycle(model, a.tau, sig, a.stride) if a.method == "snap"
               else PSNAPCycle(model, a.tau, sig, a.gamma, a.stride, a.h_rel))
        print(f"[cycle] {cyc}")
        mk = None if a.monitor_cycles == "all" else [int(k) for k in a.monitor_cycles.split(",") if k]
        res = run_cycles(cyc, x, a.cycles, a.batch, gen, save_cycles=save_k + [a.cycles], monitors=monitors,
                         monitor_cycles=mk, log_every=max(1, a.cycles // 20))
        meta.update({"tau": cyc.tau, "sigma": cyc.sigma, "cycles": a.cycles, "calls_per_cycle": cyc.calls})
        if a.method == "psnap":
            meta.update({"gamma": a.gamma, "h_rel": a.h_rel})
    else:
        if a.estimator is None:
            raise SystemExit("--method snaped / psnaped needs --estimator (scripts/train_estimator.py)")
        est = load_estimator(a.estimator, model)
        tmin = a.tau_min if a.tau_min is not None else model.tau_for_sigma(a.sigma_min, a.stride)
        tmax = a.tau_max if a.tau_max is not None else model.tau_for_sigma(a.sigma_max, a.stride)
        cyc = "psnap" if a.method == "psnaped" else "snap"
        print(f"[{a.method}] {est.kind} estimator (cost {est.cost}) | tau {tmin}-{tmax} (sigma {model.sigma(tmin):.2f}-"
              f"{model.sigma(tmax):.2f}) | at most {a.max_cycles} cycles | stop below the {a.stop_quantile:g} quantile "
              f"of held-out in-domain scores ({est.healthy_threshold(a.stop_quantile):.3f})")
        res = estimator_driven(model, x, est, cyc, tmin, tmax, a.max_cycles, stop_quantile=a.stop_quantile,
                               gamma=a.gamma, h_rel=a.h_rel, stride=a.stride, batch=a.batch, generator=gen,
                               save_cycles=save_k + [a.max_cycles], monitors=monitors, log_every=10)
        meta.update({"tau_min": tmin, "tau_max": tmax, "sigma_min": model.sigma(tmin), "sigma_max": model.sigma(tmax),
                     "max_cycles": a.max_cycles, "stop_quantile": a.stop_quantile, "estimator": a.estimator,
                     "estimator_kind": est.kind, "estimator_meta": est.meta})
        if cyc == "psnap":
            meta.update({"gamma": a.gamma, "h_rel": a.h_rel})
    meta["seconds"] = round(time.time() - t0, 1)
    out = new_run_dir(a.out, a.method + (f"_{a.label}" if a.label else ""))
    save_run(out, res, meta, names)
    extra = ""
    if a.method in ("snaped", "psnaped"):
        extra = f" | {100 * np.mean(res['stopped']):.0f} % stopped, median stop {np.median(res['cycles_run']):.0f} cycles"
    print(f"[done] {len(x)} images in {meta['seconds'] / 60:.1f} min, {np.mean(res['nfe']):.0f} network calls per image"
          f"{extra} -> {out}")
    return out


if __name__ == "__main__":
    main()
