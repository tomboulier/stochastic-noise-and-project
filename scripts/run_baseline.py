# -*- coding: utf-8 -*-
"""Run a competing method of the paper (snap/baselines) over a grid of settings: one output per setting and image.

    M="--model weights/chestmnist_224.pt --config configs/chestmnist_224.json"
    D=data/chestmnist224                                        # scripts/export_chestmnist.py --size 224

    # Fig. 6 (ChestMNIST, 224 px): the 64 first test images of each pathology, and the 64 first healthy test images
    python scripts/run_baseline.py --method sdedit   $M --input $D/cardiomegaly_test.npy --n 64 --group Cardiomegaly
    python scripts/run_baseline.py --method anoddpm  --model weights/chestmnist_224_simplex.pt --config configs/chestmnist_224.json \\
        --input $D/cardiomegaly_test.npy --n 64 --group Cardiomegaly
    python scripts/run_baseline.py --method autoddpm $M --input $D/cardiomegaly_test.npy --n 64 --group Cardiomegaly \\
        --healthy $D/healthy_val.npy --n_healthy 32
    python scripts/run_baseline.py --method vae --vae runs/vae/vae_b1.pt,runs/vae/vae_b0.1.pt,runs/vae/vae_b0.01.pt \\
        --input $D/cardiomegaly_test.npy --n 64 --group Cardiomegaly
    (--input $D/healthy_test.npy --group sain_reel for the healthy images; Effusion, Edema likewise.)

    # Table 1 (degraded faces, FFHQ ADM), one level (L = a folder of scripts/degrade_faces.py: 132 images per
    # degradation, the 100 test then the 32 validation photographs).  As in the paper's runs, the four degradations
    # go through ONE generator per setting, in the order blur, JPEG, pixelation, noise (comma-separated files are
    # concatenated in order)
    F="--model weights/ffhq.pt --config configs/ffhq_256_adm.json --recipe faces \\
       --input $L/blur.npz,$L/jpeg.npz,$L/pixelation.npz,$L/noise.npz \\
       --reference $L/clean.npz,$L/clean.npz,$L/clean.npz,$L/clean.npz"
    python scripts/run_baseline.py --method sdedit   $F                     (strong level: --taus 50,100,150,200,300,400,500)
    python scripts/run_baseline.py --method autoddpm $F --healthy $L/clean.npz --healthy_start 100 --n_healthy 32
    # DIIP: test photographs only, one degradation per run; --row_offset 0, 132, 264, 396 for blur, jpeg, pixelation,
    # noise (the row of image i in the paper's concatenated array is row_offset + start + i)
    python scripts/run_baseline.py --method diip --model weights/ffhq.pt --config configs/ffhq_256_adm.json \\
        --recipe faces --input $L/jpeg.npz --degradation jpeg --row_offset 132 --n 100 --reference $L/clean.npz

Defaults are the paper's grids for the chosen --recipe (chestmnist, the settings of Fig. 6: SDEdit tau 50..900, AnoDDPM t 100..400 with
simplex noise, AutoDDPM t_mask 200..600 and t_resample 50; faces: SDEdit 50..400, AnoDDPM 50, 100, 200 with
Gaussian noise, AutoDDPM 100, 200 and 50), and the recipe fixes the per-setting seeds of the paper's runs
(snap/baselines/__init__.py).  Batches of --batch images (16; DIIP 4) go through each setting's generator in file
order: the result depends on (seed, recipe, group, batch, the images and their order, device).

Writes a new folder <out>/<date-time>_<method>/ (snap/runs.py, SettingsRun): run.json and outputs.npz with the inputs
x0, the outputs (S, N, C, H, W) uint8, the labels, the MSE to x0 (S, N) (uint8 images read in [0, 1], float64
computation), the network calls per image (S,), and monitors mon_<name> (S, N, ...): mon_mse_reference (MSE to the
clean images of --reference), mon_p_<head> (--classifier), mon_mask_fraction (AutoDDPM), mon_iterations and
mon_capped (DIIP).  Nothing is overwritten.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from snap import load_model, load_images, first, new_run_dir  # noqa: E402
from snap.images import to_uint8, from_uint8  # noqa: E402
from snap.runs import save_settings_run  # noqa: E402
from snap.baselines import (derived_seed, setting_keys, diip_keys, make_generator, mse_uint8,  # noqa: E402
                            DEGRADATION_KEYS, DIIP_CRITERION, sdedit, anoddpm, autoddpm, autoddpm_threshold,
                            load_vae, reconstruct, DIIP)

METHODS = ("sdedit", "anoddpm", "autoddpm", "vae", "diip")
GRIDS = {"chestmnist": {"sdedit": "50,100,150,200,300,400,600,700,800,900", "anoddpm": "100,200,250,300,400",
                        "autoddpm": "200,300,400,500,600"},                    # the settings plotted in Fig. 6
         "faces": {"sdedit": "50,100,150,200,300,400", "anoddpm": "50,100,200", "autoddpm": "100,200"}}
NOISE = {"chestmnist": "simplex", "faces": "gaussian"}


def ints(s):
    return [int(v) for v in str(s).split(",") if v.strip()]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model", default=None, help="as in scripts/project.py (not used by --method vae)")
    p.add_argument("--config", default=None, help="model JSON (configs/)")
    p.add_argument("--input", required=True, help="folder of images, or a .npy / .npz array of uint8 images "
                                                   "(comma-separated: concatenated in order)")
    p.add_argument("--n", type=int, default=0, help="number of images, from --start in file order (0: all)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--resize", action="store_true", help="resize the inputs to the model's size")
    g = p.add_argument_group("seeds (snap/baselines/__init__.py)")
    g.add_argument("--recipe", choices=("chestmnist", "faces"), default="chestmnist",
                   help="seed keys and default grids of the paper's runs: chestmnist (Fig. 6) or faces (Table 1)")
    g.add_argument("--group", default=None, help="chestmnist recipe: image group of the seed keys (Cardiomegaly, "
                                                "Effusion, Edema, sain_reel for healthy images; default: input file name)")
    g.add_argument("--degradation", choices=tuple(DEGRADATION_KEYS), default=None,
                   help="faces recipe, DIIP: degradation of the input images (seed key and stopping criterion)")
    g.add_argument("--row_offset", type=int, default=0,
                   help="faces recipe, DIIP: row of the first image of --input in the paper's array of all degraded "
                        "photographs (0 when --input is that array; the row of image i is row_offset + start + i)")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("SDEdit, AnoDDPM")
    g.add_argument("--taus", default=None, help="timesteps (default: the paper's grid of the recipe)")
    g.add_argument("--stride", type=int, default=10, help="SDEdit: DDIM step")
    g.add_argument("--noise", choices=("simplex", "gaussian"), default=None,
                   help="AnoDDPM noise (default: simplex for chestmnist, gaussian for faces)")
    g.add_argument("--octaves", type=int, default=6)
    g.add_argument("--persistence", type=float, default=0.8)
    g = p.add_argument_group("AutoDDPM")
    g.add_argument("--mask_taus", default=None, help="masking levels t_mask (default: the paper's grid)")
    g.add_argument("--resample_tau", type=int, default=50)
    g.add_argument("--healthy", default=None, help="in-domain validation images for the threshold")
    g.add_argument("--n_healthy", type=int, default=32, help="number of them, from --healthy_start")
    g.add_argument("--healthy_start", type=int, default=0)
    g.add_argument("--quantile", type=float, default=0.95)
    g.add_argument("--sigma_px", type=float, default=4.0, help="blur of the anomaly map (pixels)")
    g.add_argument("--dilation_px", type=int, default=5)
    g = p.add_argument_group("VAE")
    g.add_argument("--vae", default="", help="comma-separated VAE files (scripts/train_vae.py), one setting each")
    g.add_argument("--latent", type=int, default=None, help="latent size (default: vae.json next to the file, else 256)")
    g = p.add_argument_group("DIIP")
    g.add_argument("--criterion", choices=("loss", "sharpness"), default=None,
                   help="early stopping (default: from --degradation: sharpness for blur and pixelation, else loss)")
    g.add_argument("--k_min", type=int, default=100)
    g.add_argument("--k_max", type=int, default=300)
    g.add_argument("--lr", type=float, default=1.5e-3)
    g.add_argument("--eps", type=float, default=1e-3, help="relative loss decrease of the 'loss' criterion")
    g.add_argument("--dt", type=int, default=100, help="DDIM step of the generator (999, 899, ..., 99)")
    g = p.add_argument_group("measurements (stored in the run; the methods do not use them)")
    g.add_argument("--reference", default=None, help="clean images, same order as --input: MSE stored as mon_mse_reference")
    g.add_argument("--classifier", default=None, help="MedMNIST weight file(s) / folder, or file.py:build")
    g.add_argument("--heads", default="", help="classes read on every output, e.g. Cardiomegaly,Effusion,Edema")
    g = p.add_argument_group("run")
    g.add_argument("--batch", type=int, default=None, help="images per batch (default 16; DIIP 4)")
    g.add_argument("--device", default=None)
    g.add_argument("--fp32", action="store_true", help="full precision (default: fp16 autocast on CUDA)")
    g.add_argument("--label", default="", help="suffix of the output folder")
    g.add_argument("--out", default="outputs")
    a = p.parse_args(argv)
    if a.batch is None:
        a.batch = 4 if a.method == "diip" else 16
    if a.noise is None:
        a.noise = NOISE[a.recipe]
    if a.method != "vae" and not a.model:
        p.error(f"--method {a.method} needs --model")
    if a.method == "vae" and not a.vae:
        p.error("--method vae needs --vae")
    if a.method == "autoddpm" and not a.healthy:
        p.error("--method autoddpm needs --healthy (in-domain validation images for the threshold)")
    if a.method == "diip" and a.recipe == "faces" and a.degradation is None:
        p.error("--method diip with --recipe faces needs --degradation")
    return a


def batches(x, batch):
    for s in range(0, len(x), batch):
        yield x[s:s + batch]


def load_inputs(spec, **kw):
    """load_images on each comma-separated path, concatenated in order -> (x, names)."""
    xs, names = [], []
    for path in [s for s in str(spec).split(",") if s.strip()]:
        x, n = load_images(path.strip(), **kw)
        xs.append(x)
        names += list(n)
    return torch.cat(xs, 0), names


def main(argv=None):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                    # noqa: BLE001
        pass
    a = parse(argv)
    model = None
    if a.method == "vae":
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        x, names = load_inputs(a.input)
    else:
        model = load_model(a.model, a.config, a.device, half=(False if a.fp32 else None))
        device = model.device
        print(f"[model] {model}")
        x, names = load_inputs(a.input, channels=model.channels, size=model.image_size, resize=a.resize)
    x, names = first(x, names, a.n, a.start)
    x0_u8 = to_uint8(x)
    group = a.group or os.path.splitext(os.path.basename(os.path.normpath(a.input.split(",")[0])))[0]
    print(f"[input] {len(x)} images {tuple(x.shape[1:])} from {a.input} (from #{a.start}) | recipe {a.recipe}"
          + (f", group {group!r}" if a.recipe == "chestmnist" else ""))
    seed = a.seed
    outputs, labels, nfe, settings = [], [], [], []
    mons = {}
    t0 = time.time()

    def finish(label, y_u8, calls, info, extra=None):
        outputs.append(y_u8)
        labels.append(label)
        nfe.append(float(calls))
        settings.append(dict(info, label=label, network_calls=float(calls)))
        for k, v in (extra or {}).items():
            mons.setdefault(k, []).append(np.asarray(v, np.float32))
        print(f"[{a.method}] {label} | mean MSE to x0 {mse_uint8(y_u8, x0_u8).mean():.2e} | {calls:.0f} calls per image "
              f"| {time.time() - t0:.0f} s")

    if a.method in ("sdedit", "anoddpm"):
        for tau in ints(a.taus or GRIDS[a.recipe][a.method]):
            keys = setting_keys(a.recipe, a.method, tau, group, noise=a.noise)
            s = derived_seed(seed, *keys)
            gen = make_generator(device, s)
            ys, calls = [], None
            for xb in batches(x, a.batch):
                if a.method == "sdedit":
                    y, calls = sdedit(model, xb.to(device), tau, stride=a.stride, generator=gen)
                else:
                    y, calls = anoddpm(model, xb.to(device), tau, gen, a.noise, a.octaves, a.persistence)
                ys.append(to_uint8(y))
            info = {"tau": tau, "sigma": model.sigma(tau - tau % a.stride if a.method == "sdedit" else tau),
                    "seed": s, "seed_keys": list(keys)}
            if a.method == "anoddpm":
                info.update({"noise": a.noise, "octaves": a.octaves, "persistence": a.persistence})
            finish(f"tau={tau}", np.concatenate(ys), calls, info)
    elif a.method == "autoddpm":
        xh, _ = load_inputs(a.healthy, channels=model.channels, size=model.image_size, resize=a.resize)
        xh, _ = first(xh, list(range(len(xh))), a.n_healthy, a.healthy_start)
        for tm in ints(a.mask_taus or GRIDS[a.recipe]["autoddpm"]):
            keys = setting_keys(a.recipe, "autoddpm", tm, group, a.resample_tau)
            s = derived_seed(seed, *keys)
            gen = make_generator(device, s)
            if a.recipe == "faces":                       # one generator: threshold first, then the images
                tkeys, ts_, gthr = keys, s, gen
            else:
                tkeys = setting_keys(a.recipe, "autoddpm_threshold", tm)
                ts_ = derived_seed(seed, *tkeys)
                gthr = make_generator(device, ts_)
            thr = autoddpm_threshold(model, xh, tm, a.quantile, gthr, a.batch, a.sigma_px)
            print(f"[autoddpm] t_mask {tm}: threshold {thr:.4f} (quantile {a.quantile:g} of the maps of {len(xh)} "
                  f"in-domain images)")
            ys, fr, calls = [], [], None
            for xb in batches(x, a.batch):
                y, calls, msk = autoddpm(model, xb.to(device), tm, a.resample_tau, thr, gen, a.sigma_px, a.dilation_px)
                ys.append(to_uint8(y))
                fr.append(msk.flatten(1).mean(1).cpu().numpy())
            info = {"tau_mask": tm, "tau_resample": a.resample_tau, "threshold": thr, "quantile": a.quantile,
                    "n_healthy": len(xh), "sigma_px": a.sigma_px, "dilation_px": a.dilation_px, "seed": s,
                    "seed_keys": list(keys), "threshold_seed": ts_, "threshold_seed_keys": list(tkeys)}
            finish(f"tau_mask={tm},tau_resample={a.resample_tau}", np.concatenate(ys), calls, info,
                   {"mask_fraction": np.concatenate(fr)})
    elif a.method == "vae":
        for f in [v for v in a.vae.split(",") if v]:
            net = load_vae(f, x.shape[-1], x.shape[1], a.latent, device)
            stem = os.path.splitext(os.path.basename(f))[0]
            label = f"beta={stem[5:]}" if stem.startswith("vae_b") else stem
            y = reconstruct(net, x, a.batch, device)
            finish(label, to_uint8(y), 0, {"vae": f, "latent": net.latent})
    else:                                                 # diip
        crit = a.criterion or DIIP_CRITERION.get(a.degradation, "loss")
        key = DEGRADATION_KEYS[a.degradation] if a.recipe == "faces" else group
        rows = [a.row_offset + a.start + i for i in range(len(x))]
        seeds = [derived_seed(seed, *diip_keys(key, r)) for r in rows]
        z0 = torch.stack([torch.randn(tuple(x.shape[1:]), generator=torch.Generator().manual_seed(s)) for s in seeds])
        dp = DIIP(model, a.dt, 999, a.lr, a.eps, a.k_min, a.k_max)
        print(f"[diip] criterion {crit}, k_min {a.k_min}, k_max {a.k_max}, {dp.n_calls} calls per iteration, "
              f"{dp.n_disabled} internal checkpointing switch(es) turned off")
        y, k_stop, capped = dp.restore(x, crit, z0, batch=a.batch, log=print)
        info = {"criterion": crit, "k_min": a.k_min, "k_max": a.k_max, "lr": a.lr, "eps": a.eps, "dt": a.dt,
                "seed_keys": f"('diip', {key!r}, row) with row = {a.row_offset} + {a.start} + i", "seeds": seeds,
                "capped": int(capped.sum()), "mean_iterations": float(np.mean(k_stop))}
        finish("early_stop", to_uint8(y), float(np.mean(k_stop)) * dp.n_calls, info,
               {"iterations": k_stop, "capped": capped})

    outs = np.stack(outputs)
    mse = np.stack([mse_uint8(o, x0_u8) for o in outs])
    monitors = {k: np.stack(v) for k, v in mons.items()}
    if a.reference:
        ref, _ = load_inputs(a.reference, channels=x.shape[1], size=x.shape[-1], resize=a.resize)
        ref, _ = first(ref, list(range(len(ref))), a.n, a.start)
        ref_u8 = to_uint8(ref)
        monitors["mse_reference"] = np.stack([mse_uint8(o, ref_u8) for o in outs])
    if a.classifier:
        from snap.evaluation import load_classifier
        clf = load_classifier(a.classifier, device)
        heads = [h for h in a.heads.split(",") if h]
        for h in heads or [None]:
            f = clf.head(h) if h else clf.probs
            monitors[f"p_{h}" if h else "p"] = np.stack([np.asarray(f(from_uint8(o))) for o in outs])
    meta = {"method": a.method, "recipe": a.recipe, "group": group, "degradation": a.degradation, "settings": settings,
            "model": a.model, "config": a.config, "model_repr": repr(model) if model is not None else None,
            "input": a.input, "start": a.start, "names": names, "seed": seed, "batch": a.batch,
            "fp16": bool(model.half) if model is not None else False, "device": str(device),
            "torch": torch.__version__, "reference": a.reference, "classifier": a.classifier,
            "monitors": sorted(monitors), "seconds": round(time.time() - t0, 1)}
    out = new_run_dir(a.out, a.method + (f"_{a.label}" if a.label else ""))
    save_settings_run(out, x0_u8, outs, labels, mse, nfe, meta, names, monitors)
    print(f"[done] {len(labels)} setting(s) x {len(x)} images in {meta['seconds'] / 60:.1f} min -> {out}")
    return out


if __name__ == "__main__":
    main()
