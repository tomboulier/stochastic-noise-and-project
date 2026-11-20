# -*- coding: utf-8 -*-
"""Train the few-shot estimator of SNAPED and P-SNAPED (paper, Sec. 3.2; Algorithm 2).

    # the paper's estimator: an ensemble of three small CNNs (strong augmentation, 1000 steps)
    python scripts/train_estimator.py --kind cnn \\
        --source data/chestmnist224/cardiomegaly_train.npy \\
        --healthy data/chestmnist224/healthy_train.npy --n_healthy 5000 \\
        --healthy_val data/chestmnist224/healthy_val.npy --n_val 2000 \\
        --test_source data/chestmnist224/cardiomegaly_test.npy --test_healthy data/chestmnist224/healthy_test.npy \\
        --out estimators/cnn_cardiomegaly.pt

    # an alternative not used in the paper: shrinkage LDA on the frozen U-Net's activations
    python scripts/train_estimator.py --kind lda --model ... --config ... --source ... --out estimators/lda.pt

source       images of the source domain (the N few-shot images: the first --n_source, 0 = all)
healthy      images of the model's domain for training (the first --n_healthy)
healthy_val  held-out images of the model's domain (the first --n_val): the stopping threshold s_h is a
             quantile of their scores (without it, the LAST --n_val images of --healthy are held out)
test_source, test_healthy   optional: AUROC reported on them (never used for any choice)
Images are taken in file order: arrays written by scripts/export_chestmnist.py are already shuffled.

Writes the estimator file (never over an existing one) and <out>.json (settings, AUROC, CV grid of the LDA).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from snap import load_model, load_images, first, train_cnn, fit_lda  # noqa: E402
from snap.evaluation import auroc  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=("cnn", "lda"), default="cnn")
    p.add_argument("--source", required=True)
    p.add_argument("--healthy", required=True)
    p.add_argument("--healthy_val", default=None)
    p.add_argument("--n_source", type=int, default=0, help="N (0: all)")
    p.add_argument("--n_healthy", type=int, default=5000, help="training in-domain images (0: all)")
    p.add_argument("--n_val", type=int, default=2000, help="held-out in-domain images for calibration (0: all)")
    p.add_argument("--test_source", default=None)
    p.add_argument("--test_healthy", default=None)
    p.add_argument("--n_test_healthy", type=int, default=2000)
    p.add_argument("--model", default=None, help="diffusion model (needed for --kind lda)")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True, help="estimator file to create (e.g. estimators/cnn_cardiomegaly.pt)")
    g = p.add_argument_group("cnn (paper)")
    g.add_argument("--steps", type=int, default=1000)
    g.add_argument("--seeds", default="0,1,2", help="ensemble members")
    g = p.add_argument_group("lda (not in the paper)")
    g.add_argument("--blocks", default="auto", help="U-Net blocks (down<i>, mid, up<i>), comma-separated")
    g.add_argument("--pools", default="1,2,4")
    g.add_argument("--taus_f", default="0,20,50,100,200", help="timesteps at which activations are read")
    g.add_argument("--n_f", type=int, default=4, help="noise draws averaged when tau_f > 0")
    g.add_argument("--gammas", default="0.01,0.03,0.1,0.3,1,3,10")
    g.add_argument("--folds", type=int, default=5)
    g.add_argument("--seed", type=int, default=0, help="CV folds of the LDA")
    return p.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                    # noqa: BLE001
        pass
    a = parse(argv)
    if os.path.exists(a.out) or os.path.exists(a.out + ".json"):
        raise SystemExit(f"{a.out} already exists: choose another --out (nothing is overwritten)")
    model = None
    if a.kind == "lda" or a.model is not None:
        if a.model is None:
            raise SystemExit("--kind lda reads the diffusion model's activations: pass --model")
        model = load_model(a.model, a.config, a.device)
        print(f"[model] {model}")
    ch = model.channels if model is not None else None
    size = model.image_size if model is not None else None
    xs, _ = first(*load_images(a.source, ch, size), a.n_source)
    xh_all, nh_all = load_images(a.healthy, xs.shape[1], xs.shape[-1])
    if a.healthy_val:
        xv, _ = first(*load_images(a.healthy_val, xs.shape[1], xs.shape[-1]), a.n_val)
        xh, _ = first(xh_all, nh_all, a.n_healthy)
    else:                                                # the last n_val images of --healthy are held out
        nv = a.n_val if a.n_val > 0 else len(xh_all) // 5
        xv, xh = xh_all[-nv:], xh_all[:-nv]
        xh = xh[:a.n_healthy] if a.n_healthy > 0 else xh
    print(f"[data] {len(xs)} source, {len(xh)} in-domain for training, {len(xv)} held out for calibration "
          f"({xs.shape[1]} x {xs.shape[2]} x {xs.shape[3]})")
    t0 = time.time()
    report = {"args": vars(a)}
    if a.kind == "cnn":
        device = model.device if model is not None else a.device
        est = train_cnn(xs, xh, a.steps, [int(s) for s in a.seeds.split(",")], device=device, log_every=250)
    else:
        blocks = None if a.blocks == "auto" else [b for b in a.blocks.split(",") if b]
        est, rows = fit_lda(model, xs, xh, blocks, [int(v) for v in a.pools.split(",")],
                            [int(v) for v in a.taus_f.split(",")], a.n_f, [float(v) for v in a.gammas.split(",")],
                            a.folds, seed=a.seed)
        report["cv_grid"] = rows
        print(f"[lda] selected {est.meta['selected']}")
    est.calibrate(xs, xv)
    est.meta.update({"source": a.source, "healthy": a.healthy, "healthy_val": a.healthy_val})
    report.update({"kind": a.kind, "meta": est.meta, "seconds": round(time.time() - t0, 1),
                   "s_source_median": est.source_level(), "s_healthy_median": est.healthy_threshold(0.5)})
    if a.test_source and a.test_healthy:
        ts, _ = load_images(a.test_source, xs.shape[1], xs.shape[-1])
        th, nth = first(*load_images(a.test_healthy, xs.shape[1], xs.shape[-1]), a.n_test_healthy)
        report["auroc_test"] = auroc(est.score_all(ts), est.score_all(th))
        report["n_test"] = [len(ts), len(th)]
        print(f"[test] AUROC {report['auroc_test']:.3f} ({len(ts)} source, {len(th)} in-domain)")
    est.save(a.out)
    with open(a.out + ".json", "x") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"[done] {a.kind} estimator in {(time.time() - t0) / 60:.1f} min -> {a.out} "
          f"(s_h = {est.healthy_threshold(0.5):.3f}, s_s = {est.source_level():.3f})")


if __name__ == "__main__":
    main()
