# -*- coding: utf-8 -*-
"""Post-hoc measurements on run folders (scripts/project.py, scripts/run_baseline.py), on the SAVED uint8 images.

For a run of SNAP / P-SNAP / SNAPED / P-SNAPED (trajectory.npz) the points are the input (cycle 0) and every
cycle of --save_cycles; for a baseline (outputs.npz) the points are its settings.  At every point and for every
image it computes:

    mse_x0                  MSE to the run's input x0 (uint8 / 255, mean over pixels and channels)
    p_<head>                with --classifier: probability of the class (snap.evaluation; the paper's judge is the
                            MedMNIST ResNet-18 of the 224 branch, read on the 28 x 28 image)
    lpips_ref, mse_ref      with --reference: LPIPS (AlexNet) and MSE to the CLEAN image of the same name (degraded
                            photographs: scripts/degrade_faces.py writes <degradation>-<clean name>)

and the same for the input x0 (keys ending in _x0).  It writes a NEW file <run>/measures_<name>.npz
(points: cycles (P,) and labels (P,), names (N,), the arrays above of shape (P, N), info: a JSON string); a run
that already has this file is skipped (choose another --name).

    # Fig. 6: the classifier on the fixed-level runs and the baselines of each pathology
    python scripts/evaluate_runs.py --runs "outputs/chest/*" --classifier weights/medmnist \\
        --heads Cardiomegaly,Effusion,Edema --name judge
    # Table 1: LPIPS and MSE to the clean photographs
    python scripts/evaluate_runs.py --runs "outputs/faces_moderate/*" --reference data/degraded/<date-time>_moderate/clean.npz \\
        --name faces

(The estimator-driven runs need no measures file for Fig. 6: their curves use the classifier read at EVERY cycle
during the run, project.py --classifier --heads, and their MSE at every cycle.)
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from snap.images import from_uint8, load_uint8  # noqa: E402
from snap.runs import Run, load_run  # noqa: E402


def expand(spec):
    """'a,b,outputs/*_snap' -> run folders (a glob pattern gives the folders with a run.json, sorted)."""
    out = []
    for item in [s for s in spec.split(",") if s.strip()]:
        if any(c in item for c in "*?["):
            out += sorted(d for d in glob.glob(item) if os.path.exists(os.path.join(d, "run.json")))
        else:
            out.append(item)
    return out


def points(run):
    """-> (cycles (P,) int, -1 for settings; labels (P,) str; images (P, N, C, H, W) uint8; x0 (N, C, H, W))."""
    x0 = run["x0"]
    if isinstance(run, Run):
        ks = [int(k) for k in run["frame_cycles"]]
        cyc = sorted(set(ks) | {0})
        imgs = np.stack([x0 if k == 0 else run["frames"][ks.index(k)] for k in cyc])
        return np.array(cyc, np.int32), np.array([str(k) for k in cyc]), imgs, x0
    return np.full(len(run.labels), -1, np.int32), np.array(run.labels), run["outputs"], x0


def mse_u8(y, x):
    """Per-image MSE between uint8 images, in [0, 1] (float64)."""
    return ((np.asarray(y, np.float64) / 255.0 - np.asarray(x, np.float64) / 255.0) ** 2).reshape(len(y), -1).mean(1)


class Reference:
    """Clean images by name; an output named '<degradation>-<clean name>' is matched to '<clean name>'."""

    def __init__(self, path):
        self.images, names = load_uint8(path)
        self.index = {n: i for i, n in enumerate(names)}
        self.path = path

    def of(self, names):
        idx = []
        for n in names:
            key = n if n in self.index else n.split("-", 1)[-1]
            if key not in self.index:
                raise SystemExit(f"no clean image for {n!r} in {self.path}")
            idx.append(self.index[key])
        return self.images[idx]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="run folders, comma-separated; glob patterns allowed")
    p.add_argument("--name", required=True, help="the file written is <run>/measures_<name>.npz")
    p.add_argument("--classifier", default=None, help="MedMNIST weight file(s) / folder, or file.py:build")
    p.add_argument("--heads", default="", help="classes, e.g. Cardiomegaly,Effusion,Edema")
    p.add_argument("--reference", default=None, help="clean images (.npz / .npy / folder) for LPIPS and MSE")
    p.add_argument("--lpips_net", default="alex")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                    # noqa: BLE001
        pass
    runs = expand(a.runs)
    if not runs:
        raise SystemExit(f"no run folder in {a.runs}")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    clf, heads = None, [h for h in a.heads.split(",") if h]
    if a.classifier:
        from snap.evaluation import label_index, load_classifier
        clf = load_classifier(a.classifier, device)
        cols = {h: label_index(h, clf.labels) for h in heads}
        if not heads:
            raise SystemExit("--classifier needs --heads")
    ref = Reference(a.reference) if a.reference else None
    if ref is not None:
        from snap.evaluation import lpips_distance
    if clf is None and ref is None:
        print("[evaluate] neither --classifier nor --reference: only the MSE to x0 is measured")
    for d in runs:
        target = os.path.join(d, f"measures_{a.name}.npz")
        if os.path.exists(target):
            print(f"[skip] {target} exists")
            continue
        t0 = time.time()
        run = load_run(d)
        cyc, labels, imgs, x0 = points(run)
        tab = {"cycles": cyc, "labels": labels, "names": np.array(run.names),
               "mse_x0": np.stack([mse_u8(y, x0) for y in imgs])}
        sets = list(imgs) + [x0]                            # the points, then the input
        if clf is not None:
            pr = [clf.probs(from_uint8(y)) for y in sets]
            for h, j in cols.items():
                tab[f"p_{h}"] = np.stack([q[:, j] for q in pr[:-1]]).astype(np.float32)
                tab[f"p_{h}_x0"] = pr[-1][:, j].astype(np.float32)
        if ref is not None:
            c = ref.of(run.names)
            ct = from_uint8(c)
            lp = [lpips_distance(from_uint8(y), ct, net=a.lpips_net, device=device) for y in sets]
            ms = [mse_u8(y, c) for y in sets]
            tab["lpips_ref"], tab["lpips_ref_x0"] = np.stack(lp[:-1]).astype(np.float32), lp[-1].astype(np.float32)
            tab["mse_ref"], tab["mse_ref_x0"] = np.stack(ms[:-1]), ms[-1]
        info = {"run": d, "method": run.method, "kind": "cycles" if isinstance(run, Run) else "settings",
                "classifier": a.classifier, "heads": heads, "reference": a.reference,
                "lpips_net": a.lpips_net if ref is not None else None, "device": str(device)}
        tab["info"] = np.array(json.dumps(info))
        with open(target, "xb") as f:
            np.savez_compressed(f, **tab)
        print(f"[measures] {d}: {len(cyc)} points x {len(run.names)} images ({time.time() - t0:.0f} s) -> {target}")


if __name__ == "__main__":
    main()
