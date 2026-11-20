# -*- coding: utf-8 -*-
"""Smoke test of the paper's figure scripts (paper/fig*.py) on tiny run folders, CPU, in about a minute.

    python tests/test_figures.py [--out test_outputs]     # writes only into a new folder <out>/<date-time>_figtest/

Run folders are made with the real methods (SNAP, P-SNAP, SNAPED on the tiny random U-Net of test_snap.py, with
fake classifier heads read at every cycle) and saved with snap.runs.save_run, plus one synthetic RGB run; then every
figure script is run on them in both styles and must write its PDF and PNG.  This checks the plumbing, not the look:
the look is checked on real runs against the paper.
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "paper"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from snap import DiffusionModel, snap, psnap, snaped, new_run_dir, save_run  # noqa: E402
from snap.estimators import FunctionEstimator  # noqa: E402
from snap.model import linear_schedule  # noqa: E402

HEADS = ("Edema", "Effusion", "Cardiomegaly")


def head(j):
    """A fake classifier head: a probability that depends on the image."""
    return lambda x: torch.sigmoid(8 * x.float().mean((1, 2, 3)) + j).cpu().numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="test_outputs")
    a = p.parse_args(argv)
    torch.set_grad_enabled(False)
    import test_snap
    out = new_run_dir(a.out, "figtest")
    runs_dir = os.path.join(out, "runs")
    os.mkdir(runs_dir)
    m = DiffusionModel(test_snap.tiny_net(), linear_schedule(), half=False)
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(6, 1, 16, 16, generator=g) * 2 - 1) * 0.8
    names = [f"{i:05d}" for i in range(6)]
    mon = {f"p_{h}": head(j) for j, h in enumerate(HEADS)}
    K, keep = 6, (1, 2, 3, 4, 6)

    def store(res, meta, label):
        return save_run(new_run_dir(runs_dir, label), res, meta, names)

    R = {}
    R["snap"] = store(snap(m, x, sigma=1.24, n_cycles=K, batch=4, generator=torch.Generator().manual_seed(1),
                           save_cycles=keep, monitors=mon), {"method": "snap", "sigma": 1.24}, "snap")
    R["psnap"] = store(psnap(m, x, sigma=1.24, n_cycles=K, gamma=4.67, batch=4, generator=torch.Generator().manual_seed(1),
                             save_cycles=keep, monitors=mon), {"method": "psnap", "sigma": 1.24, "gamma": 4.67}, "psnap")
    est = FunctionEstimator(lambda z: z.float().mean((1, 2, 3)).cpu().numpy(),
                            calibration={"healthy_scores": [float(v) for v in np.linspace(-0.2, 0.2, 9)],
                                         "source_median": 0.6})
    R["snaped"] = store(snaped(m, x, est, tau_min=100, tau_max=400, max_cycles=K, batch=4,
                               generator=torch.Generator().manual_seed(2), save_cycles=keep, monitors=mon),
                        {"method": "snaped", "sigma_min": 0.34, "sigma_max": 2.04}, "snaped")
    rng = np.random.default_rng(0)                       # an RGB run (format only)
    x0 = rng.integers(0, 256, (6, 3, 20, 20), dtype=np.uint8)
    res = {"x": torch.from_numpy(x0.astype(np.float32) / 127.5 - 1), "x0": x0,
           "frames": {k: rng.integers(0, 256, x0.shape, dtype=np.uint8) for k in keep},
           "mse": rng.uniform(1e-3, 1e-2, (K + 1, 6)).astype(np.float32), "nfe": np.full(6, 1.0)}
    R["rgb"] = store(res, {"method": "psnap", "sigma": 1.24, "gamma": 3.0}, "rgb")
    res["frames"] = {k: rng.integers(0, 256, x0.shape, dtype=np.uint8) for k in keep}
    R["rgb_snap"] = store(res, {"method": "snap", "sigma": 1.24}, "rgb_snap")
    cal = os.path.join(out, "calibration.json")
    with open(cal, "x") as f:
        json.dump({"heads": {h: {"threshold": 0.5} for h in HEADS}}, f)

    import fig1_teaser
    import fig3_drift_examples
    import fig4_drift_mse
    import fig5_snap_vs_snaped
    import fig7_paintings
    figs = os.path.join(out, "figures")
    pairs = ",".join(f"{h}={{}}" for h in HEADS)
    calls = []
    for style in ("da", "printed"):
        calls += [
            (fig1_teaser, ["--runs", R["psnap"] + "," + R["rgb"], "--images", "0,00001", "--cycles", "0,1,2,4;0,1,3,6",
                           "--style", style]),
            (fig3_drift_examples, ["--snap", R["snap"] + "," + R["rgb_snap"], "--psnap", R["psnap"] + "," + R["rgb"],
                                   "--images", "2,1", "--cycles", "0,1,2,4,6", "--style", style]),
            (fig5_snap_vs_snaped, ["--images", "Edema:0,Effusion:3,Cardiomegaly:5", "--snap", pairs.format(*[R["snap"]] * 3),
                                   "--snaped", pairs.format(*[R["snaped"]] * 3), "--calibration", cal,
                                   "--cycles", "1,2,4,6", "--style", style]),
            (fig7_paintings, ["--run", R["rgb"], "--images", "0,2,4", "--cycles", "0,1,2,3,4", "--style", style]),
        ]
    calls += [(fig7_paintings, ["--run", R["rgb"], "--cycles", "0,1,2,3,4", "--images", "0", "--cycle_titles"]),
              (fig4_drift_mse, ["--snap", R["snap"] + ";" + R["snap"] + "," + R["rgb_snap"],
                                "--psnap", R["psnap"] + ";" + R["psnap"] + "," + R["rgb"], "--gain_label", "4.67",
                                "--n_boot", "200"]),
              (fig4_drift_mse, ["--snap", R["snap"], "--psnap", R["psnap"], "--titles", "Chest radiographs",
                                "--gain_label", "none", "--scales", "loglog", "--n_boot", "200"])]
    failed = 0
    for mod, args in calls:
        d = mod.main(args + ["--out", figs])
        ok = bool(glob.glob(os.path.join(d, "*.pdf"))) and bool(glob.glob(os.path.join(d, "*.png")))
        print(("  ok    " if ok else "  FAIL  ") + f"{mod.__name__} {' '.join(args[-2:])}")
        failed += not ok
    print(f"\n{'all figures written' if not failed else f'{failed} figure(s) FAILED'} (outputs in {out})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
