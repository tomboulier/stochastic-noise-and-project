# -*- coding: utf-8 -*-
"""Self-checks of the benchmark scripts (Fig. 6, Table 1 and their data preparation), on CPU, in a few minutes.

    python tests/test_benchmark.py [--out test_outputs]      # writes only into a new folder <out>/<date-time>_benchmark/

Tiny random diffusers U-Nets (tests/test_snap.py), synthetic images, a stub classifier (brightness) and a stub LPIPS
(mean squared difference; the real one needs `pip install lpips` and downloads AlexNet weights):

1. snap/degradations.py: deterministic, noise of seed 1000 + index, clean = identity;
2. scripts/prepare_faces.py: colour filter, fixed order (sorted, permutation of seed 0), splits;
3. scripts/degrade_faces.py: names <degradation>-<split>_<i>, the same degradation as snap/degradations.py;
4. Table 1: scripts/project.py (SNAP, P-SNAP) and baseline run folders -> scripts/evaluate_runs.py --reference ->
   paper/table1_degraded_faces.py; the chosen settings and the numbers are recomputed independently;
5. Fig. 6: scripts/calibrate_classifier.py, scripts/project.py (SNAP, P-SNAP, SNAPED, P-SNAPED with the classifier
   as a monitor), baseline run folders -> scripts/evaluate_runs.py --classifier -> paper/fig6_benchmark.py; the pooled
   points are recomputed independently from the saved images.
"""

import argparse
import importlib.util
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from snap.degradations import DEGRADATIONS, degrade  # noqa: E402
from snap.images import new_run_dir, save_npy  # noqa: E402
from snap.runs import Run, load_run, save_settings_run  # noqa: E402

CHECKS = {"failed": 0}


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        CHECKS["failed"] += 1


def script(rel):
    """A script of the repository, imported as a module (its main(argv) is called in-process)."""
    path = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(rel))[0] + "_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stub_lpips():
    """A stand-in for the lpips package: LPIPS(a, b) = mean squared difference, shape (N, 1, 1, 1)."""
    m = types.ModuleType("lpips")

    class LPIPS(torch.nn.Module):
        def __init__(self, net="alex", verbose=False):
            super().__init__()

        def forward(self, a, b):
            return ((a - b) ** 2).mean((1, 2, 3), keepdim=True)
    m.LPIPS = LPIPS
    sys.modules["lpips"] = m


STUB_CLASSIFIER = '''import numpy as np
from snap.evaluation import Classifier


class Brightness(Classifier):
    """p_j = brightness + 0.02 j - 0.1 (a monotone stand-in for the MedMNIST judge)."""

    def probs(self, x):
        b = ((x.float() + 1) / 2).mean(dim=tuple(range(1, x.dim()))).cpu().numpy()
        return np.stack([np.clip(b + 0.02 * j - 0.1, 0, 1) for j in range(14)], 1)


def build(device=None):
    return Brightness()
'''

MODEL_RGB = '''import torch
from snap import DiffusionModel
from snap.model import linear_schedule


def build(device="cpu", half=None):
    from diffusers import UNet2DModel
    torch.manual_seed(0)
    net = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=1, block_out_channels=[16, 32],
                      down_block_types=["DownBlock2D", "AttnDownBlock2D"], up_block_types=["AttnUpBlock2D", "UpBlock2D"],
                      norm_num_groups=8)
    with torch.no_grad():
        net.conv_out.weight.mul_(0.01)
        net.conv_out.bias.mul_(0.01)
    return DiffusionModel(net.eval().requires_grad_(False).to(device), linear_schedule(), half=False)
'''

MODEL_GRAY = '''import sys
sys.path.insert(0, %r)
import test_snap
from snap import DiffusionModel
from snap.model import linear_schedule


def build(device="cpu", half=None):
    return DiffusionModel(test_snap.tiny_net().to(device), linear_schedule(), half=False)
'''


def write(path, text):
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
    return path


def u8_images(rng, n, size, c, lo, hi):
    """Smooth random uint8 images (n, c, size, size) with values roughly in [lo, hi]."""
    base = rng.uniform(lo, hi, (n, c, 4, 4)).astype(np.float32)
    x = torch.nn.functional.interpolate(torch.from_numpy(base), size=(size, size), mode="bilinear", align_corners=False)
    x = x + torch.from_numpy(rng.normal(0, 6, (n, c, size, size)).astype(np.float32))
    return x.clamp(0, 255).round().to(torch.uint8).numpy()


# --------------------------------------------------------------------------- 1 - 3
def test_data(out):
    from PIL import Image
    print("1. degradations")
    rng = np.random.default_rng(0)
    im = u8_images(rng, 1, 64, 3, 40, 220)[0]
    for g in DEGRADATIONS:
        a, b = degrade(im, g, {"blur": 3, "jpeg": 8, "pixelation": 8, "noise": 0.1}[g], 5), \
            degrade(im, g, {"blur": 3, "jpeg": 8, "pixelation": 8, "noise": 0.1}[g], 5)
        check(a.shape == im.shape and a.dtype == np.uint8 and np.array_equal(a, b) and not np.array_equal(a, im),
              f"{g}: uint8 (3, 64, 64), deterministic, changes the image")
    x = im.transpose(1, 2, 0).astype(np.float32) / 255 + np.random.default_rng(1005).normal(0, 0.1, (64, 64, 3)).astype(np.float32)
    check(np.array_equal(degrade(im, "noise", 0.1, 5), (np.clip(x, 0, 1) * 255).round().astype(np.uint8).transpose(2, 0, 1)),
          "noise of photograph 5 = numpy default_rng(1005), clipped to [0, 1]")
    check(np.array_equal(degrade(im, "clean", 0, 0), im), "clean: identity")
    pix = degrade(im, "pixelation", 8, 0)
    check(bool((pix[:, :8, :8] == pix[:, :1, :1]).all()), "pixelation x8: constant 8 x 8 blocks")

    print("2. prepare_faces")
    src = os.path.join(out, "faces_src")
    os.mkdir(src)
    colour = u8_images(rng, 14, 40, 3, 20, 235)
    for i, a in enumerate(colour):
        Image.fromarray(np.ascontiguousarray(a.transpose(1, 2, 0)[:, :30])).save(os.path.join(src, f"c{i:02d}.png"))
    for i in range(3):                                                    # black-and-white photographs
        g = u8_images(rng, 1, 40, 1, 20, 235)[0, 0]
        Image.fromarray(np.stack([g, g, g], 2)).save(os.path.join(src, f"g{i:02d}.png"))
    prep = script("scripts/prepare_faces.py").main(["--input", src, "--name", "photo", "--color_only", "1", "--size", "32",
                                                    "--splits", "test=4,val=3,train=-1", "--out", os.path.join(out, "faces")])
    z = {s: np.load(os.path.join(prep, f"photo_{s}.npz")) for s in ("test", "val", "train")}
    order = [f"c{i:02d}" for i in np.random.default_rng(0).permutation(14)]
    got = [str(n) for s in ("test", "val", "train") for n in z[s]["names"]]
    check([len(z[s]["images"]) for s in ("test", "val", "train")] == [4, 3, 7] and z["test"]["images"].shape[1:] == (3, 32, 32),
          "splits 4 / 3 / 7 of 32 x 32 colour images, the 3 black-and-white ones excluded")
    check(got == order, "order: files sorted by name, then permutation of seed 0")

    print("3. degrade_faces")
    deg = script("scripts/degrade_faces.py").main([
        "--test", f"{prep}/photo_test.npz:4,{prep}/photo_train.npz:2", "--val", f"{prep}/photo_val.npz:3",
        "--level", "moderate", "--out", os.path.join(out, "degraded")])
    clean = np.load(os.path.join(deg, "clean.npz"))
    blur = np.load(os.path.join(deg, "blur.npz"))
    noise = np.load(os.path.join(deg, "noise.npz"))
    check(list(clean["names"]) == [f"test_{i:03d}" for i in range(6)] + [f"val_{i:03d}" for i in range(3)]
          and list(blur["names"])[0] == "blur-test_000", "names test_000 ... val_002, blur-test_000 ...")
    check(np.array_equal(noise["images"][7], degrade(clean["images"][7], "noise", 0.1, 7))
          and np.array_equal(clean["images"][4], z["train"]["images"][0]), "photograph 7: noise seed 1007; test = test + train")
    deg_s = script("scripts/degrade_faces.py").main([
        "--test", f"{prep}/photo_test.npz:4,{prep}/photo_train.npz:2", "--val", f"{prep}/photo_val.npz:3",
        "--level", "strong", "--degradations", "blur,noise", "--out", os.path.join(out, "degraded")])
    return deg, deg_s


# --------------------------------------------------------------------------- 4
def baseline_runs(root, inp, model, tag, test_only_diip=True):
    """SDEdit-like (one SNAP cycle at two levels), AutoDDPM-like and DIIP-like (test photographs only, in two parts)."""
    from snap.images import from_uint8, load_uint8, to_uint8
    from snap.methods import SNAPCycle
    a, names = load_uint8(inp)
    x = from_uint8(a)
    g = torch.Generator().manual_seed(0)
    outs = [to_uint8(SNAPCycle(model, tau=t)(x, g)) for t in (50, 100)]
    mse = np.stack([((o / 255.0 - a / 255.0) ** 2).reshape(len(a), -1).mean(1) for o in outs])
    save_settings_run(new_run_dir(root, f"sdedit_{tag}"), a, np.stack(outs), ["tau50", "tau100"], mse, [6, 11],
                      {"method": "sdedit", "input": inp}, names)
    blurred = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
    o = to_uint8(0.5 * (x + blurred))[None]
    save_settings_run(new_run_dir(root, f"autoddpm_{tag}"), a, o, ["tm100_tr50"], np.zeros((1, len(a))), [17],
                      {"method": "autoddpm", "input": inp}, names)
    te = [i for i, n in enumerate(names) if "-test_" in n]
    for part in (te[:3], te[3:]):
        o = to_uint8(SNAPCycle(model, tau=30)(x[part], g))[None]
        save_settings_run(new_run_dir(root, f"diip_{tag}"), a[part], o, ["stop"], np.zeros((1, len(part))), [1000],
                          {"method": "diip", "input": inp}, [names[i] for i in part])


def test_table1(out, deg, deg_s):
    print("4. Table 1")
    stub_lpips()
    model_py = write(os.path.join(out, "model_rgb.py"), MODEL_RGB)
    project = script("scripts/project.py")
    from snap import load_model
    model = load_model(model_py + ":build", device="cpu", half=False)
    roots = {}
    for level, d, degs, sigmas in (("moderate", deg, DEGRADATIONS, (0.34, 0.72)), ("strong", deg_s, ("blur", "noise"), (0.34,))):
        root = os.path.join(out, f"faces_{level}")
        roots[level] = root
        for g in degs:
            inp = os.path.join(d, f"{g}.npz")
            for s in sigmas:
                common = ["--model", model_py + ":build", "--input", inp, "--sigma", str(s), "--cycles", "3",
                          "--save_cycles", "1,2,3", "--batch", "16", "--device", "cpu", "--out", root]
                project.main(common + ["--method", "snap"])
                project.main(common + ["--method", "psnap", "--gamma", "3", "--h_rel", "0.003"])
            baseline_runs(root, inp, model, g)
    ev = script("scripts/evaluate_runs.py")
    for level, d in (("moderate", deg), ("strong", deg_s)):
        ev.main(["--runs", os.path.join(roots[level], "*"), "--reference", os.path.join(d, "clean.npz"), "--name", "faces",
                 "--device", "cpu"])
    tab = script("paper/table1_degraded_faces.py").main([
        "--level", f"moderate={os.path.join(roots['moderate'], '*')}", "--level", f"strong={os.path.join(roots['strong'], '*')}",
        "--n_boot", "200", "--out", os.path.join(out, "paper")])
    tex = open(os.path.join(tab, "table1.tex"), encoding="utf-8").read()
    chosen = json.load(open(os.path.join(tab, "settings.json"), encoding="utf-8"))
    check("\\emph{moderate level}" in tex and "\\emph{strong level}" in tex and "DIIP~\\cite{diip}" in tex
          and "with $\\gamma = 3$" in tex and "blur of radius 3 pixels" in tex and "6 test photographs" in tex,
          "table1.tex: two levels, DIIP row, gamma and degradation parameters in the caption")
    # independent recomputation: P-SNAP, moderate, blur
    vals = {}
    for d in sorted(os.listdir(roots["moderate"])):
        run = load_run(os.path.join(roots["moderate"], d))
        if isinstance(run, Run) and run.method == "psnap" and run.names[0].startswith("blur-"):
            m = np.load(os.path.join(run.path, "measures_faces.npz"))
            for i, k in enumerate(m["cycles"]):
                if k >= 1:
                    vals[(run.meta["tau"], int(k))] = (m["lpips_ref"][i], m["mse_ref"][i], run.names)
    names = next(iter(vals.values()))[2]
    va = np.array(["-val_" in n for n in names])
    te = ~va
    best = min(sorted(vals), key=lambda k: float(np.mean(vals[k][0][va])))
    check(chosen["moderate|blur|psnap"] == f"tau{best[0]}_k{best[1]}",
          f"P-SNAP blur: setting {chosen['moderate|blur|psnap']} = lowest mean validation LPIPS")
    import csv
    rows = list(csv.DictReader(open(os.path.join(tab, "table1.csv"), encoding="utf-8")))
    r = next(r for r in rows if r["level"] == "moderate" and r["degradation"] == "blur" and r["method"] == "psnap")
    check(abs(float(r["lpips"]) - float(np.mean(vals[best][0][te]))) < 1e-9
          and abs(float(r["mse"]) - float(np.mean(vals[best][1][te]))) < 1e-12 and r["n_test"] == "6",
          "P-SNAP blur: LPIPS and MSE = test means at that setting")
    rd = next(r for r in rows if r["level"] == "moderate" and r["degradation"] == "noise" and r["method"] == "diip")
    check(rd["n_test"] == "6", "DIIP in two parts with test photographs only: merged, chosen on the test photographs")
    clean = np.load(os.path.join(deg, "clean.npz"))["images"]
    nz = np.load(os.path.join(deg, "noise.npz"))["images"]
    e = ((nz[:6] / 255.0 - clean[:6] / 255.0) ** 2).reshape(6, -1).mean(1).mean()
    ri = next(r for r in rows if r["level"] == "moderate" and r["degradation"] == "noise" and r["method"] == "input")
    check(abs(float(ri["mse"]) - e) < 1e-12, "Degraded input: MSE of the degraded test photographs to the clean ones")
    body = [l for l in tex.splitlines() if l.strip().startswith(("SNAP", "P-SNAP", "SDEdit", "AutoDDPM", "DIIP"))]
    check(len(body) == 10 and all(l.count("&") == 8 for l in body), "LaTeX rows: 5 methods x 2 levels, 4 x (LPIPS, MSE)")


# --------------------------------------------------------------------------- 5
def test_fig6(out):
    print("5. Fig. 6")
    rng = np.random.default_rng(3)
    clf_py = write(os.path.join(out, "stub_classifier.py"), STUB_CLASSIFIER)
    model_py = write(os.path.join(out, "model_gray.py"), MODEL_GRAY % os.path.join(ROOT, "tests"))
    data = os.path.join(out, "chest")
    os.mkdir(data)
    paths = ("Cardiomegaly", "Effusion", "Edema")
    def chest(n, shift):                                                # overlapping brightness: mixed detections
        a = u8_images(rng, n, 16, 1, 40, 160)[:, 0].astype(np.float32) + rng.normal(shift, 30, (n, 1, 1))
        return a.clip(0, 255).round().astype(np.uint8)
    save_npy(chest(40, 0), os.path.join(data, "healthy_test.npy"))
    save_npy(chest(40, 0), os.path.join(data, "healthy_train.npy"))
    for j, P in enumerate(paths):
        save_npy(chest(8, 30 + 5 * j), os.path.join(data, f"{P.lower()}_test.npy"))
        save_npy(chest(30, 30 + 5 * j), os.path.join(data, f"{P.lower()}_train.npy"))
    cal = os.path.join(out, "calibration.json")
    script("scripts/calibrate_classifier.py").main([
        "--classifier", clf_py + ":build", "--negatives", os.path.join(data, "healthy_test.npy"), "--positives",
        ",".join(f"{P}={os.path.join(data, P.lower() + '_test.npy')}" for P in paths), "--device", "cpu", "--out", cal])
    from snap import load_model
    from snap.estimators import train_cnn
    from snap.images import load_images
    model = load_model(model_py + ":build", device="cpu", half=False)
    project = script("scripts/project.py")
    runs = {k: {} for k in ("snap", "psnap", "snaped", "psnaped")}
    base = {}
    for P in paths:
        inp = os.path.join(data, f"{P.lower()}_test.npy")
        root = os.path.join(out, "fig6_runs", P)
        common = ["--model", model_py + ":build", "--input", inp, "--n", "6", "--batch", "4", "--device", "cpu",
                  "--classifier", clf_py + ":build", "--heads", P, "--out", root]
        runs["snap"][P] = project.main(common + ["--method", "snap", "--sigma", "0.52", "--cycles", "4", "--save_cycles", "0,1,2,3,4"])
        runs["psnap"][P] = project.main(common + ["--method", "psnap", "--sigma", "0.52", "--gamma", "4.67", "--cycles", "4",
                                                  "--save_cycles", "0,1,2,3,4"])
        src, _ = load_images(os.path.join(data, f"{P.lower()}_train.npy"))
        hl, _ = load_images(os.path.join(data, "healthy_train.npy"))
        with torch.enable_grad():
            est = train_cnn(src[:20], hl[:30], steps=30, seeds=(0,), widths=(8, 8, 16, 16), device="cpu")
        est.calibrate(src[:20], hl[30:])
        ef = est.save(os.path.join(out, f"cnn_{P}.pt"))
        for m in ("snaped", "psnaped"):
            runs[m][P] = project.main(common + ["--method", m, "--estimator", ef, "--max_cycles", "6",
                                                 "--sigma_min", "0.34", "--sigma_max", "0.72", "--save_cycles", "0,6"])
        a = np.load(inp)[:6, None]
        x = torch.from_numpy(a.astype(np.float32) / 127.5 - 1)
        items = []
        for fam, meta, labels, f in (("sdedit", {"method": "sdedit"}, ["tau50", "tau100"], lambda t: x * (1 - t / 400)),
                                     ("anoddpm", {"method": "anoddpm", "noise": "simplex"}, ["t100", "t200"], lambda t: x - t / 500),
                                     ("anoddpm", {"method": "anoddpm", "noise": "gaussian"}, ["t100"], lambda t: x - 0.1),
                                     ("autoddpm", {"method": "autoddpm"}, ["tm200_tr50"], lambda t: x * 0.9),
                                     ("vae", {"method": "vae"}, ["b0.1", "b1"], None)):
            ts = [int("".join(c for c in lab.split("_")[0] if c.isdigit()) or 1) for lab in labels]
            o = [((f(t) if f else x * 0.5 + float(t) / 10).clamp(-1, 1) + 1) * 127.5 for t in ts]
            o = np.stack([v.round().to(torch.uint8).numpy() for v in o])
            d = new_run_dir(os.path.join(out, "fig6_runs", P), fam)
            save_settings_run(d, a, o, labels, np.zeros((len(labels), 6)), np.zeros(len(labels)), dict(meta, input=inp),
                              [f"{i:05d}" for i in range(6)])
            items.append(d)
        base[P] = items
    ev = script("scripts/evaluate_runs.py")
    ev.main(["--runs", os.path.join(out, "fig6_runs", "*", "*"), "--classifier", clf_py + ":build",
             "--heads", ",".join(paths), "--name", "judge", "--device", "cpu"])
    spec = lambda k: ";".join(f"{P}={runs[k][P]}" for P in paths)                     # noqa: E731
    fig6 = script("paper/fig6_benchmark.py")
    args = ["--calibration", cal, "--n", "6", "--smooth", "3", "--out", os.path.join(out, "paper")] + \
        sum([[f"--{k}", spec(k)] for k in runs], []) + ["--baselines", ";".join(f"{P}=" + ",".join(base[P]) for P in paths)]
    d = fig6.main(args)
    pts = json.load(open(os.path.join(d, "fig6_points.json"), encoding="utf-8"))
    thr = {P: v["threshold"] for P, v in json.load(open(cal))["heads"].items()}
    from snap.evaluation import label_index, load_classifier
    from snap.images import from_uint8
    clf = load_classifier(clf_py + ":build")
    dets, mses = [], []
    for P in paths:
        r = load_run(runs["psnap"][P])
        y = r["frames"][list(r["frame_cycles"]).index(3)]
        p = clf.probs(from_uint8(y))[:, label_index(P)]
        dets.append(np.mean(p > thr[P]))
        mses.append(np.mean(((y / 255.0 - r["x0"] / 255.0) ** 2).reshape(len(y), -1).mean(1)))
    q = next(v for v in pts["fixed"]["psnap"] if v["cycle"] == 3)
    check(abs(q["det"] - np.mean(dets)) < 1e-12 and abs(q["mse"] - np.mean(mses)) < 1e-12,
          f"P-SNAP cycle 3: pooled detection {q['det']:.3f} and MSE {q['mse']:.2e} recomputed from the saved images")
    dd = [np.mean(load_run(runs["snaped"][P]).monitor(f"p_{P}")[:, :6] > thr[P], 1) for P in paths]
    check(np.allclose(pts["driven"]["snaped"]["det"], np.mean(dd, 0)), "SNAPED: detection at every cycle from the monitor")
    check(set(pts["baselines"]) == {"sdedit", "anoddpm_simplex", "autoddpm", "vae"},
          "baselines: AnoDDPM Gaussian left out, the families read in run.json")
    check(abs(pts["healthy_rate"] - np.mean([json.load(open(cal))["heads"][P]["rate_neg"] for P in paths])) < 1e-12,
          "dotted line: mean healthy detection rate of the calibration")
    check(os.path.exists(os.path.join(d, "fig6_benchmark.png")) and os.path.exists(os.path.join(d, "fig6_benchmark.pdf")),
          "fig6_benchmark.pdf / .png written")
    d2 = fig6.main(["--calibration", cal, "--n", "6", "--measures", "none", "--out", os.path.join(out, "paper"),
                    "--snap", spec("snap"), "--psnap", spec("psnap")])
    p2 = json.load(open(os.path.join(d2, "fig6_points.json"), encoding="utf-8"))
    check(len(p2["fixed"]["psnap"]) == 5, "without measures: the classifier read during the run (monitors)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="test_outputs")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    out = new_run_dir(a.out, "benchmark")
    deg, deg_s = test_data(out)
    test_table1(out, deg, deg_s)
    test_fig6(out)
    print(f"\n{'all checks passed' if not CHECKS['failed'] else str(CHECKS['failed']) + ' check(s) FAILED'} (outputs in {out})")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
