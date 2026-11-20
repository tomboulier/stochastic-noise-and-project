# -*- coding: utf-8 -*-
"""Fig. 6 of the paper: detection rate against the mean MSE to the input x0, three pathologies pooled.

Every point pools the three pathologies (Cardiomegaly, Effusion, Edema; the first 64 test images of each, single
label): at the same setting, detection rate and mean MSE are averaged over the pathologies, each pathology read by
its own head of the classifier at its own threshold (Youden's index on the real test images,
scripts/calibrate_classifier.py).  An image is detected when p > threshold.

    SNAP, P-SNAP          fixed level: one curve along the cycles, one point per KEPT cycle (--save_cycles; cycle 0 =
                          the input), read on the saved uint8 images; MSE between uint8 images in [0, 1]
    SNAPED, P-SNAPED      estimator-driven: the classifier read at EVERY cycle during the run (monitor) and the MSE at
                          every cycle; an image keeps its stopped state; the pooled curve is smoothed by a centred
                          moving average over --smooth (15) cycles (edge padding), on both coordinates
    baselines             one mark per setting (SDEdit, AnoDDPM simplex, AutoDDPM, VAE), on the uint8 outputs;
                          a setting is drawn when all pathologies have it; AnoDDPM Gaussian is left out (--without)
    dotted line           mean over the pathologies of the detection rate of ALL healthy test images (calibration)

Runs needed, per pathology P (ChestMNIST 224 px, the healthy DDPM, the first 64 images of <p>_test.npy):

    M="--model weights/chestmnist_224.pt --config configs/chestmnist_224.json --input data/chestmnist224/<p>_test.npy --n 64"
    C="--classifier weights/medmnist --heads P"
    python scripts/project.py $M --method snap  --sigma 1.24 --cycles 600 --out outputs/fig6/P     # default --save_cycles
    python scripts/project.py $M --method psnap --sigma 1.24 --gamma 4.67 --cycles 600 --out outputs/fig6/P
    python scripts/project.py $M --method snaped  --estimator estimators/cnn_<p>.pt $C --out outputs/fig6/P
    python scripts/project.py $M --method psnaped --estimator estimators/cnn_<p>.pt --gamma 4.67 $C --out outputs/fig6/P
        (estimator: scripts/train_estimator.py --kind cnn on all single-label training images of P; sigma 0.34 - 2.04,
         at most 500 cycles, stop below the median of held-out healthy scores: the defaults)
    python scripts/run_baseline.py ...   on the same 64 images; the settings drawn in the paper's figure:
        sdedit tau 50, 100, 150, 200, 300, 400, 600, 700, 800, 900; anoddpm (simplex noise, its own retrained DDPM)
        t 100, 200, 250, 300, 400; autoddpm masking t 200, 300, 400, 500, 600 with resampling 50; vae beta 0.01, 0.1, 1
    python scripts/calibrate_classifier.py --classifier weights/medmnist --negatives data/chestmnist224/healthy_test.npy \\
        --positives Cardiomegaly=...,Effusion=...,Edema=... --out calibration/medmnist_r18_224_test.json
    python scripts/evaluate_runs.py --runs "outputs/fig6/*/*" --classifier weights/medmnist --heads P --name judge
        (the judge on the saved images of the fixed-level runs and on the baseline outputs)

Then (CPU, seconds):

    python paper/fig6_benchmark.py --calibration calibration/medmnist_r18_224_test.json \\
        --snap "Cardiomegaly=<run>;Effusion=<run>;Edema=<run>" --psnap "..." --snaped "..." --psnaped "..." \\
        --baselines "Cardiomegaly=<run>,<run>,...;Effusion=...;Edema=..."

A baseline's family is read from its run.json (method; AnoDDPM: its noise) or given as family:<run>
(sdedit, anoddpm_simplex, anoddpm_gaussian, autoddpm, vae).  Without a measures_<--measures>.npz file, a run falls
back on the classifier read during the run (project.py --classifier, on the float images) and says so.
Writes a new folder <out>/<date-time>_fig6/: fig6_benchmark.pdf / .png (column width), fig6_points.json and .csv.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from snap import plotting  # noqa: E402
from snap.images import new_run_dir  # noqa: E402
from snap.runs import Run, load_run  # noqa: E402

PATHOLOGIES = ("Cardiomegaly", "Effusion", "Edema")
COLORS = {"psnap": plotting.C_PSNAP, "psnaped": plotting.C_PSNAP, "snap": plotting.C_SNAP, "snaped": plotting.C_SNAP,
          "sdedit": "#b0b0b0", "anoddpm_simplex": "#eb6834", "anoddpm_gaussian": "#eb6834", "autoddpm": "#0f8a7a",
          "vae": plotting.INK}
NAMES = {"sdedit": "SDEdit", "anoddpm_simplex": "AnoDDPM (simplex)", "anoddpm_gaussian": "AnoDDPM (Gaussian)",
         "autoddpm": "AutoDDPM", "vae": "VAE (healthy)"}
STYLE = {"anoddpm_gaussian": dict(ls=(0, (4, 1.6)), marker="s", mfc="white"), "anoddpm_simplex": dict(ls="-", marker="s"),
         "sdedit": dict(ls="-", marker="o", mfc="white"), "autoddpm": dict(ls="-", marker="^"),
         "vae": dict(ls="none", marker="D")}
BASELINES = ("sdedit", "anoddpm_simplex", "anoddpm_gaussian", "autoddpm", "vae")
ALIASES = {"sdedit": "sdedit", "autoddpm": "autoddpm", "vae": "vae", "anoddpm_s": "anoddpm_simplex",
           "anoddpm_simplex": "anoddpm_simplex", "anoddpm_g": "anoddpm_gaussian", "anoddpm_gaussian": "anoddpm_gaussian"}


# --------------------------------------------------------------------------- reading
def parse_spec(text):
    """'P=a,b;Q=c' -> {P: [a, b], Q: [c]}."""
    out = {}
    for block in [b for b in (text or "").split(";") if b.strip()]:
        P, _, items = block.partition("=")
        out.setdefault(P.strip(), []).extend([x.strip() for x in items.split(",") if x.strip()])
    return out


def family_of(item):
    """'family:path' or 'path' -> (family, path); the family of a bare path is read in its run.json."""
    head, sep, tail = item.partition(":")
    if sep and head.lower().replace("-", "_") in ALIASES and not os.path.exists(item):
        return ALIASES[head.lower().replace("-", "_")], tail
    with open(os.path.join(item, "run.json")) as f:
        meta = json.load(f)
    m = str(meta.get("method", "")).lower().replace("-", "_")
    if m in ALIASES:
        return ALIASES[m], item
    if m.startswith("anoddpm"):                                  # the noise: run.json, or each setting's
        noise = " ".join([str(meta.get(k, "")) for k in ("noise", "noise_type")]
                         + [str(s.get("noise", "")) for s in meta.get("settings", []) if isinstance(s, dict)]).lower()
        if "simplex" in m or "simplex" in noise:
            return "anoddpm_simplex", item
        if "gauss" in m or "gauss" in noise:
            return "anoddpm_gaussian", item
    raise SystemExit(f"{item}: cannot tell the baseline family from run.json (method {m!r}); "
                     f"give it as <family>:{item} ({', '.join(BASELINES)})")


def read_measures(run, name):
    f = os.path.join(run.path, f"measures_{name}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=False)
    m = {k: z[k] for k in z.files}
    if [str(x) for x in m["names"]] != run.names:
        raise SystemExit(f"{f}: its images are not those of the run")
    return m


def head_monitor(run, P):
    """The classifier read during the run for head P: mon_p_<P> (M, N, ...) or column P of mon_p (M, N, 14)."""
    if f"mon_p_{P}" in run:
        return np.asarray(run[f"mon_p_{P}"], float)
    if "mon_p" in run:
        from snap.evaluation import label_index
        return np.asarray(run["mon_p"], float)[..., label_index(P)]
    return None


def mse_u8(y, x):
    return ((np.asarray(y, np.float64) / 255.0 - np.asarray(x, np.float64) / 255.0) ** 2).reshape(len(y), -1).mean(1)


def fixed_points(path, P, thr, a):
    """Fixed-level run -> {cycle: (detection rate, mean MSE)} on the first a.n images, and the run."""
    run = load_run(path)
    if not isinstance(run, Run):
        raise SystemExit(f"{path}: not a run of scripts/project.py")
    n = min(a.n, run.n)
    m = read_measures(run, a.measures)
    if m is not None and f"p_{P}" in m:
        ks, p = [int(k) for k in m["cycles"]], np.asarray(m[f"p_{P}"], float)
        mse = np.asarray(m["mse_x0"], float)
    else:
        mon = head_monitor(run, P)
        if mon is None:
            raise SystemExit(f"{path}: no measures_{a.measures}.npz with p_{P} and no classifier monitor "
                             "(run scripts/evaluate_runs.py)")
        mc = [int(k) for k in run["monitor_cycles"]]
        fc = [int(k) for k in run["frame_cycles"]]
        ks = [k for k in sorted(set(fc) | {0}) if k in mc]
        p = np.stack([mon[mc.index(k)] for k in ks])
        x0 = run["x0"]
        mse = np.stack([mse_u8(x0 if k == 0 else run["frames"][fc.index(k)], x0) for k in ks])
        print(f"[fig6] {path}: no measures_{a.measures}.npz, classifier read during the run (float images)")
    return {k: (float(np.mean(p[i, :n] > thr)), float(np.mean(mse[i, :n]))) for i, k in enumerate(ks)}, run


def driven_curve(path, P, thr, a):
    """Estimator-driven run -> (detection rate (K+1,), mean MSE (K+1,)) at every cycle, first a.n images."""
    run = load_run(path)
    mon = head_monitor(run, P)
    if mon is None or len(mon) != len(run["mse"]):
        raise SystemExit(f"{path}: the classifier must be read at every cycle (project.py --classifier --heads {P})")
    n = min(a.n, run.n)
    return np.mean(mon[:, :n] > thr, 1), np.mean(np.asarray(run["mse"], float)[:, :n], 1), run


def baseline_points(path, P, thr, a):
    """Baseline run -> {setting label: (detection rate, mean MSE)}, first a.n images."""
    run = load_run(path)
    if isinstance(run, Run):
        raise SystemExit(f"{path}: not a baseline run (outputs.npz)")
    n = min(a.n, len(run.names))
    m = read_measures(run, a.measures)
    if m is not None and f"p_{P}" in m:
        labels, p, mse = [str(s) for s in m["labels"]], np.asarray(m[f"p_{P}"], float), np.asarray(m["mse_x0"], float)
    else:
        mon = head_monitor(run, P)
        if mon is None:
            raise SystemExit(f"{path}: no measures_{a.measures}.npz with p_{P} (run scripts/evaluate_runs.py)")
        labels, p = run.labels, mon
        mse = np.stack([mse_u8(y, run["x0"]) for y in run["outputs"]])
        print(f"[fig6] {path}: no measures_{a.measures}.npz, classifier stored by the baseline")
    return {s: (float(np.mean(p[i, :n] > thr)), float(np.mean(mse[i, :n]))) for i, s in enumerate(labels)}


def smooth(v, w):
    """Centred moving average over w values, edges padded with the end values."""
    w = max(1, int(w))
    if w == 1:
        return np.asarray(v, float)
    return np.convolve(np.pad(np.asarray(v, float), (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, "valid")


def pooled(per_p, pathos):
    """{P: {key: (det, mse)}} -> [(key, det, mse)] for the keys present for every pathology (mean over pathologies)."""
    keys = [k for k in per_p[pathos[0]] if all(k in per_p[P] for P in pathos)]
    return [(k, float(np.mean([per_p[P][k][0] for P in pathos])), float(np.mean([per_p[P][k][1] for P in pathos])))
            for k in keys]


def one_value(values, what):
    v = sorted({round(float(x), 6) for x in values if x is not None})
    if len(v) > 1:
        print(f"[fig6] warning: different {what} across pathologies: {v}")
    return v[0] if v else None


# --------------------------------------------------------------------------- figure
def draw(a, curves, traj, bases, healthy, sigma, gamma, out):
    plt = plotting.apply()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    fs_ax, fs_gr, fs_lg = 8.0, 7.0, 7.0
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    for meth in ("snap", "psnap"):                               # fixed level, along the cycles
        if meth in curves:
            pts = curves[meth]
            ax.plot([p[2] for p in pts], [p[1] for p in pts], color=COLORS[meth], lw=1.5,
                    ls="-" if meth == "psnap" else (0, (4, 1.6)), alpha=1)
    for m in ("snaped", "psnaped"):                              # estimator-driven, mean trajectories
        if m in traj:
            ax.plot(smooth(traj[m][1], a.smooth), smooth(traj[m][0], a.smooth), color=COLORS[m], lw=1.2,
                    ls=(0, (1, 1.2)), alpha=0.95, zorder=5)
    for m, pts in bases.items():                                 # one mark per setting
        ax.plot([p[2] for p in pts], [p[1] for p in pts], color=COLORS[m], lw=0.9, ms=3.6, **STYLE[m])
    if healthy is not None:
        ax.axhline(healthy, color=plotting.GREY, lw=0.8, ls=":")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
    ax.grid(True, color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=fs_gr, length=2.5)
    ax.set_xlabel("mean MSE to x₀ (×10⁻²)", fontsize=fs_ax)
    ax.set_ylabel("detection rate, 3 pathologies\n(MedMNIST ResNet-18 judge)" if len(a.pathos) == 3 else
                  f"detection rate, {len(a.pathos)} pathologies\n(MedMNIST ResNet-18 judge)", fontsize=fs_ax)
    g_ = f"γ = {gamma:g}" if gamma is not None else ""
    fx_ = "" if g_ else "fixed "                                 # with the gain in the legend: "σ = 1.24"
    s_ = f"{fx_}σ = {sigma:.2f}" if sigma is not None else "fixed σ"
    h = []
    if "psnap" in curves:
        h.append(Line2D([], [], color=COLORS["psnap"], lw=1.5, label="P-SNAP (" + ", ".join(x for x in (g_, s_) if x) + ")"))
    if "snap" in curves:
        h.append(Line2D([], [], color=COLORS["snap"], lw=1.5, ls=(0, (4, 1.6)), label=f"SNAP ({s_})"))
    for m, lab in (("psnaped", "P-SNAPED" + (f" ({g_})" if g_ else "")), ("snaped", "SNAPED")):
        if m in traj:
            h.append(Line2D([], [], color=COLORS[m], ms=9, ls=(0, (1, 1.3)), lw=1.1, label=lab))
    h += [Line2D([], [], color=COLORS[m], lw=0.9, ms=3.6, label=NAMES[m], **STYLE[m]) for m in bases]
    if healthy is not None:
        h.append(Line2D([], [], color=plotting.GREY, lw=0.8, ls=":", label="healthy test images"))
    lg = ax.legend(handles=h, loc="upper right", ncol=1, fontsize=fs_lg, frameon=True, framealpha=0.92,
                   facecolor="white", edgecolor="none", handlelength=2.4, borderpad=0.4, labelspacing=0.35,
                   borderaxespad=0.2)
    lg.set_zorder(10)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 1.0))
    paths = plotting.save(fig, out, a.name)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", required=True, help="JSON of scripts/calibrate_classifier.py")
    p.add_argument("--snap", default="", help="P=run;P=run (fixed level)")
    p.add_argument("--psnap", default="")
    p.add_argument("--snaped", default="", help="P=run;... (estimator-driven, classifier read at every cycle)")
    p.add_argument("--psnaped", default="")
    p.add_argument("--baselines", default="", help="P=run,run,...;... (scripts/run_baseline.py); family:run to force")
    p.add_argument("--without", default="anoddpm_gaussian", help="baseline families left out (paper: AnoDDPM Gaussian)")
    p.add_argument("--pathologies", default=",".join(PATHOLOGIES))
    p.add_argument("--n", type=int, default=64, help="first images of every run (paper: 64)")
    p.add_argument("--measures", default="judge", help="read <run>/measures_<name>.npz (scripts/evaluate_runs.py)")
    p.add_argument("--smooth", type=int, default=15, help="moving average of the estimator-driven curves (cycles)")
    p.add_argument("--gamma", type=float, default=None, help="gain in the legend (default: from the P-SNAP runs)")
    p.add_argument("--sigma", type=float, default=None, help="level in the legend (default: from the fixed-level runs)")
    p.add_argument("--width", type=float, default=plotting.COLUMN_WIDTH)
    p.add_argument("--height", type=float, default=2.9)
    p.add_argument("--name", default="fig6_benchmark", help="file name of the figure")
    p.add_argument("--out", default="paper_outputs")
    a = p.parse_args(argv)
    a.pathos = [x for x in a.pathologies.split(",") if x]
    with open(a.calibration) as f:
        cal = json.load(f)["heads"]
    missing = [P for P in a.pathos if P not in cal]
    if missing:
        raise SystemExit(f"{a.calibration}: no threshold for {missing}")
    thr = {P: float(cal[P]["threshold"]) for P in a.pathos}
    rates = [cal[P].get("rate_neg") for P in a.pathos]
    healthy = float(np.mean(rates)) if all(r is not None for r in rates) else None
    without = {ALIASES.get(x, x) for x in a.without.split(",") if x}

    curves, metas, per = {}, {"sigma": [], "gamma": [], "gamma_ed": []}, {}
    for meth in ("snap", "psnap"):
        spec = parse_spec(getattr(a, meth))
        if not spec:
            continue
        per[meth] = {}
        for P in a.pathos:
            if len(spec.get(P, [])) != 1:
                raise SystemExit(f"--{meth}: one run per pathology expected for {P}")
            per[meth][P], run = fixed_points(spec[P][0], P, thr[P], a)
            metas["sigma"].append(run.meta.get("sigma"))
            if meth == "psnap":
                metas["gamma"].append(run.meta.get("gamma"))
        curves[meth] = sorted(pooled(per[meth], a.pathos), key=lambda t: t[0])
    traj, per_traj = {}, {}
    for meth in ("snaped", "psnaped"):
        spec = parse_spec(getattr(a, meth))
        if not spec:
            continue
        dets, mses = [], []
        for P in a.pathos:
            if len(spec.get(P, [])) != 1:
                raise SystemExit(f"--{meth}: one run per pathology expected for {P}")
            d_, m_, run = driven_curve(spec[P][0], P, thr[P], a)
            dets.append(d_)
            mses.append(m_)
            per_traj.setdefault(meth, {})[P] = {"det": d_.tolist(), "mse": m_.tolist()}
            if meth == "psnaped":
                metas["gamma_ed"].append(run.meta.get("gamma"))
        K = min(len(d) for d in dets)
        traj[meth] = (np.mean([d[:K] for d in dets], 0), np.mean([m[:K] for m in mses], 0))
    bases, per_base = {}, {}
    spec = parse_spec(a.baselines)
    if spec:
        fam = {}
        for P in a.pathos:
            for item in spec.get(P, []):
                f_, path = family_of(item)
                if f_ in without:
                    continue
                d = fam.setdefault(f_, {}).setdefault(P, {})
                for k, v in baseline_points(path, P, thr[P], a).items():
                    if k in d:
                        raise SystemExit(f"{f_} {P}: setting {k} given twice")
                    d[k] = v
        for f_ in BASELINES:
            if f_ in fam and all(P in fam[f_] for P in a.pathos):
                pts = pooled(fam[f_], a.pathos)
                if pts:
                    bases[f_] = sorted(pts, key=lambda t: t[2])
                    per_base[f_] = fam[f_]
            elif f_ in fam:
                print(f"[fig6] {f_}: not given for every pathology, left out")
    if not (curves or traj or bases):
        raise SystemExit("nothing to draw: give --snap / --psnap / --snaped / --psnaped / --baselines")
    sigma = a.sigma if a.sigma is not None else one_value(metas["sigma"], "sigma")
    gamma = a.gamma if a.gamma is not None else one_value(metas["gamma"] or metas["gamma_ed"], "gamma")
    if metas["gamma"] and metas["gamma_ed"] and one_value(metas["gamma_ed"], "gamma") != one_value(metas["gamma"], "gamma"):
        print("[fig6] warning: P-SNAP and P-SNAPED runs have different gains; the legend shows one")
    out = new_run_dir(a.out, "fig6")
    paths = draw(a, curves, traj, bases, healthy, sigma, gamma, out)
    pts = {"pathologies": a.pathos, "thresholds": thr, "healthy_rate": healthy, "n": a.n, "smooth": a.smooth,
           "sigma": sigma, "gamma": gamma,
           "fixed": {m: [{"cycle": int(k), "det": d, "mse": e} for k, d, e in v] for m, v in curves.items()},
           "fixed_per_pathology": {m: {P: {str(k): list(x) for k, x in v[P].items()} for P in v} for m, v in per.items()},
           "driven": {m: {"det": t[0].tolist(), "mse": t[1].tolist(), "det_smoothed": smooth(t[0], a.smooth).tolist(),
                          "mse_smoothed": smooth(t[1], a.smooth).tolist()} for m, t in traj.items()},
           "driven_per_pathology": per_traj,
           "baselines": {m: [{"setting": k, "det": d, "mse": e} for k, d, e in v] for m, v in bases.items()},
           "baselines_per_pathology": {m: {P: {k: list(x) for k, x in v[P].items()} for P in v} for m, v in per_base.items()}}
    with open(os.path.join(out, "fig6_points.json"), "x", encoding="utf-8") as f:
        json.dump(pts, f, indent=1)
    with open(os.path.join(out, "fig6_points.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "point", "detection_rate", "mean_mse"] + [f"{q}_{P}" for P in a.pathos for q in ("det", "mse")])
        for m, v in curves.items():
            for k, d, e in v:
                w.writerow([m, f"cycle {k}", f"{d:.6f}", f"{e:.6e}"] + [f"{x:.6g}" for P in a.pathos for x in per[m][P][k]])
        for m, t in traj.items():
            for k in range(len(t[0])):
                w.writerow([m, f"cycle {k}", f"{t[0][k]:.6f}", f"{t[1][k]:.6e}"]
                           + [f"{per_traj[m][P][q][k]:.6g}" for P in a.pathos for q in ("det", "mse")])
        for m, v in bases.items():
            for k, d, e in v:
                w.writerow([m, k, f"{d:.6f}", f"{e:.6e}"] + [f"{x:.6g}" for P in a.pathos for x in per_base[m][P][k]])
    for m, v in curves.items():
        print(f"[{m}] {len(v)} points, last: cycle {v[-1][0]}, detection {v[-1][1]:.3f}, MSE {v[-1][2]:.2e}")
    for m, t in traj.items():
        print(f"[{m}] {len(t[0])} cycles, last: detection {t[0][-1]:.3f}, MSE {t[1][-1]:.2e}")
    for m, v in bases.items():
        print(f"[{m}] " + ", ".join(f"{k}: {d:.3f} / {e:.2e}" for k, d, e in v))
    print(f"[healthy] {healthy}")
    print(f"[done] {', '.join(paths)}")
    return out


if __name__ == "__main__":
    main()
