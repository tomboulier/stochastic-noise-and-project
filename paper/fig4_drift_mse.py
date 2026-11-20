# -*- coding: utf-8 -*-
"""Fig. 4: mean MSE to the input along the cycles of SNAP and P-SNAP, on in-domain images; column width.

One panel per domain; in each, one curve per method: the mean over the images of the MSE to x0 (images in [0, 1],
read at EVERY cycle from trajectory.npz) and its 95 % bootstrap interval over the images (1000 draws, the same
draws at every cycle).  Log-log axes (and a linear version).  The images of several runs of a method can be pooled
(e.g. two batches of photographs): they must have the same number of cycles and the same level.

Runs of the paper: the four runs of Fig. 3 (paper/fig3_drift_examples.py): 64 healthy test radiographs and
8 face photographs, SNAP and P-SNAP (gamma = 4.67), sigma = 1.24, 600 cycles.

    python paper/fig4_drift_mse.py --snap "<chest snap run>;<face snap run>" --psnap "<chest psnap run>;<face psnap run>" \\
        --out figures

(panels separated by ';', runs pooled in a panel by ',').  The legend reads "P-SNAP (gamma = <gain of the P-SNAP
runs>)" under the panels; --gain_label none writes "P-SNAP" in the upper left of the first panel.
Writes figures/<date-time>_fig4_drift_mse/ (loglog and lin versions).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from common import P, Run, out_dir, pyplot, write_json  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap", required=True, help="SNAP runs: panels separated by ';', runs pooled in a panel by ','")
    p.add_argument("--psnap", required=True, help="P-SNAP runs, same layout")
    p.add_argument("--titles", default="Chest radiographs (ChestMNIST);Face photographs (CelebA-HQ)",
                   help="panel titles, separated by ';' (' (' starts a second line)")
    p.add_argument("--gain_label", default="auto", help="auto (gamma of the P-SNAP runs), none, or a value")
    p.add_argument("--k_max", type=int, default=0, help="curves up to this cycle (0: all)")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--scales", default="loglog,lin")
    p.add_argument("--width", type=float, default=P.COLUMN_WIDTH)
    p.add_argument("--name", default="fig4_drift_mse")
    p.add_argument("--out", default="figures")
    return p.parse_args(argv)


def pooled(paths, k_max):
    """MSE (K+1, N) of the images of one or several runs of a method, and (sigma, gamma, names, sources)."""
    runs = [Run(s) for s in paths.split(",") if s.strip()]
    M = [np.asarray(r["mse"], np.float64) for r in runs]
    if k_max:
        if any(m.shape[0] < k_max + 1 for m in M):
            raise SystemExit(f"runs shorter than k_max = {k_max}: {[m.shape for m in M]}")
        M = [m[:k_max + 1] for m in M]
    sig = {round(float(r.meta.get("sigma", float("nan"))), 6) for r in runs}
    if len({m.shape[0] for m in M}) != 1 or len(sig) != 1:
        raise SystemExit(f"pooled runs differ in cycles or level: {[(m.shape, r.meta.get('sigma')) for m, r in zip(M, runs)]}")
    names = [n for r in runs for n in r.names]
    if len(runs) > 1:                            # the same input in two pooled runs would be counted twice
        seen = {}
        for j, r in enumerate(runs):
            for n, x in zip(r.names, r["x0"]):
                seen.setdefault(x.tobytes(), {})[j] = f"{os.path.basename(r.path)}:{n}"
        twice = [list(v.values()) for v in seen.values() if len(v) > 1]
        if twice:
            print(f"[fig4] warning: {len(twice)} inputs appear in several pooled runs, e.g. {twice[0]}")
    return np.concatenate(M, 1), {"sigma": sig.pop(), "gamma": runs[0].meta.get("gamma"), "n_images": len(names),
                                  "cycles": int(M[0].shape[0] - 1), "runs": [r.path for r in runs],
                                  "methods": sorted({r.method for r in runs})}


def draw(plt, panels, label_psnap, width, log, n_boot):
    from matplotlib.ticker import FuncFormatter
    from snap.evaluation import bootstrap_ci
    fs = 8
    fig, axs = plt.subplots(1, len(panels), figsize=(width, 2.25), sharey=False, squeeze=False)
    axs = axs[0]
    for ax, (title, D) in zip(axs, panels):
        for key in ("snap", "psnap"):
            M = D[key][0]
            mean = M.mean(1)
            lo, hi = bootstrap_ci(M, n_boot=n_boot, level=0.95, seed=0)
            k = np.arange(M.shape[0])
            sel = k >= (1 if log else 0)
            name, color = P.METHODS[key]
            ax.plot(k[sel], mean[sel], color=color, lw=1.3, label=name if key == "snap" else label_psnap)
            ax.fill_between(k[sel], lo[sel], hi[sel], color=color, alpha=0.18, lw=0)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(title.replace(" (", "\n("), fontsize=fs, pad=3, color=P.GREY, fontweight="bold")
        ax.set_xlabel("cycle $k$", fontsize=fs)
        ax.tick_params(labelsize=fs - 1, width=0.6, length=2.5)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("mean MSE to $x_0$", fontsize=fs)
    bottom = 0.0
    if label_psnap != "P-SNAP":                  # longer legend with the gain: one line under the panels
        h, lab = axs[0].get_legend_handles_labels()
        fig.legend(h, lab, fontsize=fs - 1, frameon=False, loc="lower center", ncol=2, handlelength=1.8,
                   bbox_to_anchor=(0.5, 0.0), columnspacing=1.5)
        bottom = 0.2 / fig.get_figheight()
    else:
        axs[0].legend(fontsize=fs - 1, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, bottom, 1, 1))
    return fig


def main(argv=None):
    a = parse(argv)
    sp, pp = [v for v in a.snap.split(";") if v.strip()], [v for v in a.psnap.split(";") if v.strip()]
    titles = a.titles.split(";")
    if len(pp) != len(sp) or len(titles) < len(sp):
        raise SystemExit(f"{len(sp)} SNAP panels: give as many P-SNAP panels and --titles")
    panels, info = [], []
    for title, s_runs, p_runs in zip(titles, sp, pp):
        D = {"snap": pooled(s_runs, a.k_max), "psnap": pooled(p_runs, a.k_max)}
        panels.append((title, D))
        info.append({"title": title, **{k: dict(v[1], mean_mse={str(c): float(v[0][c].mean())
                                                                for c in (1, 10, 50, 150, 300, 600) if c < v[0].shape[0]})
                                        for k, v in D.items()}})
        for k, (M, meta) in D.items():
            print(f"[fig4] {title} | {k:5s} sigma {meta['sigma']:.2f} | {M.shape[1]} images | K = {M.shape[0] - 1} | "
                  f"MSE at the last cycle {M[-1].mean():.2e}")
    if a.gain_label == "none":
        label = "P-SNAP"
    elif a.gain_label == "auto":
        g = {D["psnap"][1]["gamma"] for _, D in panels}
        if len(g) == 1 and None not in g:
            label = f"P-SNAP (γ = {float(g.pop()):g})"
        else:
            print(f"[fig4] warning: P-SNAP runs with gains {g}: the legend reads P-SNAP (--gain_label to set it)")
            label = "P-SNAP"
    else:
        label = f"P-SNAP (γ = {a.gain_label})"
    plt = pyplot()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    out = out_dir(a.out, a.name)
    for scale in [s for s in a.scales.split(",") if s]:
        fig = draw(plt, panels, label, a.width, scale == "loglog", a.n_boot)
        P.save(fig, out, f"{a.name}_{scale}")
        plt.close(fig)
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})
    write_json({"figure": "Fig. 4", "width_in": a.width, "legend": label, "n_boot": a.n_boot, "panels": info}, out,
               "figure.json")
    print(f"[fig4] -> {out}")
    return out


if __name__ == "__main__":
    main()
