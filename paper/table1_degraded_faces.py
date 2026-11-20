# -*- coding: utf-8 -*-
"""Table 1 of the paper: degraded photographs of faces, LPIPS and MSE (x 1e-3) to the clean photograph.

For every level (moderate, strong), degradation (blur, JPEG, pixelation, noise) and method:
  - the setting is the one with the lowest mean LPIPS on the VALIDATION photographs (SNAP and P-SNAP: a setting is
    (noise level, number of cycles k >= 1), all the levels of the method's runs together; a baseline: its settings);
  - at that setting, on the TEST photographs: mean LPIPS and mean MSE to the clean photograph (both columns describe
    the same output images), with 95 % bootstrap intervals over the photographs (md / csv);
  - the best value of each column among the methods of the table is in bold (compared at the printed precision);
  - P-SNAP against every other method, paired on the same photographs (paired.csv).
"Degraded input" is the degraded photograph itself.  MSE: images in [0, 1], mean over pixels and channels.

Runs needed, for each level (L = a folder of scripts/degrade_faces.py --level moderate | strong: 100 test + 32
validation photographs per degradation; all.npz = the four degradations in one array), with the FFHQ 256 model:

    M="--model weights/ffhq.pt --config configs/ffhq_256_adm.json --input $L/all.npz --batch 16 --cycles 10 \\
       --save_cycles 1,2,3,4,5,6,7,8,9,10 --out outputs/faces_moderate"
    python scripts/project.py $M --method snap  --sigma 0.34          # and --sigma 0.72
    python scripts/project.py $M --method psnap --gamma 3 --h_rel 0.003 --sigma 0.34          # and --sigma 0.72
        (the paper's runs: settings (sigma, k) with sigma 0.34 and 0.72 and k = 1 ... 10; strong level: sigma 0.34,
         0.72 and 1.24.  One run per level of noise; per-degradation inputs <degradation>.npz work as well)
    python scripts/run_baseline.py --method sdedit | autoddpm | diip ... --recipe faces   (see its docstring: SDEdit tau
        50 ... 400, strong level also 500; AutoDDPM masking 100, 200, resampling 50; DIIP early stopping)
    python scripts/evaluate_runs.py --runs "outputs/faces_moderate/*" --reference $L/clean.npz --name faces

Then (CPU, seconds):

    python paper/table1_degraded_faces.py --level "moderate=outputs/faces_moderate/*" --level "strong=outputs/faces_strong/*"

A run's method is read in its run.json (snap, psnap; baselines: sdedit, autoddpm, diip, anoddpm_gaussian, vae) or
given as method:<run>.  Runs of the same method are merged (several levels for SNAP / P-SNAP, parts of the
photographs for a baseline).  The image names (<degradation>-<split>_<i>) give the degradation and the split; the
degradation parameters are read in the degradations.json next to each run's input.
Writes a new folder <out>/<date-time>_table1/: table1.tex, table1.md, table1.csv, paired.csv, settings.json.
"""

import argparse
import csv
import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from snap.degradations import COLUMN_NAMES, DEGRADATIONS  # noqa: E402
from snap.evaluation import bootstrap_ci  # noqa: E402
from snap.images import new_run_dir  # noqa: E402
from snap.runs import Run, load_run  # noqa: E402

TEX = {"diip": "DIIP~\\cite{diip}", "autoddpm": "AutoDDPM~\\cite{autoddpm}", "sdedit": "SDEdit~\\cite{sdedit}",
       "anoddpm_gaussian": "AnoDDPM~\\cite{anoddpm}", "anoddpm_simplex": "AnoDDPM~\\cite{anoddpm}", "vae": "VAE~\\cite{VAE}",
       "snap": "SNAP", "psnap": "P-SNAP"}
NAMES = {"input": "Degraded input", "diip": "DIIP", "autoddpm": "AutoDDPM", "sdedit": "SDEdit",
         "anoddpm_gaussian": "AnoDDPM (Gaussian)", "anoddpm_simplex": "AnoDDPM (simplex)", "vae": "VAE",
         "snap": "SNAP", "psnap": "P-SNAP"}
ALIASES = {"snap": "snap", "psnap": "psnap", "sdedit": "sdedit", "autoddpm": "autoddpm", "diip": "diip", "vae": "vae",
           "anoddpm": "anoddpm_gaussian", "anoddpm_g": "anoddpm_gaussian", "anoddpm_gaussian": "anoddpm_gaussian",
           "anoddpm_s": "anoddpm_simplex", "anoddpm_simplex": "anoddpm_simplex"}
REF = "psnap"


# --------------------------------------------------------------------------- reading
def expand(spec):
    out = []
    for item in [s for s in spec.split(",") if s.strip()]:
        fam, path = None, item
        head, sep, tail = item.partition(":")
        if sep and head.lower() in ALIASES and not os.path.exists(item):
            fam, path = ALIASES[head.lower()], tail
        if any(c in path for c in "*?["):
            out += [(fam, d) for d in sorted(glob.glob(path)) if os.path.exists(os.path.join(d, "run.json"))]
        else:
            out.append((fam, path))
    return out


def family(run, forced):
    if forced:
        return forced
    m = str(run.method).lower().replace("-", "_")
    noise = " ".join([str(run.meta.get(k, "")) for k in ("noise", "noise_type")]
                     + [str(s.get("noise", "")) for s in run.meta.get("settings", []) if isinstance(s, dict)]).lower()
    if m.startswith("anoddpm") and "simplex" in noise:
        return "anoddpm_simplex"
    if m in ALIASES:
        return ALIASES[m]
    raise SystemExit(f"{run.path}: unknown method {m!r}; give it as <method>:{run.path}")


def degradation_params(run):
    """{degradation: {parameter, value, description}} from the degradations.json next to the run's input (the first
    one when several inputs were concatenated), or None."""
    inp = str(run.meta.get("input", "")).split(",")[0]
    f = os.path.join(os.path.dirname(inp), "degradations.json")
    if inp and os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            return json.load(fh).get("degradations")
    return None


class Level:
    """All the runs of one level: per method and degradation, LPIPS / MSE (settings x photographs)."""

    def __init__(self, name, items, measures):
        self.name, self.cols, self.split = name, {}, {}
        self.data = {}                                   # (method, degradation) -> {setting: {"lp", "ms", "nfe", "order"}}
        self.input = {}                                  # (degradation, clean name) -> (lpips, mse)
        self.x0 = {}                                     # (degradation, clean name) -> uint8 input
        self.params, self.gammas, self.sources = None, set(), []
        for forced, path in items:
            run = load_run(path)
            fam = family(run, forced)
            f = os.path.join(path, f"measures_{measures}.npz")
            if not os.path.exists(f):
                raise SystemExit(f"{f} missing: scripts/evaluate_runs.py --reference <clean.npz> --name {measures}")
            m = dict(np.load(f, allow_pickle=False))
            if "lpips_ref" not in m:
                raise SystemExit(f"{f}: no LPIPS to the clean photographs (evaluate_runs.py --reference)")
            if [str(x) for x in m["names"]] != run.names:
                raise SystemExit(f"{f}: its images are not those of the run")
            self.sources.append((fam, path))
            if fam == "psnap" and run.meta.get("gamma") is not None:
                self.gammas.add(float(run.meta["gamma"]))
            prm = degradation_params(run)
            if prm is not None:
                if self.params is not None and prm != self.params:
                    raise SystemExit(f"{path}: its inputs do not have the degradations of the other runs of {name}")
                self.params = prm
            if isinstance(run, Run):
                if fam not in ("snap", "psnap"):
                    raise SystemExit(f"{path}: a run of {run.method} cannot be a fixed-level SNAP / P-SNAP setting")
                tau, calls = int(run.meta["tau"]), float(run.meta.get("calls_per_cycle", np.nan))
                sig = float(run.meta.get("sigma", np.nan))
                sett = [(i, f"tau{tau}_k{int(k)}", (tau, int(k)), calls * int(k), f"σ = {sig:.2f}, k = {int(k)}")
                        for i, k in enumerate(m["cycles"]) if int(k) >= 1]
            else:
                nfe = np.asarray(run["nfe"], float)
                sett = [(i, str(s), None, float(nfe[i]) if i < len(nfe) else np.nan, str(s))
                        for i, s in enumerate(m["labels"])]
            for j, n in enumerate(run.names):
                if "-" not in n:
                    raise SystemExit(f"{path}: image {n!r} is not named <degradation>-<split>_<i> (scripts/degrade_faces.py)")
                g, clean = n.split("-", 1)
                sp = clean.split("_", 1)[0]
                if (g, clean) in self.x0:
                    if not np.array_equal(self.x0[(g, clean)], run["x0"][j]):
                        raise SystemExit(f"{path}: input {n} differs from the same input in another run of {name}")
                else:
                    self.x0[(g, clean)] = run["x0"][j]
                    self.input[(g, clean)] = (float(m["lpips_ref_x0"][j]), float(m["mse_ref_x0"][j]))
                    self.cols.setdefault(g, []).append(clean)
                    self.split[(g, clean)] = sp
                for i, key, order, nfe, label in sett:
                    S = self.data.setdefault((fam, g), {}).setdefault(key, {"vals": {}, "nfe": nfe, "order": order,
                                                                             "label": label})
                    if clean in S["vals"]:
                        raise SystemExit(f"{fam} {g} {key}: photograph {clean} measured twice ({path})")
                    S["vals"][clean] = (float(m["lpips_ref"][i, j]), float(m["mse_ref"][i, j]))

    def matrices(self, fam, g):
        """-> (setting keys, labels, nfe, LP (R, M), MS (R, M)) over the photographs of degradation g."""
        D = self.data[(fam, g)]
        keys = list(D)
        if all(D[k]["order"] is not None for k in keys):          # SNAP / P-SNAP: by level, then cycle
            keys.sort(key=lambda k: D[k]["order"])
        names = self.cols[g]
        LP = np.full((len(keys), len(names)), np.nan)
        MS = np.full((len(keys), len(names)), np.nan)
        for r, k in enumerate(keys):
            for c, n in enumerate(names):
                if n in D[k]["vals"]:
                    LP[r, c], MS[r, c] = D[k]["vals"][n]
        return keys, [D[k]["label"] for k in keys], [D[k]["nfe"] for k in keys], LP, MS


def ci(v, n_boot):
    lo, hi = bootstrap_ci(np.asarray(v, float), n_boot=n_boot, level=0.95, seed=0)
    return float(lo), float(hi)


def choose(LP, va, te):
    """Index of the setting with the lowest mean LPIPS on the validation photographs (on the test photographs when
    the method has no validation photograph); the first one in case of a tie."""
    sub = LP[:, va] if np.isfinite(LP[:, va]).any() else LP[:, te]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(sub, 1)
    return int(np.nanargmin(means))


# --------------------------------------------------------------------------- LaTeX
def level_sentence(lv, degs, first):
    if not lv.params:
        return ""
    parts = []
    for g in degs:
        if g not in lv.params:
            continue
        v = format(float(lv.params[g]["value"]), "g")
        parts.append({"blur": f"blur of radius {v} pixels", "jpeg": f"JPEG quality {v}", "pixelation": f"pixelation $\\times${v}",
                      "noise": f"noise {v}"}[g] if first else
                     {"blur": f"{v} pixels", "jpeg": f"quality {v}", "pixelation": f"$\\times${v}", "noise": v}[g])
    return f"{lv.name.capitalize()} level: " + ", ".join(parts) + ". "


def latex(a, levels, rows, degs, methods):
    d = {(r["level"], r["degradation"], r["method"]): r for r in rows}
    nc = 2 * len(degs)
    env = "table*" if a.full_width else "table"
    b = [f"\\begin{{{env}}}[t]", "  \\centering", "  \\small" if a.full_width else "  \\footnotesize",
         "  \\setlength{\\tabcolsep}{" + ("5pt" if a.full_width else "2pt") + "}",
         "  \\begin{tabular}{@{}l " + " ".join("cc" for _ in degs) + "@{}}", "    \\toprule",
         "    & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{COLUMN_NAMES[g]}}}" for g in degs) + " \\\\",
         "    " + "".join(f"\\cmidrule({'lr' if i < len(degs) - 1 else 'l'}){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(degs))),
         "    method & " + " & ".join("LPIPS & MSE" for _ in degs) + " \\\\"]
    meths = [("input", "Degraded input")] + [(m, TEX.get(m, m)) for m in methods]
    for lv in levels:
        b += ["    \\midrule", f"    \\multicolumn{{{nc + 1}}}{{@{{}}l}}{{\\emph{{{lv.name} level}}}} \\\\"]
        best = {}
        for g in degs:
            for q in ("lpips", "mse"):
                vs = [d[(lv.name, g, m)][q] for m, _ in meths[1:] if (lv.name, g, m) in d]
                best[(g, q)] = min(vs) if vs else None
        for m, tex in meths:
            cells = []
            for g in degs:
                r = d.get((lv.name, g, m))
                for q, fmt, scale in (("lpips", ".3f", 1.0), ("mse", ".2f", 1e3)):
                    if r is None:
                        cells.append("--")
                        continue
                    s = format(r[q] * scale, fmt)
                    bold = m != "input" and best[(g, q)] is not None and s == format(best[(g, q)] * scale, fmt)
                    cells.append(f"\\textbf{{{s}}}" if bold else s)
            b.append(f"    {tex} & " + " & ".join(cells) + " \\\\")
    lv0 = levels[0]
    r0 = d.get((lv0.name, degs[0], "input"))
    counts = (f"{r0['n_test']} test photographs per degradation, settings chosen on {r0['n_val']} validation photographs"
              if r0 else "")
    gam = sorted(set().union(*[lv.gammas for lv in levels]))
    g_txt = f", with $\\gamma = {gam[0]:g}$" if len(gam) == 1 else ""
    caption = ("Degraded face photographs: LPIPS ($\\downarrow$) and MSE ($\\times10^{-3}$, $\\downarrow$) to the clean "
               f"photograph ({counts}). "
               + "".join(level_sentence(lv, degs, i == 0) for i, lv in enumerate(levels))
               + "The setting of each method (noise level and number of cycles for SNAP and P-SNAP"
               + g_txt + ") is the one with the lowest mean LPIPS on the validation photographs; both columns are "
               "measured at that setting, on the same output images.")
    b += ["    \\bottomrule", "  \\end{tabular}", f"  \\caption{{{caption}}}", "  \\label{tab:faces}", f"\\end{{{env}}}"]
    return "% Table 1, written by paper/table1_degraded_faces.py\n" + "\n".join(b) + "\n"


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", action="append", required=True,
                   help="'name=run,run,...' (glob patterns allowed; method:run to force a method); repeat per level")
    p.add_argument("--measures", default="faces", help="read <run>/measures_<name>.npz (scripts/evaluate_runs.py)")
    p.add_argument("--methods", default="diip,autoddpm,sdedit,snap,psnap", help="rows of the LaTeX table, in order")
    p.add_argument("--degradations", default=",".join(DEGRADATIONS), help="columns, in order")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--full_width", type=int, default=0, help="1: table* (\\small, 5 pt columns); 0: one column (paper)")
    p.add_argument("--out", default="paper_outputs")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:                                    # noqa: BLE001
        pass
    degs = [g for g in a.degradations.split(",") if g]
    methods = [ALIASES.get(m, m) for m in a.methods.split(",") if m]
    levels = []
    for spec in a.level:
        name, _, items = spec.partition("=")
        runs = expand(items)
        if not runs:
            raise SystemExit(f"--level {name}: no run")
        levels.append(Level(name.strip(), runs, a.measures))
        print(f"[{name}] {len(runs)} runs")
    rows, paired, chosen = [], [], {}
    for lv in levels:
        rank = {m: i for i, m in enumerate(methods + [m for m in ALIASES.values() if m not in methods])}
        fams = sorted({f for f, _ in lv.data}, key=lambda f: rank.get(f, 99))
        for g in degs:
            if g not in lv.cols:
                print(f"[{lv.name}] {g}: no photograph, empty column")
                continue
            names = lv.cols[g]
            sp = np.array([lv.split[(g, n)] for n in names])
            te, va = sp == "test", sp == "val"
            inp = np.array([lv.input[(g, n)] for n in names])
            desc = lv.params[g]["description"] if lv.params and g in lv.params else ""
            e_lp, e_ms = inp[te, 0], inp[te, 1]
            rows.append({"level": lv.name, "degradation": g, "parameter": desc, "method": "input", "setting": "-",
                         "n_test": int(te.sum()), "n_val": int(va.sum()), "lpips": float(e_lp.mean()),
                         "lpips_ci": ci(e_lp, a.n_boot), "mse": float(e_ms.mean()), "mse_ci": ci(e_ms, a.n_boot), "nfe": 0.0})
            per_img = {}
            for fam in fams:
                if (fam, g) not in lv.data:
                    continue
                keys, labels, nfe, LP, MS = lv.matrices(fam, g)
                rv = choose(LP, va, te)
                tm = te & np.isfinite(LP[rv]) & np.isfinite(MS[rv])
                lp, ms = LP[rv, tm], MS[rv, tm]
                per_img[fam] = (LP[rv], MS[rv], tm)
                chosen[f"{lv.name}|{g}|{fam}"] = keys[rv]
                rows.append({"level": lv.name, "degradation": g, "parameter": desc, "method": fam, "setting": labels[rv],
                             "n_test": int(tm.sum()), "n_val": int(va.sum()), "lpips": float(lp.mean()),
                             "lpips_ci": ci(lp, a.n_boot), "mse": float(ms.mean()), "mse_ci": ci(ms, a.n_boot),
                             "nfe": float(nfe[rv])})
            if REF in per_img:
                lr, mr, tr = per_img[REF]
                for fam, (lo, mo, to) in per_img.items():
                    if fam == REF:
                        continue
                    c = tr & to
                    dl, dm = lr[c] - lo[c], mr[c] - mo[c]
                    paired.append({"level": lv.name, "degradation": g, "against": fam, "n": int(c.sum()),
                                   "d_lpips": float(dl.mean()), "d_lpips_ci": ci(dl, a.n_boot),
                                   "d_mse": float(dm.mean()), "d_mse_ci": ci(dm, a.n_boot)})
    out = new_run_dir(a.out, "table1")
    for name, L in (("table1.csv", rows), ("paired.csv", paired)):
        if not L:
            continue
        with open(os.path.join(out, name), "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(c for r in L for c in r)))
            w.writeheader()
            w.writerows(L)
    f3 = lambda v: f"{1e3 * v:.2f}"                                                    # noqa: E731
    t = ["# Table 1: degraded photographs of faces, LPIPS and MSE to the clean photograph\n",
         "Setting of each method chosen on the validation photographs (lowest mean LPIPS), LPIPS and MSE (x1e-3) measured "
         "on the test photographs at that setting. 95 % bootstrap intervals over the photographs.\n",
         "| level | degradation | method | setting | n test / val | LPIPS [CI] | MSE x1e-3 [CI] | network calls |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t.append(f"| {r['level']} | {r['degradation']} ({r['parameter']}) | {NAMES.get(r['method'], r['method'])} | "
                 f"{r['setting']} | {r['n_test']} / {r['n_val']} | {r['lpips']:.3f} [{r['lpips_ci'][0]:.3f} ; "
                 f"{r['lpips_ci'][1]:.3f}] | {f3(r['mse'])} [{f3(r['mse_ci'][0])} ; {f3(r['mse_ci'][1])}] | {r['nfe']:.0f} |")
    if paired:
        t += ["\n## P-SNAP against each method, paired on the same photographs (P-SNAP - other; < 0: P-SNAP better)\n",
              "| level | degradation | against | n | d LPIPS [95 % CI] | d MSE x1e-3 [95 % CI] |", "|---|---|---|---|---|---|"]
        for r in paired:
            t.append(f"| {r['level']} | {r['degradation']} | {NAMES.get(r['against'], r['against'])} | {r['n']} | "
                     f"{r['d_lpips']:+.3f} [{r['d_lpips_ci'][0]:+.3f} ; {r['d_lpips_ci'][1]:+.3f}] | {1e3 * r['d_mse']:+.2f} "
                     f"[{1e3 * r['d_mse_ci'][0]:+.2f} ; {1e3 * r['d_mse_ci'][1]:+.2f}] |")
    t.append("\nSources: " + " ; ".join(f"{lv.name} = " + ", ".join(f"{f}:{os.path.basename(s.rstrip('/'))}"
                                                                     for f, s in lv.sources) for lv in levels))
    with open(os.path.join(out, "table1.md"), "x", encoding="utf-8") as f:
        f.write("\n".join(t) + "\n")
    print("\n".join(t))
    shown = [m for m in methods if any(r["method"] == m for r in rows)]
    for m in methods:
        if m not in shown:
            print(f"[table1] warning: no run of {m}, row left out of the LaTeX table")
    with open(os.path.join(out, "table1.tex"), "x", encoding="utf-8") as f:
        f.write(latex(a, levels, rows, degs, shown))
    with open(os.path.join(out, "settings.json"), "x", encoding="utf-8") as f:
        json.dump(chosen, f, indent=1)
    print(f"[done] -> {out}")
    return out


if __name__ == "__main__":
    main()
