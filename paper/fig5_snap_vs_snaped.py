# -*- coding: utf-8 -*-
"""Fig. 5: one test image per pathology, projected with SNAP at a fixed level and with SNAPED; column width.

Rows: the input, then the cycles k; for each pathology two columns, SNAP and SNAPED.  In every thumbnail a small
box gives p, the probability of the pathology read by the classifier (red when p > the threshold of that head),
and the MSE to the input (x 1e-3).  SNAP's last image is framed; SNAPED's stopping image is framed in the first
row whose cycle is >= its stop (its stopping cycle written in the box), and its later cells are grey.

Runs of the paper (ChestMNIST 224 px, the 64 first test images of each pathology; the classifier is read at every
cycle; the default --save_cycles keeps 5, 20, 50, 150).  For each P in Edema, Effusion, Cardiomegaly:

    M="--model weights/chestmnist_224.pt --config configs/chestmnist_224.json"
    python scripts/project.py $M --input data/chestmnist224/<P>_test.npy --n 64 --method snap --sigma 1.24 \\
        --cycles 150 --classifier weights/medmnist --heads <P>
    python scripts/project.py $M --input data/chestmnist224/<P>_test.npy --n 64 --method snaped \\
        --estimator estimators/cnn_<P>.pt --sigma_min 0.34 --sigma_max 2.04 --max_cycles 500 \\
        --classifier weights/medmnist --heads <P>
    # thresholds of the heads, calibrated once on real test images (Youden's index)
    python scripts/calibrate_classifier.py --classifier weights/medmnist \\
        --negatives data/chestmnist224/healthy_test.npy \\
        --positives Edema=data/chestmnist224/edema_test.npy,Effusion=...,Cardiomegaly=... --out calibration/r18.json

Paper version (Edema #52, Effusion #55, Cardiomegaly #31):

    python paper/fig5_snap_vs_snaped.py --images Edema:52,Effusion:55,Cardiomegaly:31 \\
        --snap Edema=<run>,Effusion=<run>,Cardiomegaly=<run> --snaped Edema=<run>,Effusion=<run>,Cardiomegaly=<run> \\
        --calibration calibration/r18.json --cycles 5,20,50,150 --out figures

--style printed (default): the version printed in the paper (3.4 in wide, smaller labels).  Writes figures/<date-time>_fig5_snap_vs_snaped/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import P, Run, STYLES, check_frames, image_index, ints, out_dir, pyplot, same_input, thumb, write_json  # noqa: E402,E501

# drawing parameters of the two styles (inches, fractions of the figure, points)
LAYOUT = {
    "da": dict(width=P.COLUMN_WIDTH, left=0.165, right=0.978, sep=0.16, space=0.035, top=0.34, bottom=0.03,
               method_fs=P.TEXT - 1, row_fs=P.TEXT, row_dx=0.015, patho_fs=P.TEXT, patho_dy=0.03,
               corner="SNAP\nσ = {:.2f}", corner_fs=P.TEXT - 1, corner_x=0.004),
    "printed": dict(width=3.4, left=0.115, right=0.995, sep=0.14, space=0.04, top=0.34, bottom=0.04,
                    method_fs=5.2, row_fs=5.6, row_dx=0.012, patho_fs=6.0, patho_dy=0.05,
                    corner="SNAP:\nσ = {:.2f}", corner_fs=4.8, corner_x=0.012),
}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default="Edema:52,Effusion:55,Cardiomegaly:31",
                   help="pathology:image, one block of two columns each (image: name or position in the runs)")
    p.add_argument("--snap", required=True, help="pathology=run folder,... (SNAP at a fixed level)")
    p.add_argument("--snaped", required=True, help="pathology=run folder,... (SNAPED, same inputs)")
    p.add_argument("--calibration", default=None, help="thresholds file of scripts/calibrate_classifier.py")
    p.add_argument("--thresholds", default="", help="pathology=threshold,... (instead of / over --calibration)")
    p.add_argument("--monitor", default="p_{}", help="monitor of the runs holding p ({} = the pathology)")
    p.add_argument("--cycles", default="5,20,50,150", help="rows after the input; SNAP stops at the last one")
    p.add_argument("--box_fs", type=float, default=P.BOX, help="size of the p / MSE boxes (pt)")
    p.add_argument("--style", choices=STYLES, default="printed")
    p.add_argument("--name", default="fig5_snap_vs_snaped")
    p.add_argument("--out", default="figures")
    return p.parse_args(argv)


def pairs(s):
    out = {}
    for item in [v for v in s.split(",") if v.strip()]:
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


def cells(snap, ed, i, mon, ks):
    """Per method, the list of cells (image or None, p, mse, framed, grey, extra line) of the rows [0] + ks."""
    cols = [0] + list(ks)
    c = ed.stop_cycle(i)
    a, b = [], []
    for j, k in enumerate(cols):
        a.append((snap.frame(k, i), float(snap.monitor(mon, k)[i]), float(snap["mse"][k, i]), j == len(cols) - 1,
                  False, None))
    done = False
    last = int(ed["frame_cycles"][-1])
    if last < c:
        raise SystemExit(f"{ed.path}: last kept frame {last} < stopping cycle {c} of image {i}")
    for k in cols:
        if done:
            b.append((None, None, None, False, True, None))
        elif k < c:
            b.append((ed.frame(k, i), float(ed.monitor(mon, k)[i]), float(ed["mse"][k, i]), False, False, None))
        else:                                                     # first row at or after the stop: the frozen image
            b.append((ed["frames"][-1][i], float(ed.monitor(mon, c)[i]), float(ed["mse"][c, i]), True, False,
                      f"k = {c}"))
            done = True
    return a, b


def draw_cell(ax, item, k, thr, fs):
    im, p, mse, framed, grey, extra = item
    if grey:
        ax.set_facecolor(P.LIGHT)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return
    thumb(ax, im, frame=P.INK if framed else None, lw=1.6 if framed else 0.6)
    lines = [f"p {p:.3f}"] + ([f"MSE {1e3 * mse:.1f}"] if k > 0 else []) + ([extra] if extra else [])
    P.box(ax, lines, p > thr, fs=fs)


def main(argv=None):
    a = parse(argv)
    L = LAYOUT[a.style]
    ks = ints(a.cycles)
    rs, re_ = pairs(a.snap), pairs(a.snaped)
    thr = {}
    if a.calibration:
        from snap.evaluation import load_thresholds
        thr.update(load_thresholds(a.calibration))
    thr.update({k: float(v) for k, v in pairs(a.thresholds).items()})
    blocks = []
    for item in [v for v in a.images.split(",") if v.strip()]:
        patho, _, img = item.partition(":")
        if patho not in rs or patho not in re_:
            raise SystemExit(f"{patho}: give its runs with --snap {patho}=... and --snaped {patho}=...")
        if patho not in thr:
            raise SystemExit(f"{patho}: no threshold (--calibration / --thresholds)")
        snap, ed = Run(rs[patho]), Run(re_[patho])
        i = image_index(snap, img)
        j = image_index(ed, img)
        same_input(snap, ed, i, j)
        check_frames(snap, ks)
        mon = a.monitor.format(patho)
        for r in (snap, ed):
            if f"mon_{mon}" not in r:
                raise SystemExit(f"{r.path}: no monitor {mon} (project.py --classifier ... --heads {patho})")
        if j != i:
            raise SystemExit(f"{patho}: image {img} has different positions in the two runs ({i}, {j})")
        blocks.append({"P": patho, "i": i, "snap": snap, "ed": ed, "mon": mon, "thr": thr[patho],
                       "stop": ed.stop_cycle(i)})
    sigmas = {round(float(b["snap"].meta["sigma"]), 2) for b in blocks}
    if len(sigmas) != 1:
        print(f"[fig5] warning: the SNAP runs have different levels {sorted(sigmas)}; the first one is written")
    sig = float(blocks[0]["snap"].meta["sigma"])

    plt = pyplot()
    if a.style == "printed":
        plt.rcParams.update({"font.size": 7})
    cols = [0] + ks
    nc, nb = len(cols), len(blocks)
    W = L["width"]
    wr = []
    for b in range(nb):
        wr += [1, 1] + ([L["sep"]] if b < nb - 1 else [])
    sp = L["space"]
    u = W * (L["right"] - L["left"]) / (sum(wr) + (len(wr) - 1) * sp * (sum(wr) / len(wr)))
    H = u * (nc + (nc - 1) * sp) + L["top"] + L["bottom"]
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(nc, len(wr), width_ratios=wr, left=L["left"], right=L["right"], top=1 - L["top"] / H,
                          bottom=L["bottom"] / H, wspace=sp, hspace=sp)
    for b, B in enumerate(blocks):
        axes = []
        for r, column in enumerate(cells(B["snap"], B["ed"], B["i"], B["mon"], ks)):
            for cc, item in enumerate(column):
                ax = fig.add_subplot(gs[cc, 3 * b + r])
                axes.append(ax)
                draw_cell(ax, item, cols[cc], B["thr"], a.box_fs)
                if cc == 0:
                    name, color = P.METHODS["snap" if r == 0 else "snaped"]
                    ax.set_title(name, fontsize=L["method_fs"], color=color, fontweight="bold", pad=2)
                if b == 0 and r == 0:
                    bb = ax.get_position()
                    fig.text(L["left"] - L["row_dx"], (bb.y0 + bb.y1) / 2, "input" if cols[cc] == 0 else f"k = {cols[cc]}",
                             ha="right", va="center", fontsize=L["row_fs"])
        pa, pb = axes[0].get_position(), axes[-1].get_position()
        fig.text((pa.x0 + pb.x1) / 2, 1 - L["patho_dy"] / H, B["P"], ha="center", va="top", fontsize=L["patho_fs"],
                 color=P.GREY, fontweight="bold")
    fig.text(L["corner_x"], 1 - L["patho_dy"] / H, L["corner"].format(sig), ha="left", va="top",
             fontsize=L["corner_fs"], color=P.C_SNAP)
    out = out_dir(a.out, a.name)
    P.save(fig, out, a.name)
    plt.close(fig)
    write_json({"figure": "Fig. 5", "style": a.style, "width_in": W, "cycles": ks, "snap_sigma": sig, "blocks": [
        {"pathology": B["P"], "image": B["snap"].names[B["i"]], "snaped_stop": B["stop"], "threshold": B["thr"],
         "snap_run": B["snap"].path, "snaped_run": B["ed"].path,
         "snaped_sigma_range": [B["ed"].meta.get("sigma_min"), B["ed"].meta.get("sigma_max")]} for B in blocks]},
        out, "figure.json")
    print(f"[fig5] -> {out} (SNAPED stops: " + ", ".join(f"{B['P']} k = {B['stop']}" for B in blocks) + ")")
    return out


if __name__ == "__main__":
    main()
