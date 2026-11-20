# -*- coding: utf-8 -*-
"""Fig. 3: drift of in-domain images under SNAP and P-SNAP at a fixed level; column width.

One block per image (a healthy radiograph, a face photograph): the input once, on the left, then two rows, SNAP
and P-SNAP, at the same cycles; the MSE to the input (x 1e-3, computed on the 8-bit images shown) in a small box
in each thumbnail.  SNAP and P-SNAP of a block must be run on the same inputs.

Runs of the paper (sigma = 1.24, 600 cycles; the default --save_cycles keeps 10, 50, 150, 300, 600):

    M="--model weights/chestmnist_224.pt --config configs/chestmnist_224.json"
    python scripts/project.py $M --input data/chestmnist224/healthy_test.npy --n 64 --method snap  --sigma 1.24 --cycles 600
    python scripts/project.py $M --input data/chestmnist224/healthy_test.npy --n 64 --method psnap --sigma 1.24 --gamma 4.67 --cycles 600
    F="--model weights/ffhq.pt --config configs/ffhq_256_adm.json --input data/celebahq_test --n 8"
    python scripts/project.py $F --method snap  --sigma 1.24 --cycles 600
    python scripts/project.py $F --method psnap --sigma 1.24 --gamma 4.67 --h_rel 0.003 --cycles 600

Paper version (radiograph #0, photograph <i> of the face runs):

    python paper/fig3_drift_examples.py --snap <chest snap run>,<face snap run> --psnap <chest psnap run>,<face psnap run> \\
        --images 0,<i> --out figures

--style printed (default): the version printed in the paper (serif labels, 7 in wide, methods "SNAP" / "Projected SNAP", MSE in scientific
notation, no block labels).  Writes figures/<date-time>_fig3_drift/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import P, Run, STYLES, check_frames, image_index, ints, mse_u8, out_dir, pyplot, same_input, thumb, write_json  # noqa: E402,E501


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap", required=True, help="SNAP run folders, one block each, comma-separated")
    p.add_argument("--psnap", required=True, help="P-SNAP run folders on the same inputs, in the same order")
    p.add_argument("--images", default="0,0", help="image of each block (name or position in its runs)")
    p.add_argument("--labels", default="Chest|radiograph,Face|photograph", help="label of each block ('|' = new line)")
    p.add_argument("--cycles", default="0,10,50,150,300,600", help="columns (0 = the input, shown once)")
    p.add_argument("--no_mse", action="store_true", help="no MSE box")
    p.add_argument("--style", choices=STYLES, default="printed")
    p.add_argument("--width", type=float, default=0.0, help="inches (default: 3.28 da, 7.0 printed)")
    p.add_argument("--name", default="fig3_drift")
    p.add_argument("--out", default="figures")
    return p.parse_args(argv)


def mse_text(v):
    """printed style: MSE 4.2x10^-3."""
    import numpy as np
    e = int(np.floor(np.log10(v))) if v > 0 else 0
    return f"MSE {v / 10 ** e:.1f}$\\times 10^{{{e}}}$" if v > 0 else "MSE 0"


def main(argv=None):
    a = parse(argv)
    da = a.style == "da"
    ks = ints(a.cycles)
    kcols = [k for k in ks if k != 0]
    sr, pr = [v for v in a.snap.split(",") if v.strip()], [v for v in a.psnap.split(",") if v.strip()]
    imgs, labs = [v for v in a.images.split(",") if v.strip()], a.labels.split(",")
    if len(pr) != len(sr) or len(imgs) < len(sr):
        raise SystemExit(f"{len(sr)} SNAP runs: give as many --psnap runs and --images")
    labs += [""] * (len(sr) - len(labs))
    blocks = []
    for s_run, p_run, img, label in zip(sr, pr, imgs, labs):
        rs, rp = Run(s_run), Run(p_run)
        i = image_index(rs, img)
        j = image_index(rp, img)
        same_input(rs, rp, i, j)
        for r in (rs, rp):
            check_frames(r, kcols)
        blocks.append({"runs": [("snap", rs, i), ("psnap", rp, j)], "label": label.replace("|", "\n")})

    from matplotlib.gridspec import GridSpec
    plt = pyplot(serif=not da)
    ncol = 1 + len(kcols)
    if da:
        margin, W, gap, top = 0.47, (a.width or P.COLUMN_WIDTH), 0.07, 0.17
    else:
        margin, W, gap, top = 0.32, (a.width or 7.0), 0.10, 0.24
    c = (W - margin) / ncol                                       # side of a thumbnail (inches)
    tight = da and c < 0.45                                       # many columns: smaller titles
    f_title = (P.TEXT - 1.5 if tight else P.TEXT) if da else 10
    f_method = (P.TEXT - 1.5 if tight else P.TEXT) if da else 9
    nl = sum(len(B["runs"]) for B in blocks)
    bottom = 0.05 if da else 0.0
    H = top + nl * c + (len(blocks) - 1) * gap + bottom
    fig = plt.figure(figsize=(W, H))
    hr = []
    for j, B in enumerate(blocks):
        hr += [c] * len(B["runs"]) + ([gap] if j < len(blocks) - 1 else [])
    gs = GridSpec(len(hr), ncol, figure=fig, left=margin / W, right=1 - 0.004, bottom=max(0.002, bottom / H),
                  top=1 - top / H, height_ratios=hr, wspace=0.025, hspace=0.025)
    labels, rows = [], []
    r0 = 0
    for j, B in enumerate(blocks):
        _, run0, i0 = B["runs"][0]
        x0 = run0["x0"][i0]
        ax = fig.add_subplot(gs[r0:r0 + len(B["runs"]), 0])
        thumb(ax, x0)
        if j == 0:
            ax.set_title("input", fontsize=f_title, pad=3)
        first = ax.get_position()
        for r, (key, run, i) in enumerate(B["runs"]):
            for cc, k in enumerate(kcols):
                ax = fig.add_subplot(gs[r0 + r, 1 + cc])
                x = run.frame(k, i)
                thumb(ax, x)
                v = mse_u8(x, x0)
                rows.append({"block": j, "image": run.names[i], "method": key, "sigma": run.meta.get("sigma"), "k": k,
                             "mse": v})
                if not a.no_mse and da:
                    P.box(ax, [f"MSE {1e3 * v:.1f}"], False)
                elif not a.no_mse:
                    ax.text(0.5, 0.02, mse_text(v), transform=ax.transAxes, ha="center", va="bottom", fontsize=5.8,
                            color="white", bbox=dict(boxstyle="square,pad=0.12", fc="black", ec="none", alpha=0.5))
                if j == 0 and r == 0:
                    ax.set_title((f"k={k}" if tight else f"k = {k}") if da else f"$k = {k}$", fontsize=f_title, pad=3)
                if cc == 0:
                    bb = ax.get_position()
                    name, color = P.METHODS[key]
                    labels.append(((bb.y0 + bb.y1) / 2, name if da else {"snap": "SNAP", "psnap": "Projected SNAP"}[key],
                                   color))
                last = ax.get_position()
        if da and B["label"]:
            fig.text(0.004, (first.y1 + last.y0) / 2, B["label"], rotation=90, ha="left", va="center",
                     fontsize=P.TEXT - 1, color=P.GREY, fontweight="bold")
        r0 += len(B["runs"]) + 1
    for y, t, col in labels:
        if da:
            fig.text((margin - 0.03) / W, y, t, rotation=90, ha="right", va="center", fontsize=f_method, color=col,
                     fontweight="bold")
        else:
            fig.text(0.004, y, t, rotation=90, ha="left", va="center", fontsize=9, color=col)
    out = out_dir(a.out, a.name)
    P.save(fig, out, a.name)
    plt.close(fig)
    write_json({"figure": "Fig. 3", "style": a.style, "width_in": W, "cycles": kcols, "blocks": [
        {"label": B["label"], "runs": [{"method": key, "run": run.path, "sigma": run.meta.get("sigma"),
                                        "gamma": run.meta.get("gamma"), "image": run.names[i]} for key, run, i in B["runs"]]}
        for B in blocks], "mse_in_thumbnails": rows}, out, "figure.json")
    print(f"[fig3] -> {out}")
    return out


if __name__ == "__main__":
    main()
