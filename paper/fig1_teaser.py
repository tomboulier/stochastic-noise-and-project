# -*- coding: utf-8 -*-
"""Fig. 1: out-of-domain images projected with P-SNAP; one row per image, input + cycles, page width.

Each row comes from its own run folder (scripts/project.py) with its own image and cycles; all rows have the
same number of columns.  Labels "Out-of-domain image" -> "Projected image" joined by an arrow; nothing else.

Runs of the paper (P-SNAP at a fixed level; the default --save_cycles keeps 0, 1, 2, 3, 5):

    # radiograph: ChestMNIST 224 px DDPM (healthy), isolated cardiomegalies of the test split
    python scripts/project.py --model weights/chestmnist_224.pt --config configs/chestmnist_224.json \\
        --input data/chestmnist224/cardiomegaly_test.npy --n 64 --method psnap --sigma 2.04 --gamma 4.67 --cycles 5
    # painted portrait: FFHQ 256 DDPM, test paintings (64)
    python scripts/project.py --model weights/ffhq.pt --config configs/ffhq_256_adm.json --input data/paintings_test \\
        --method psnap --sigma 2.04 --gamma 3 --h_rel 0.003 --cycles 4
    # pixelated photograph (x8): FFHQ 256 DDPM
    python scripts/project.py --model weights/ffhq.pt --config configs/ffhq_256_adm.json --input data/faces_pixel8 \\
        --method psnap --sigma 0.72 --gamma 3 --h_rel 0.003 --cycles 5

Paper version (radiograph #29, painting #31, photograph #4 of the runs above; the defaults):

    python paper/fig1_teaser.py --runs <radiograph run>,<painting run>,<photograph run> --out figures

--middle P-SNAP writes the method name on the arrow (the variant of the figure with the method).  --style printed (default): the version printed in the paper
(serif labels, 7 in wide, 10 pt); --style da: the later unified style (sans-serif, drawn at the printed width).  Writes figures/<date-time>_fig1_teaser/ (PDF, PNG, JSON).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import P, Run, STYLES, arrow_header, check_frames, grid, image_index, ints, out_dir, pyplot, thumb, write_json  # noqa: E402,E501


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="run folders, one row each, comma-separated")
    p.add_argument("--images", default="29,31,4", help="image of each row (name or position in its run)")
    p.add_argument("--cycles", default="0,1,2,3,5;0,1,2,3,4;0,1,2,3,5",
                   help="cycles of each row, rows separated by ';' (0 = the input); same number in every row")
    p.add_argument("--left", default="Out-of-domain image")
    p.add_argument("--right", default="Projected image")
    p.add_argument("--middle", default="", help="text on the arrow (e.g. P-SNAP); empty in the paper")
    p.add_argument("--style", choices=STYLES, default="printed")
    p.add_argument("--name", default="fig1_teaser")
    p.add_argument("--out", default="figures")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    paths = [v for v in a.runs.split(",") if v.strip()]
    imgs = [v for v in a.images.split(",") if v.strip()]
    cycs = [v for v in a.cycles.split(";") if v.strip()]
    if len(imgs) < len(paths) or len(cycs) < len(paths):
        raise SystemExit(f"{len(paths)} runs: give as many --images and --cycles rows")
    rows = []
    for path, img, cyc in zip(paths, imgs, cycs):
        run = Run(path)
        i = image_index(run, img)
        ks = ints(cyc)
        check_frames(run, ks)
        rows.append({"run": run, "i": i, "ks": ks})
    nc = len(rows[0]["ks"])
    if any(len(r["ks"]) != nc for r in rows):
        raise SystemExit(f"every row needs the same number of cycles ({[r['ks'] for r in rows]})")
    da = a.style == "da"
    plt = pyplot(serif=not da)
    W, fs = (P.PAGE_WIDTH, P.TITLE) if da else (7.0, 10.0)
    band = fs * 1.55 / 72                                         # header: one line of labels
    fig, gs, H = grid(plt, W, [1] * len(rows), nc, left=0.006, right=0.994, top=band, bottom=0.012,
                      wspace=0.03, hspace=0.05)
    axs = [[fig.add_subplot(gs[r, c]) for c in range(nc)] for r in range(len(rows))]
    for r, row in enumerate(rows):
        for c, k in enumerate(row["ks"]):
            thumb(axs[r][c], row["run"].frame(k, row["i"]))
    arrow_header(fig, axs[0], W, H, band, a.left, a.right, fs, middle=a.middle or None,
                 middle_color=P.C_PSNAP if da else P.INK, middle_bold=da)
    out = out_dir(a.out, a.name)
    P.save(fig, out, a.name)
    plt.close(fig)
    write_json({"figure": "Fig. 1", "style": a.style, "width_in": W, "rows": [
        {"run": r["run"].path, "method": r["run"].method, "sigma": r["run"].meta.get("sigma"),
         "gamma": r["run"].meta.get("gamma"), "image": r["run"].names[r["i"]], "cycles": r["ks"]} for r in rows]},
        out, "figure.json")
    print(f"[fig1] -> {out}")
    return out


if __name__ == "__main__":
    main()
