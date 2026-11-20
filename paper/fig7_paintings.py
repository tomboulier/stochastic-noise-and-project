# -*- coding: utf-8 -*-
"""Fig. 7: painted portraits and their images along the first P-SNAP cycles; column width.

One row per painting (input, then the cycles), labels "Paintings" -> "Projected images" joined by an arrow with
"P-SNAP" on it.  Reads one run folder of scripts/project.py.

Run of the paper (FFHQ 256 DDPM, the 64 test paintings, P-SNAP at sigma = 2.04, 4 cycles):

    python scripts/project.py --model weights/ffhq.pt --config configs/ffhq_256_adm.json --input data/paintings_test \\
        --method psnap --sigma 2.04 --gamma 3 --h_rel 0.003 --cycles 4

Paper version (paintings #26, #28, #41):

    python paper/fig7_paintings.py --run <run folder> --images 26,28,41 --cycles 0,1,2,3,4 --out figures

--cycle_titles adds "input", "k = 1" ... above the columns, under the arrow.  --style printed (default): the version printed in the paper
(serif labels, 3.3 in wide, 7.5 pt, method name in black).  Writes figures/<date-time>_fig7_paintings/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import P, Run, STYLES, arrow_header, check_frames, grid, image_index, ints, out_dir, pyplot, thumb, write_json  # noqa: E402,E501


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder (P-SNAP on paintings)")
    p.add_argument("--images", default="26,28,41", help="images shown, one row each (names or positions in the run)")
    p.add_argument("--cycles", default="0,1,2,3,4", help="columns (0 = the input)")
    p.add_argument("--left", default="Paintings")
    p.add_argument("--middle", default="P-SNAP")
    p.add_argument("--right", default="Projected images")
    p.add_argument("--cycle_titles", action="store_true", help="cycle above each column")
    p.add_argument("--style", choices=STYLES, default="printed")
    p.add_argument("--name", default="fig7_paintings")
    p.add_argument("--out", default="figures")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    run = Run(a.run)
    idx = [image_index(run, s) for s in a.images.split(",") if s.strip()]
    ks = ints(a.cycles)
    check_frames(run, ks)
    da = a.style == "da"
    plt = pyplot(serif=not da)
    W, fs = (P.COLUMN_WIDTH, P.TITLE) if da else (3.3, 7.5)
    nc = len(ks)
    band = fs * 1.55 / 72                                         # the arrow line
    top = band + (fs * 1.25 / 72 if a.cycle_titles else 0)        # + the cycle titles
    fig, gs, H = grid(plt, W, [1] * len(idx), nc, left=0.006, right=0.994, top=top, bottom=0.012,
                      wspace=0.03, hspace=0.03)
    axs = [[fig.add_subplot(gs[r, c]) for c in range(nc)] for r in range(len(idx))]
    for r, i in enumerate(idx):
        for c, k in enumerate(ks):
            thumb(axs[r][c], run.frame(k, i))
            if a.cycle_titles and r == 0:
                axs[r][c].set_title("input" if k == 0 else f"$k = {k}$", fontsize=fs - 1.5, pad=2)
    arrow_header(fig, axs[0], W, H, band, a.left, a.right, fs, middle=a.middle or None,
                 middle_color=P.C_PSNAP if da else P.INK, middle_bold=da)
    out = out_dir(a.out, a.name)
    P.save(fig, out, a.name)
    plt.close(fig)
    write_json({"figure": "Fig. 7", "style": a.style, "width_in": W, "run": run.path, "method": run.method,
                "sigma": run.meta.get("sigma"), "gamma": run.meta.get("gamma"), "images": [run.names[i] for i in idx],
                "cycles": ks}, out, "figure.json")
    print(f"[fig7] -> {out}")
    return out


if __name__ == "__main__":
    main()
