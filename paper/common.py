# -*- coding: utf-8 -*-
"""Helpers shared by the paper's figure scripts (paper/fig*.py).

The scripts only READ run folders written by scripts/project.py (snap/runs.py) and draw; nothing is
recomputed and nothing is overwritten (every call writes into a new dated folder).

Two drawing styles:
    da        the common style of the paper's figures (snap/plotting.py): sans-serif, drawn at the printed width
    printed   the earlier versions of Figs. 1, 3, 5 and 7 (serif STIX labels for 1, 3 and 7, smaller labels for 5)
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from snap import plotting as P  # noqa: E402
from snap.images import new_run_dir  # noqa: E402
from snap.runs import Run  # noqa: E402

__all__ = ["P", "Run", "pyplot", "thumb", "grid", "arrow_header", "image_index", "ints", "check_frames",
           "same_input", "mse_u8", "out_dir", "write_json", "STYLES"]

STYLES = ("da", "printed")


def pyplot(serif=False):
    """matplotlib in the paper style (snap.plotting.apply: fonts, line widths, 300 dpi); serif=True: STIX labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P.apply(plt)
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})
    if serif:
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    return plt


def thumb(ax, u8, frame=None, lw=0.6):
    """A (C, H, W) uint8 image without axes, Lanczos-resampled; frame: colour of a frame of width lw (points)."""
    if u8.shape[0] == 1:
        ax.imshow(u8[0], cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    else:
        ax.imshow(np.transpose(u8, (1, 2, 0)), interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        if frame is None:
            sp.set_visible(False)
        else:
            sp.set_visible(True)
            sp.set_color(frame)
            sp.set_linewidth(lw)


def grid(plt, width, rows, nc, left, right, top, bottom, wspace=0.04, hspace=0.04):
    """A figure whose image cells are square: rows = relative heights of the rows (1 = one image).
    top, bottom in inches.  -> (fig, gridspec, height in inches)."""
    u = width * (right - left) / (nc + (nc - 1) * wspace)
    mean = float(np.mean(rows))
    H = u * (sum(rows) + (len(rows) - 1) * hspace * mean) + top + bottom
    fig = plt.figure(figsize=(width, H))
    gs = fig.add_gridspec(len(rows), nc, height_ratios=rows, left=left, right=right, top=1 - top / H,
                          bottom=bottom / H, wspace=wspace, hspace=hspace)
    return fig, gs, H


def arrow_header(fig, first_row, width, H, band, left, right, fs, middle=None, middle_color=P.INK,
                 middle_bold=False):
    """Labels above the first and the last column, joined by a grey arrow (band: height of the header in inches);
    optional `middle` text on the arrow."""
    from matplotlib.patches import FancyArrowPatch
    y = 1 - band / H * 0.48
    tx = []
    for ax, txt in ((first_row[0], left), (first_row[-1], right)):
        bb = ax.get_position()
        tx.append(fig.text((bb.x0 + bb.x1) / 2, y, txt, ha="center", va="center", fontsize=fs, color=P.INK))
    rend = fig.canvas.get_renderer()                 # the labels stay inside the figure; the arrow runs between them
    ext = [t.get_window_extent(rend).transformed(fig.transFigure.inverted()) for t in tx]
    if ext[0].x0 < 0.004:
        tx[0].set_ha("left")
        tx[0].set_x(0.004)
    if ext[1].x1 > 0.996:
        tx[1].set_ha("right")
        tx[1].set_x(0.996)
    ext = [t.get_window_extent(rend).transformed(fig.transFigure.inverted()) for t in tx]
    gap = 0.06 / width                               # 0.06 in of white on each side
    x0, x1 = ext[0].x1 + gap, ext[1].x0 - gap
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, color=P.GREY, lw=0.8,
                                   mutation_scale=1, arrowstyle="-|>,head_length=4,head_width=2.2"))
    if middle:
        fig.text((x0 + x1) / 2, y, middle, ha="center", va="center", fontsize=fs, color=middle_color,
                 fontweight="bold" if middle_bold else "normal",
                 bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"))


def ints(s):
    """'0,1,2' -> [0, 1, 2]."""
    return [int(v) for v in str(s).split(",") if v.strip() != ""]


def image_index(run, spec):
    """Index of an image of the run: its name (e.g. 00029), else a position (29)."""
    try:
        i = run.index(str(spec))
    except ValueError:
        raise SystemExit(f"{run.path}: no image named {spec!r} (names: {run.names[:5]} ...)")
    if not 0 <= i < run.n:
        raise SystemExit(f"{run.path}: image {spec} outside the {run.n} images of the run")
    return i


def check_frames(run, ks):
    """Every cycle of ks (k > 0) must be a kept frame of the run (project.py --save_cycles)."""
    kept = [int(k) for k in run["frame_cycles"]]
    miss = [k for k in ks if int(k) != 0 and int(k) not in kept]
    if miss:
        raise SystemExit(f"{run.path}: cycles {miss} not kept (kept: {kept}); rerun project.py with --save_cycles")


def same_input(a, b, i, j=None):
    """The two runs start from the same image (i in a, j in b)."""
    j = i if j is None else j
    if not np.array_equal(a["x0"][i], b["x0"][j]):
        raise SystemExit(f"different inputs: image {i} of {a.path} and image {j} of {b.path}")


def mse_u8(x, x0):
    """MSE between two uint8 images, in [0, 1] (as written in the thumbnails)."""
    return float(np.mean((x.astype(np.float64) / 255.0 - x0.astype(np.float64) / 255.0) ** 2))


def out_dir(root, name):
    """A new folder root/<date-time>_<name>."""
    return new_run_dir(root, name)


def write_json(obj, out, name):
    with open(os.path.join(out, name), "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
