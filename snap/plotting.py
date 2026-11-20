# -*- coding: utf-8 -*-
"""The common style of the paper's figures.

Every figure is drawn at its printed width (CVPR: column 3.28 in, page 6.875 in), so a font size in
points is the size read on the page: titles, row / column / method / domain names 9 pt, ticks and
legends 8 pt, the small p / MSE boxes inside thumbnails 3.4 pt.  Sans-serif font (DejaVu Sans) for text
and mathematics; methods in bold colour (SNAP blue, P-SNAP purple), domains and pathologies in grey
bold, the stopping / last image framed in ink, cells after a stop in light grey.
"""

COLUMN_WIDTH = 3.28             # inches, CVPR \columnwidth
PAGE_WIDTH = 6.875              # inches, CVPR \textwidth
TITLE = 9.0                     # pt
TEXT = 8.0                      # pt
BOX = 3.4                       # pt, p / MSE boxes in thumbnails
C_SNAP = "#2a78d6"
C_PSNAP = "#4a3aa7"
GREY = "#6b6b6b"
INK = "#2b2b2b"
RED = "#c0392b"
LIGHT = "#d9d9d9"
METHODS = {"snap": ("SNAP", C_SNAP), "psnap": ("P-SNAP", C_PSNAP), "snaped": ("SNAPED", C_SNAP),
           "psnaped": ("P-SNAPED", C_PSNAP)}


def apply(plt=None):
    """Set the matplotlib style; returns pyplot."""
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": TEXT,
        "axes.titlesize": TITLE, "axes.labelsize": TEXT, "xtick.labelsize": TEXT - 1, "ytick.labelsize": TEXT - 1,
        "legend.fontsize": TEXT, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300})
    return plt


def box(ax, lines, detected, fs=BOX, corner="bottom"):
    """p / MSE box at the bottom (or top) left of a thumbnail; red background = still detected."""
    y, va = (0.022, "bottom") if corner == "bottom" else (0.978, "top")
    ax.text(0.022, y, "\n".join(lines), transform=ax.transAxes, ha="left", va=va, fontsize=fs, color="white",
            linespacing=1.0, fontweight="bold" if detected else "normal",
            bbox=dict(boxstyle="square,pad=0.10", fc=RED if detected else "black", alpha=0.85 if detected else 0.55, lw=0))


def method_label(fig_or_ax, x, y, key, **kw):
    """Method name in bold, in its colour."""
    name, c = METHODS[key]
    return fig_or_ax.text(x, y, name, fontsize=kw.pop("fontsize", TITLE), color=c, fontweight="bold", **kw)


def thumbnail(ax, u8):
    """Show a (C, H, W) uint8 image without axes."""
    import numpy as np
    a = u8[0] if u8.shape[0] == 1 else np.transpose(u8, (1, 2, 0))
    ax.imshow(a, cmap="gray" if u8.shape[0] == 1 else None, vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, out, name, formats=("pdf", "png")):
    """Save a figure as new files out/name.<fmt> (never over an existing file)."""
    import os
    paths = []
    for fmt in formats:
        p = os.path.join(out, f"{name}.{fmt}")
        with open(p, "xb") as f:
            fig.savefig(f, format=fmt, bbox_inches=None)
        paths.append(p)
    return paths
