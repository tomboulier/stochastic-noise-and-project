# -*- coding: utf-8 -*-
"""Degrade clean photographs of faces as in the paper (Table 1): Gaussian blur, JPEG, pixelation, Gaussian noise.

    # the paper's photographs (scripts/prepare_faces.py on the "plain" folder): 100 test photographs = the 64 of
    # photo_test and the first 36 of photo_train, 32 validation photographs = the first 32 of photo_val
    P=data/faces/<date-time>_photo
    python scripts/degrade_faces.py --test $P/photo_test.npz:64,$P/photo_train.npz:36 --val $P/photo_val.npz:32 --level moderate
    python scripts/degrade_faces.py --test $P/photo_test.npz:64,$P/photo_train.npz:36 --val $P/photo_val.npz:32 --level strong

    moderate    blur radius 3 px, JPEG quality 8, pixelation x8, noise 0.1        (snap/degradations.py)
    strong      blur radius 6 px, JPEG quality 4, pixelation x12, noise 0.2
    --blur, --jpeg, --pixelation, --noise override a level's value.

The clean set is the concatenation of --test then --val (each 'path:n' = the first n images of an array or a folder);
the noise of photograph i (its position in that set, 0-based) is drawn with numpy seed 1000 + i.

Writes a new folder <out>/<date-time>_<level>/:
    clean.npz               images uint8 (N, 3, H, W), names test_000 ... test_099, val_000 ... val_031
    <degradation>.npz       the degraded images, names <degradation>-<clean name> (e.g. blur-test_000): the input
                            of scripts/project.py and scripts/run_baseline.py (--input <folder>/blur.npz)
    all.npz                 the four degraded sets concatenated in the order blur, jpeg, pixelation, noise (--all 1):
                            as in the paper's runs, where every degradation went through one generator per setting
    degradations.json       level, parameters, sources, names and split of every image
The names carry the degradation and the split, which paper/table1_degraded_faces.py reads; scripts/evaluate_runs.py
--reference <folder>/clean.npz matches each output to its clean photograph by the part after '-'.
Nothing is overwritten.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from snap.degradations import DEGRADATIONS, LEVELS, PARAMETER, degrade, describe  # noqa: E402
from snap.images import load_uint8, new_run_dir  # noqa: E402


def read_sets(spec):
    """'a.npz:64,b.npz:36' -> (uint8 (N, C, H, W), source names)."""
    imgs, names = [], []
    for item in [s for s in spec.split(",") if s.strip()]:
        path, n = item, None
        head, sep, tail = item.rpartition(":")
        if sep and tail.strip().lstrip("-").isdigit():
            path, n = head, int(tail)
        a, nm = load_uint8(path, 3)
        if a.dtype != np.uint8:
            raise SystemExit(f"{path}: expected uint8 images")
        if a.shape[1] == 1:
            a = np.repeat(a, 3, 1)
        if n is not None and n >= 0:
            if n > len(a):
                raise SystemExit(f"{path}: {n} images requested, {len(a)} available")
            a, nm = a[:n], nm[:n]
        imgs.append(a)
        names += [f"{os.path.basename(path)}:{x}" for x in nm]
    if not imgs:
        return None, []
    return np.concatenate(imgs), names


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True, help="clean test photographs: 'path:n,path:n' (arrays or folders)")
    p.add_argument("--val", default="", help="clean validation photographs, same format")
    p.add_argument("--level", default="moderate", choices=sorted(LEVELS))
    p.add_argument("--degradations", default=",".join(DEGRADATIONS))
    for k in DEGRADATIONS:
        p.add_argument(f"--{k}", type=float, default=None, help=f"{k} parameter ({PARAMETER[k]}), overrides --level")
    p.add_argument("--all", type=int, default=1, help="1: also all.npz, every degradation concatenated in order")
    p.add_argument("--out", default="data/degraded")
    a = p.parse_args(argv)
    degs = [d for d in a.degradations.split(",") if d]
    for d in degs:
        if d not in DEGRADATIONS:
            raise SystemExit(f"unknown degradation {d} (one of {', '.join(DEGRADATIONS)})")
    values = {d: (getattr(a, d) if getattr(a, d) is not None else LEVELS[a.level][d]) for d in degs}
    custom = any(getattr(a, d) is not None for d in degs)
    xt, st = read_sets(a.test)
    xv, sv = read_sets(a.val) if a.val else (None, [])
    clean = xt if xv is None else np.concatenate([xt, xv])
    split = ["test"] * len(xt) + ["val"] * (0 if xv is None else len(xv))
    names = [f"test_{i:03d}" for i in range(len(xt))] + [f"val_{i:03d}" for i in range(len(clean) - len(xt))]
    out = new_run_dir(a.out, a.level + ("_custom" if custom else ""))
    with open(os.path.join(out, "clean.npz"), "xb") as f:
        np.savez_compressed(f, images=clean, names=np.array(names), split=np.array(split))
    every = []
    for d in degs:
        deg = np.stack([degrade(clean[i], d, values[d], i) for i in range(len(clean))])
        with open(os.path.join(out, f"{d}.npz"), "xb") as f:
            np.savez_compressed(f, images=deg, names=np.array([f"{d}-{n}" for n in names]), split=np.array(split))
        print(f"[{d}] {describe(d, values[d])}: {len(deg)} images -> {os.path.join(out, d + '.npz')}")
        every.append(deg)
    if a.all and len(degs) > 1:          # the paper's runs: every degradation through one generator, in this order
        with open(os.path.join(out, "all.npz"), "xb") as f:
            np.savez_compressed(f, images=np.concatenate(every), names=np.array([f"{d}-{n}" for d in degs for n in names]),
                                split=np.array(split * len(degs)))
        print(f"[all] {len(degs) * len(clean)} images ({', '.join(degs)}) -> {os.path.join(out, 'all.npz')}")
    meta = {"level": a.level, "custom": custom,
            "degradations": {d: {"parameter": PARAMETER[d], "value": values[d], "description": describe(d, values[d])}
                             for d in degs},
            "noise_seed": "numpy default_rng(1000 + i), i = position of the photograph in clean.npz",
            "test": a.test, "val": a.val, "n_test": int(len(xt)), "n_val": int(len(clean) - len(xt)),
            "names": names, "split": split, "sources": st + sv}
    with open(os.path.join(out, "degradations.json"), "x", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"[done] {len(xt)} test + {len(clean) - len(xt)} validation photographs, level {a.level} -> {out}")
    return out


if __name__ == "__main__":
    main()
