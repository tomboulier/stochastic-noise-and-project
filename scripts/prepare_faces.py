# -*- coding: utf-8 -*-
"""Prepare the photographs and painted portraits of the paper as uint8 arrays (N, 3, 256, 256), split into
test / validation / training sets in a fixed order.

    # photographs of faces: the "plain" folder of "Glasses and Coverings" (Table 1). Colour photographs only
    # (mean |R - G| + |G - B| above 4 grey levels on a 64 x 64 thumbnail), shuffled with seed 0: 64 test, 64 val, rest train
    python scripts/prepare_faces.py --input datasets/plain --name photo --color_only 1 --splits test=64,val=64,train=-1

    # painted portraits (SNGFaces; Fig. 7): all images under the folder, sorted by path, shuffled with seed 0
    python scripts/prepare_faces.py --input datasets/500paintings --name painting --recursive 1 --splits test=64,val=64,train=-1

    # CelebA-HQ 256 (figures of drifting / degraded photographs): rows of the Hugging Face parquet file
    # (korexyz/celeba-hq-256x256, split "validation"), shuffled with seed 0, no crop and no JPEG pass
    python scripts/prepare_faces.py --input validation-00000-of-00001.parquet --name celebahq --preprocess none \\
        --splits test=100,val=32,train=16

Order: the files are sorted by path (a parquet file: its row order), filtered (--color_only), then permuted by
numpy.random.default_rng(--seed).permutation (--order shuffle) and cut into the splits in the order given.
Pre-processing (--preprocess crop_jpeg, the default): centred square crop, LANCZOS resize to --size, JPEG
round trip at quality --jpeg (90), the same for every image so that a judge cannot read the file format.

Writes a new folder <out>/<date-time>_<name>/: <name>_<split>.npz (images uint8 (N, 3, S, S), names),
readable by scripts/project.py --input and scripts/degrade_faces.py, and <name>_files.json (source files of each
split, in order; excluded black-and-white images; settings).  Nothing is overwritten.
"""

import argparse
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from snap.images import new_run_dir  # noqa: E402

EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_files(folder, recursive=False):
    """Image files of a folder, sorted by path."""
    if recursive:
        out = [os.path.join(d, f) for d, _, fs in os.walk(folder) for f in fs if f.lower().endswith(EXT)]
    else:
        out = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(EXT)]
    if not out:
        raise SystemExit(f"no image in {folder}")
    return sorted(out, key=lambda p: os.path.relpath(p, folder).replace(os.sep, "/"))


def preprocess(source, size=256, quality=90):
    """Path or PIL image -> uint8 (3, size, size): centred square crop, LANCZOS, JPEG round trip."""
    from PIL import Image
    im = (Image.open(source) if isinstance(source, str) else source).convert("RGB")
    w, h = im.size
    c = min(w, h)
    im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c)).resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), np.uint8).transpose(2, 0, 1).copy()


def resize_only(im, size=256):
    """PIL image -> uint8 (3, size, size): RGB, bicubic resize only if the size differs (CelebA-HQ 256 as is)."""
    from PIL import Image
    im = im.convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, np.uint8).transpose(2, 0, 1).copy()


def is_color(path, threshold):
    """Mean |R - G| + |G - B| (grey levels) on a 64 x 64 thumbnail above the threshold."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), np.float32)
    return float(np.mean(np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2]))) > threshold


def parse_splits(text):
    """'test=64,val=64,train=-1' -> [(name, n)]; -1: all remaining images."""
    out = []
    for item in [s for s in text.split(",") if s.strip()]:
        name, _, n = item.partition("=")
        out.append((name.strip(), int(n)))
    if sum(1 for _, n in out if n < 0) > 1:
        raise SystemExit("--splits: at most one split can take the rest (-1)")
    return out


def stems(paths, root):
    """Image names: the file stem, or the relative path (with '__') when stems collide."""
    s = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(set(s)) == len(s):
        return s
    return [os.path.splitext(os.path.relpath(p, root))[0].replace(os.sep, "__").replace("/", "__") for p in paths]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="folder of images, or a .parquet file with an 'image' column")
    p.add_argument("--name", required=True, help="prefix of the output files, e.g. photo, painting, celebahq")
    p.add_argument("--splits", default="test=64,val=64,train=-1", help="name=count in order; -1: the rest")
    p.add_argument("--order", default="shuffle", choices=("shuffle", "sorted"))
    p.add_argument("--seed", type=int, default=0, help="seed of the permutation (paper: 0)")
    p.add_argument("--recursive", type=int, default=0, help="1: images in sub-folders too (paintings)")
    p.add_argument("--color_only", type=int, default=0, help="1: drop black-and-white photographs (photographs)")
    p.add_argument("--color_threshold", type=float, default=4.0)
    p.add_argument("--preprocess", default="crop_jpeg", choices=("crop_jpeg", "none"))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--jpeg", type=int, default=90, help="quality of the JPEG round trip")
    p.add_argument("--out", default="data/faces")
    a = p.parse_args(argv)
    splits = parse_splits(a.splits)
    from PIL import Image
    excluded, sha = [], None
    if a.input.lower().endswith(".parquet"):
        import pyarrow.parquet as pq
        table = pq.read_table(a.input)
        col = table.column("image").to_pylist()
        sources = [f"row{i:05d}" for i in range(len(col))]
        with open(a.input, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()

        def read(i):
            d = col[i]
            im = Image.open(io.BytesIO(d["bytes"] if isinstance(d, dict) else d))
            return resize_only(im, a.size) if a.preprocess == "none" else preprocess(im, a.size, a.jpeg)
        names_all = sources
    else:
        files = list_files(a.input, bool(a.recursive))
        if a.color_only:
            keep = [f for f in files if is_color(f, a.color_threshold)]
            excluded = sorted(set(files) - set(keep))
            files = keep
        sources = files

        def read(i):
            if a.preprocess == "none":
                with Image.open(sources[i]) as im:
                    return resize_only(im, a.size)
            return preprocess(sources[i], a.size, a.jpeg)
        names_all = stems(files, a.input)
    n = len(sources)
    order = np.random.default_rng(a.seed).permutation(n) if a.order == "shuffle" else np.arange(n)
    need = sum(c for _, c in splits if c >= 0)
    if need > n:
        raise SystemExit(f"{need} images requested, {n} available")
    out = new_run_dir(a.out, a.name)
    info = {"input": a.input, "settings": vars(a), "n_available": n, "excluded_black_and_white": excluded,
            "sha256": sha, "splits": {}}
    start = 0
    for split, c in splits:
        stop = n if c < 0 else start + c
        idx = order[start:stop]
        imgs = np.stack([read(int(i)) for i in idx]) if len(idx) else np.zeros((0, 3, a.size, a.size), np.uint8)
        names = np.array([names_all[int(i)] for i in idx])
        with open(os.path.join(out, f"{a.name}_{split}.npz"), "xb") as f:
            np.savez_compressed(f, images=imgs, names=names)
        info["splits"][split] = {"n": int(len(idx)), "positions": [int(i) for i in idx],
                                 "sources": [sources[int(i)] for i in idx]}
        print(f"[{a.name}] {split}: {len(idx)} images -> {os.path.join(out, f'{a.name}_{split}.npz')}")
        start = stop
    with open(os.path.join(out, f"{a.name}_files.json"), "x", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    if excluded:
        print(f"[{a.name}] {len(excluded)} black-and-white images excluded")
    print(f"[done] -> {out}")
    return out


if __name__ == "__main__":
    main()
