# -*- coding: utf-8 -*-
"""Export ChestMNIST (MedMNIST v2 / MedMNIST+) to the uint8 arrays used by the paper's recipes.

    python scripts/export_chestmnist.py --size 224 --pathologies Cardiomegaly,Effusion,Edema --out data/chestmnist224

For each split (train, val, test) it writes
    healthy_<split>.npy            images whose 14 labels are all 0
    <pathology>_<split>.npy        images carrying this pathology ONLY (single label), e.g. cardiomegaly_test.npy
    *_indices.npy                  their indices in the MedMNIST split
uint8 arrays (N, size, size), SHUFFLED with --seed (default 0) and cut to --max_healthy_<split>: the paper's
image sets are the FIRST images of these arrays (64 test images per pathology, 64 healthy test images, 5000
healthy training images and 2000 healthy validation images for the estimator, N = all single-label training
images: 754 cardiomegalies, 2730 effusions, 479 edemas).  Needs `pip install medmnist` (downloads to ~/.medmnist).
"""

import argparse
import os

import numpy as np

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
          "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def _save_new(a, path):
    with open(path, "xb") as f:
        np.save(f, a)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=224, choices=(28, 64, 128, 224))
    p.add_argument("--pathologies", default="Cardiomegaly,Effusion,Edema", help=", ".join(LABELS))
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--seed", type=int, default=0, help="shuffling seed (paper: 0)")
    p.add_argument("--max_healthy_train", type=int, default=0, help="keep the first N healthy images (0: all)")
    p.add_argument("--max_healthy_val", type=int, default=0)
    p.add_argument("--max_healthy_test", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    import medmnist
    names = [n.lower() for n in LABELS]
    paths = [names.index(s.strip().lower().replace(" ", "_")) for s in a.pathologies.split(",") if s.strip()]
    os.makedirs(a.out, exist_ok=True)
    for split in [s for s in a.splits.split(",") if s]:
        ds = medmnist.ChestMNIST(split=split, download=True, size=a.size)
        lab = np.asarray(ds.labels)
        imgs = np.asarray(ds.imgs)
        groups = {"healthy": np.where(lab.sum(1) == 0)[0]}
        for j in paths:
            groups[LABELS[j].lower()] = np.where((lab[:, j] == 1) & (lab.sum(1) == 1))[0]
        for g, idx in groups.items():
            idx = idx.copy()
            np.random.default_rng(a.seed).shuffle(idx)            # the order of the paper's experiments
            cap = getattr(a, f"max_healthy_{split}", 0) if g == "healthy" else 0
            if cap and cap > 0:
                idx = idx[:cap]
            im = imgs[idx]
            if im.ndim == 4 and im.shape[-1] == 1:
                im = im[..., 0]
            _save_new(im.astype(np.uint8), os.path.join(a.out, f"{g}_{split}.npy"))
            _save_new(idx.astype(np.int64), os.path.join(a.out, f"{g}_{split}_indices.npy"))
            print(f"[{split}] {g}: {len(idx)} images -> {os.path.join(a.out, f'{g}_{split}.npy')}")


if __name__ == "__main__":
    main()
