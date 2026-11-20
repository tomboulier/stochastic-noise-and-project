# -*- coding: utf-8 -*-
"""Calibrate the detection threshold of a classifier once, on REAL images (Youden's index).

    python scripts/calibrate_classifier.py --classifier weights/medmnist \\
        --negatives data/chestmnist224/healthy_test.npy \\
        --positives Cardiomegaly=data/chestmnist224/cardiomegaly_test.npy,Effusion=data/chestmnist224/effusion_test.npy,Edema=data/chestmnist224/edema_test.npy \\
        --out calibration/medmnist_r18_224_test.json

For each head: positives = images carrying this pathology only, negatives = healthy images (all of them,
as in the paper: the whole test split).  An image is detected when p > threshold.  The thresholds are
never re-optimised on projected images.  Writes a new JSON file:
    {"classifier": ..., "heads": {"Cardiomegaly": {"index", "threshold", "youden_J", "auc", "n_pos", "n_neg",
     "rate_pos", "rate_neg"}, ...}}
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snap.evaluation import load_classifier, youden_threshold, auroc, rate, label_index  # noqa: E402
from snap.images import load_images  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifier", required=True)
    p.add_argument("--positives", required=True, help="Head=path,Head=path")
    p.add_argument("--negatives", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: nothing is overwritten")
    clf = load_classifier(a.classifier, a.device)
    xn, _ = load_images(a.negatives)
    pn = clf.probs(xn)
    res = {"classifier": a.classifier, "negatives": a.negatives, "heads": {}}
    for item in a.positives.split(","):
        head, path = item.split("=", 1)
        j = label_index(head, clf.labels)
        xp, _ = load_images(path)
        pp = clf.probs(xp)[:, j]
        thr, J = youden_threshold(pp, pn[:, j])
        res["heads"][head] = {"index": j, "threshold": thr, "youden_J": J, "auc": auroc(pp, pn[:, j]),
                              "n_pos": int(len(pp)), "n_neg": int(len(pn)), "rate_pos": rate(pp, thr),
                              "rate_neg": rate(pn[:, j], thr), "positives": path}
        print(f"[{head}] AUC {res['heads'][head]['auc']:.3f} | threshold {thr:.4f} | detected: "
              f"{rate(pp, thr):.2f} of positives, {rate(pn[:, j], thr):.2f} of negatives")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "x") as f:
        json.dump(res, f, indent=1)
    print(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()
