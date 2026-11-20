# -*- coding: utf-8 -*-
"""Evaluation: a pathology classifier read at a calibrated threshold, MSE, LPIPS, bootstrap intervals.

The methods never see the classifier: it only measures whether the pathology has been removed.

    Classifier            probs(x) -> numpy (N, n_classes), x (N, C, H, W) in [-1, 1]
    MedMNISTClassifier    the ResNet-18 / ResNet-50 of the MedMNIST v2 benchmark (published weights),
                          224-branch read on the 28 x 28 image ("area28": area down-sampling to 28, 8-bit
                          quantisation, nearest up-sampling to 224, as at its training) -- the paper's judge
    load_classifier       a MedMNIST weight file / folder, or 'my_classifier.py:build'
    youden_threshold      threshold maximising sensitivity + specificity - 1, calibrated once on REAL
                          images (single-label positives against healthy images), never on outputs
"""

import contextlib
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["Classifier", "MedMNISTClassifier", "load_classifier", "CHESTMNIST_LABELS", "youden_threshold",
           "auroc", "rate", "lpips_distance", "bootstrap_ci"]

CHESTMNIST_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                     "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]


def label_index(name, labels=CHESTMNIST_LABELS):
    key = str(name).lower().replace("_", " ")
    for i, n in enumerate(labels):
        if n.lower() == key:
            return i
    return int(name)


# --------------------------------------------------------------------------- classifiers
class Classifier:
    """Base class: probs(x) -> numpy (N, n_classes) for x in [-1, 1] on any device."""

    labels = CHESTMNIST_LABELS

    def probs(self, x):
        raise NotImplementedError

    def head(self, name):
        """f(x) -> probability of one class, numpy (N,): a monitor for scripts/project.py."""
        j = label_index(name, self.labels)
        return lambda x: self.probs(x)[:, j]


def _unwrap(obj):
    if isinstance(obj, dict) and "net" in obj and isinstance(obj["net"], dict):
        return obj["net"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _medmnist_net(state):
    """torchvision ResNet (224 branch, key fc.weight) from a benchmark state dict."""
    from torchvision.models import resnet18, resnet50
    if "fc.weight" not in state:
        raise ValueError("not a 224-branch MedMNIST weight file (no fc.weight)")
    arch = resnet50 if "layer1.0.conv3.weight" in state else resnet18
    net = arch(weights=None, num_classes=int(state["fc.weight"].shape[0]))
    net.load_state_dict(state, strict=True)
    return net, int(state["conv1.weight"].shape[1])


class MedMNISTClassifier(Classifier):
    """Ensemble (mean of sigmoids) of MedMNIST v2 benchmark networks of the 224 branch.

    The 224-branch networks were trained on 28 x 28 images enlarged x8 by nearest neighbour, so an
    image of any size is read on its 28 x 28 version ("area28").  fp16 autocast on CUDA, as in the paper.
    """

    def __init__(self, weight_files, device=None, fp16=True, batch=128):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.nets, self.in_ch = [], []
        for f in weight_files:
            net, c = _medmnist_net(_unwrap(torch.load(f, map_location="cpu", weights_only=False)))
            self.nets.append(net.eval().requires_grad_(False).to(self.device))
            self.in_ch.append(c)
        if not self.nets:
            raise FileNotFoundError("no MedMNIST weight file given")
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.batch = int(batch)
        self.files = list(weight_files)

    def _input(self, x):
        x01 = ((x.to(self.device).float() + 1.0) / 2.0).clamp(0.0, 1.0)
        x28 = x01 if x01.shape[-1] == 28 else F.interpolate(x01, size=(28, 28), mode="area")
        x28 = torch.round(x28.clamp(0.0, 1.0) * 255.0) / 255.0
        z = (F.interpolate(x28, size=(224, 224), mode="nearest") - 0.5) / 0.5
        return z.repeat(1, 3, 1, 1) if self.in_ch[0] == 3 and z.shape[1] == 1 else z

    @torch.no_grad()
    def probs(self, x):
        ctx = torch.autocast("cuda", dtype=torch.float16) if self.fp16 else contextlib.nullcontext()
        out = []
        for s in range(0, x.shape[0], self.batch):
            z = self._input(x[s:s + self.batch])
            with ctx:
                p = torch.stack([torch.sigmoid(n(z).float()) for n in self.nets], 0).mean(0)
            out.append(p.float().cpu())
        return torch.cat(out, 0).numpy()


def load_classifier(spec, device=None, arch="resnet18"):
    """'a.pth[,b.pth]' or a folder of *.pth (MedMNIST benchmark, 224 branch, architecture `arch`; several
    files = an ensemble), or 'my_classifier.py:build' (a function build(device) returning a Classifier)."""
    if ".py:" in str(spec):
        import importlib.util
        fname, _, func = str(spec).rpartition(":")
        ms = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(fname))[0], fname)
        mod = importlib.util.module_from_spec(ms)
        ms.loader.exec_module(mod)
        return getattr(mod, func)(device=device)
    files = []
    for part in str(spec).split(","):
        files += sorted(glob.glob(os.path.join(part, "**", "*.pth"), recursive=True)) if os.path.isdir(part) else [part]
    keep = []
    for f in files:                                    # the 224 branch only (the 28 branch has linear.weight)
        try:
            st = _unwrap(torch.load(f, map_location="cpu", weights_only=False))
        except Exception:                              # noqa: BLE001
            continue
        if "fc.weight" not in st:
            continue
        a = "resnet50" if "layer1.0.conv3.weight" in st else "resnet18"
        if arch is None or a == arch:
            keep.append(f)
    return MedMNISTClassifier(keep, device)


# --------------------------------------------------------------------------- statistics
def auroc(pos, neg):
    """P(pos > neg) + P(pos = neg) / 2."""
    pos, neg = np.asarray(pos, np.float64).ravel(), np.sort(np.asarray(neg, np.float64).ravel())
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def youden_threshold(pos, neg):
    """(threshold, J): the midpoint between observed scores (or 0, 1) that maximises tpr - fpr, where an
    image is detected when its score is STRICTLY above the threshold; the first maximum is kept."""
    pos, neg = np.asarray(pos, float).ravel(), np.asarray(neg, float).ravel()
    cand = np.unique(np.concatenate([pos, neg]))
    if len(cand) == 0:
        return 0.5, float("nan")
    thr = np.concatenate([[0.0], (cand[:-1] + cand[1:]) / 2 if len(cand) > 1 else [], [1.0]])
    sp, sn = np.sort(pos), np.sort(neg)
    tpr = (len(sp) - np.searchsorted(sp, thr, "right")) / len(sp)
    fpr = (len(sn) - np.searchsorted(sn, thr, "right")) / len(sn)
    j = tpr - fpr
    k = int(j.argmax())
    return float(thr[k]), float(j[k])


def load_thresholds(path):
    """{head: threshold} from a file written by scripts/calibrate_classifier.py."""
    import json
    with open(path) as f:
        return {h: float(v["threshold"]) for h, v in json.load(f)["heads"].items()}


def rate(scores, thr):
    """Fraction of scores strictly above thr (detection rate)."""
    s = np.asarray(scores, float)
    return float((s > thr).mean()) if s.size else float("nan")


def bootstrap_ci(values, n_boot=1000, level=0.95, seed=0):
    """Percentile bootstrap interval of the mean over the LAST axis (the images are resampled).
    values (..., N) -> (low, high), each of shape (...)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.shape[-1], (n_boot, v.shape[-1]))
    means = v[..., idx].mean(-1)                                  # (..., n_boot)
    a = (1 - level) / 2
    return np.quantile(means, a, axis=-1), np.quantile(means, 1 - a, axis=-1)


_LPIPS = {}


def lpips_distance(x, y, net="alex", device=None, batch=32):
    """LPIPS (Zhang et al. 2018, pip install lpips) between images in [-1, 1], per image -> numpy (N,)."""
    import lpips
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (net, str(device))
    if key not in _LPIPS:
        _LPIPS[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    f = _LPIPS[key]
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            out.append(f(x[s:s + batch].to(device).float(), y[s:s + batch].to(device).float()).flatten().cpu())
    return torch.cat(out).numpy()
