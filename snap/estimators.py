# -*- coding: utf-8 -*-
"""Few-shot estimators of the distance to the model's domain (estimator-driven SNAP).

The distance delta of an image to the manifold cannot be read from the model.  With N
images of the source domain (e.g. pathological radiographs) and many images of the model's
domain (healthy radiographs), we learn a score s(x), larger for source-like images, used as a
monotone proxy of delta.  Two estimators:

    LDAEstimator  shrinkage linear discriminant on the activations of one block of the frozen
                  U-Net at a small noise level tau_f, spatially average-pooled p x p:
                      w = (Sigma_h + gamma I)^-1 (mu_s - mu_h),  s(x) = <phi(x) - mu_h, w> / s_h
                  (z-scored features).  Block, pooling, tau_f and gamma are chosen by
                  cross-validation INSIDE the N source images (and the training in-domain images).
    CNNEstimator  small CNN trained on the same N images against in-domain images: four
                  convolutional blocks, strong geometric augmentation, a fixed number of steps
                  (no early stopping: nothing is held out of the N), ensemble of three seeds.

Both are calibrated by two levels: the scores of held-out in-domain images (the stopping
threshold is one of their quantiles, the median by default) and the median score of the N
source images.  Both score images quantised to 8 bits, like the images they were trained on.
"""

import math
import os

import numpy as np
import torch

__all__ = ["Estimator", "CNNEstimator", "LDAEstimator", "FunctionEstimator", "train_cnn", "fit_lda",
           "load_estimator", "auroc", "quantize"]


def quantize(x):
    """Round to the 8-bit grid (images are stored as uint8)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round() / 127.5 - 1


def auroc(pos, neg):
    """Area under the ROC curve, P(pos > neg) + P(pos = neg) / 2."""
    pos, neg = np.asarray(pos, np.float64).ravel(), np.sort(np.asarray(neg, np.float64).ravel())
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def _save_new(obj, path):
    """torch.save to a file that must not exist yet."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "xb") as f:
        torch.save(obj, f)


class Estimator:
    """score(x) -> numpy (B,) for x (B, C, H, W) in [-1, 1] on the device; larger = farther from the domain.

    cost          network calls of the diffusion model per scored image (counted in the budget)
    calibration   {"healthy_scores": scores of held-out in-domain images, "source_median": float}
    """

    kind = "base"

    def __init__(self):
        self.cost = 0
        self.calibration = {}
        self.meta = {}
        self.device = None

    def score(self, x):
        raise NotImplementedError

    def score_all(self, x, batch=512):
        """Scores of a (CPU) tensor of images, in chunks."""
        mv = (lambda t: t.to(self.device)) if self.device is not None else (lambda t: t)
        return np.concatenate([self.score(mv(x[i:i + batch])) for i in range(0, len(x), batch)])

    def calibrate(self, source, healthy_val, batch=512):
        """Store the two calibration levels (source: the N training source images)."""
        self.calibration = {"healthy_scores": self.score_all(healthy_val, batch).astype(np.float32).tolist(),
                            "source_median": float(np.median(self.score_all(source, batch)))}
        return self

    def healthy_threshold(self, q=0.5):
        """Stopping threshold: quantile q of the held-out in-domain scores."""
        if "healthy_scores" not in self.calibration:
            raise ValueError("estimator not calibrated: call calibrate(source, healthy_val)")
        return float(np.quantile(np.asarray(self.calibration["healthy_scores"], np.float32), q))

    def source_level(self):
        return float(self.calibration["source_median"])

    def save(self, path):
        _save_new({"kind": self.kind, "state": self.state(), "calibration": self.calibration, "meta": self.meta}, path)
        return path


class FunctionEstimator(Estimator):
    """Any callable score_fn(x) -> numpy (B,) used as an estimator (e.g. an external classifier)."""

    kind = "function"

    def __init__(self, score_fn, cost=0, calibration=None, device=None):
        super().__init__()
        self.score_fn, self.cost, self.calibration, self.device = score_fn, cost, dict(calibration or {}), device

    def score(self, x):
        return np.asarray(self.score_fn(x), np.float32)


# ============================================================================ CNN
def default_widths(image_size):
    """Four blocks below 128 px; above, a 5 x 5 stride-2 input layer first (the last map stays small)."""
    return (32, 64, 128, 256) if int(image_size) < 128 else (16, 32, 64, 128, 256)


def cnn(widths, channels=1):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1),
                             nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    widths = tuple(widths)
    entry = []
    if len(widths) == 5:
        entry = [nn.Conv2d(channels, widths[0], 5, stride=2, padding=2), nn.BatchNorm2d(widths[0]), nn.ReLU()]
        w = widths
    elif len(widths) == 4:
        w = (channels,) + widths
    else:
        raise ValueError("widths: four or five values")
    return nn.Sequential(*entry, *[block(w[i], w[i + 1]) for i in range(4)], nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(w[-1], 1))


def augment(x, g):
    """Strong augmentation: rotation +-7 deg, scale 0.9-1.1, translation, contrast +-20 %, brightness +-0.15, noise 0.03."""
    import torch.nn.functional as F
    B, dev = x.shape[0], x.device
    ang = (2 * torch.rand(B, generator=g, device=dev) - 1) * math.radians(7)
    sc = 1 + 0.1 * (2 * torch.rand(B, generator=g, device=dev) - 1)
    tx = (2 * torch.rand(B, generator=g, device=dev) - 1) * (6 / 32)
    ty = (2 * torch.rand(B, generator=g, device=dev) - 1) * (6 / 32)
    th = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1),
                      torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    a = 1 + 0.2 * (2 * torch.rand(B, 1, 1, 1, generator=g, device=dev) - 1)
    b = 0.15 * (2 * torch.rand(B, 1, 1, 1, generator=g, device=dev) - 1)
    return (a * x + b + 0.03 * torch.randn(x.shape, generator=g, device=dev)).clamp(-1, 1)


def _train_one(xs, xh, widths, steps, seed, device, half_batch=32, lr=1e-3, weight_decay=1e-3, log_every=0):
    import torch.nn as nn
    torch.manual_seed(seed)
    net = cnn(widths, xs.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    Xs, Xh = xs.to(device), xh.to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    lossf = nn.BCEWithLogitsLoss()
    y = torch.cat([torch.ones(half_batch, device=device), torch.zeros(half_batch, device=device)])
    for step in range(int(steps)):
        net.train()
        i_s = torch.randint(0, len(Xs), (half_batch,), generator=g, device=device)
        i_h = torch.randint(0, len(Xh), (half_batch,), generator=g, device=device)
        loss = lossf(net(augment(torch.cat([Xs[i_s], Xh[i_h]]), g))[:, 0], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"[cnn] seed {seed} step {step + 1}/{steps} loss {loss.item():.3f}")
    return net.eval()


class CNNEstimator(Estimator):
    """Mean logit of an ensemble of small CNNs."""

    kind = "cnn"

    def __init__(self, nets, widths, channels, device):
        super().__init__()
        self.nets, self.widths, self.channels, self.device = list(nets), tuple(widths), int(channels), device
        self.cost = 0

    @torch.no_grad()
    def score(self, x):
        x = quantize(x)
        chunk = max(64, 1024 * 4096 // int(np.prod(x.shape[-2:])))
        out = [torch.cat([n(x[s:s + chunk])[:, 0] for s in range(0, len(x), chunk)]).cpu().numpy() for n in self.nets]
        return np.mean(out, 0)

    def state(self):
        return {"widths": list(self.widths), "channels": self.channels,
                "nets": [{k: v.cpu() for k, v in n.state_dict().items()} for n in self.nets]}

    @classmethod
    def from_state(cls, st, device):
        nets = []
        for sd in st["nets"]:
            n = cnn(st["widths"], st["channels"]).to(device)
            n.load_state_dict(sd)
            nets.append(n.eval())
        return cls(nets, st["widths"], st["channels"], device)


def train_cnn(source, healthy, steps=1000, seeds=(0, 1, 2), widths=None, device=None, log_every=0):
    """CNN ensemble.  source (N, C, H, W) and healthy (M, C, H, W) in [-1, 1], 8-bit images."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    widths = tuple(widths or default_widths(source.shape[-1]))
    nets = [_train_one(source, healthy, widths, steps, int(s), device, log_every=log_every) for s in seeds]
    est = CNNEstimator(nets, widths, source.shape[1], device)
    est.meta = {"n_source": int(len(source)), "n_healthy": int(len(healthy)), "steps": int(steps),
                "seeds": [int(s) for s in seeds], "widths": list(widths)}
    return est


# ============================================================================ LDA on U-Net activations
class Activations:
    """Outputs of named blocks of the diffusion network, captured by forward hooks (registered once)."""

    def __init__(self, model, blocks):
        self.model, self.blocks = model, tuple(blocks)
        mods = model.blocks()
        missing = [b for b in self.blocks if b not in mods]
        if missing:
            raise ValueError(f"unknown blocks {missing}; available: {list(mods)}")
        reg = getattr(model, "_snap_hooks", None)
        if reg is None:
            reg = model._snap_hooks = {"out": {}, "on": False, "names": set()}
        for b in self.blocks:
            if b not in reg["names"]:
                mods[b].register_forward_hook(
                    lambda m, i, o, b=b: reg["out"].__setitem__(b, o[0] if isinstance(o, tuple) else o) if reg["on"] else None)
                reg["names"].add(b)
        self.reg = reg

    @torch.no_grad()
    def extract(self, x, tau_f, n_f, pools, generator):
        """{(block, p): (B, C, p, p) float32}, averaged over n_f noise draws at level tau_f (tau_f = 0: the image)."""
        import torch.nn.functional as F
        m = self.model
        reps = 1 if int(tau_f) == 0 else int(n_f)
        ab = float(m.alphas_cumprod[int(tau_f)]) if int(tau_f) > 0 else 1.0
        acc = {}
        self.reg["on"] = True
        try:
            for _ in range(reps):
                xt = x if int(tau_f) == 0 else (math.sqrt(ab) * x + math.sqrt(1 - ab) *
                                                torch.randn(x.shape, device=x.device, generator=generator))
                m.forward(xt, int(tau_f))
                for b in self.blocks:
                    h = self.reg["out"][b].float()
                    for p in pools:
                        acc[(b, p)] = acc.get((b, p), 0) + F.adaptive_avg_pool2d(h, int(p)) / reps
        finally:
            self.reg["on"] = False
        return acc


def _features(act, x, tau_f, n_f, pools, device, batch, seed):
    g = torch.Generator(device=device).manual_seed(int(seed))
    out = {}
    for s in range(0, len(x), batch):
        for k, v in act.extract(quantize(x[s:s + batch].to(device)), tau_f, n_f, pools, g).items():
            out.setdefault(k, []).append(v.flatten(1).cpu().half())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def lda_gram(H, S, gammas, standardize, device):
    """Shrinkage LDA through the Gram matrix of the in-domain features (n << D).
    H (n, D) in-domain, S (N, D) source -> (mu, sd, [(w, scale) per gamma]); w unit-norm, scale = std of healthy scores."""
    H = torch.as_tensor(H, dtype=torch.float64, device=device)
    S = torch.as_tensor(S, dtype=torch.float64, device=device)
    mu = H.mean(0)
    sd = H.std(0) + 1e-6 if standardize else torch.ones_like(mu)
    Z = (H - mu) / sd
    dm = ((S - mu) / sd).mean(0)
    n, D = Z.shape
    ev, U = torch.linalg.eigh(Z @ Z.T)
    keep = ev > ev.max() * 1e-10
    ev, U = ev[keep], U[:, keep]
    V = (U.T @ Z) / ev.sqrt()[:, None]                  # principal directions (q, D)
    lam = ev / (n - 1)
    tot = float((Z ** 2).sum() / (n - 1))
    proj = V @ dm
    res = []
    for g in gammas:
        gam = g * tot / D
        w = V.T @ (proj / (lam + gam)) + (dm - V.T @ proj) / gam
        w = w / w.norm()
        res.append((w.float().cpu().numpy(), float((Z @ w).std())))
    return mu.float().cpu().numpy(), sd.float().cpu().numpy(), res


class LDAEstimator(Estimator):
    """s(x) = <(phi(x) - mu) / sd, w> / scale, phi = block activations at tau_f, pooled p x p."""

    kind = "lda"

    def __init__(self, model, block, pool, tau_f, n_f, mu, sd, w, scale, batch=None):
        super().__init__()
        self.model, self.device = model, model.device
        self.block, self.pool, self.tau_f, self.n_f = str(block), int(pool), int(tau_f), int(n_f)
        self.mu, self.sd, self.w = (torch.as_tensor(np.asarray(a, np.float32), device=self.device) for a in (mu, sd, w))
        self.scale = float(scale)
        self.act = Activations(model, [self.block])
        self.cost = 1 if self.tau_f == 0 else self.n_f
        self.batch = batch or (32 if model.image_size >= 128 else 512)

    @torch.no_grad()
    def score(self, x):
        g = torch.Generator(device=self.device).manual_seed(12345)        # the same noise at every call
        out = []
        for s in range(0, len(x), self.batch):
            F = self.act.extract(quantize(x[s:s + self.batch]), self.tau_f, self.n_f, (self.pool,), g)
            out.append(((((F[(self.block, self.pool)].flatten(1) - self.mu) / self.sd) @ self.w) / self.scale).cpu().numpy())
        return np.concatenate(out)

    def state(self):
        return {"block": self.block, "pool": self.pool, "tau_f": self.tau_f, "n_f": self.n_f,
                "mu": self.mu.cpu().numpy(), "sd": self.sd.cpu().numpy(), "w": self.w.cpu().numpy(), "scale": self.scale}

    @classmethod
    def from_state(cls, st, model):
        return cls(model, st["block"], st["pool"], st["tau_f"], st["n_f"], st["mu"], st["sd"], st["w"], st["scale"])


def default_blocks(model):
    """All blocks of the U-Net but the full-resolution ones (first down block, last up block)."""
    names = list(model.blocks())
    ups = [n for n in names if n.startswith("up")]
    return [n for n in names if n != "down0" and n != (ups[-1] if ups else None)]


def fit_lda(model, source, healthy, blocks=None, pools=(1, 2, 4), taus_f=(0, 20, 50, 100, 200), n_f=4,
            gammas=(0.01, 0.03, 0.1, 0.3, 1, 3, 10), folds=5, standardize=True, seed=0, batch=None, log=print):
    """Choose (block, pool, tau_f, gamma) by k-fold cross-validation inside the N source images (AUROC of the
    out-of-fold scores against out-of-fold in-domain images), then refit on everything.  -> (LDAEstimator, grid rows)."""
    device = model.device
    blocks = list(blocks or default_blocks(model))
    batch = batch or (32 if model.image_size >= 128 else 256)
    ns, nh = len(source), len(healthy)
    k = max(2, min(int(folds), ns))
    rng = np.random.default_rng(seed)
    fs, fh = rng.permutation(ns) % k, rng.permutation(nh) % k
    act = Activations(model, blocks)
    rows, best, feats_best = [], None, None
    for tf in taus_f:
        Fs = _features(act, source, tf, n_f, pools, device, batch, seed + 1000 + int(tf))
        Fh = _features(act, healthy, tf, n_f, pools, device, batch, seed + 2000 + int(tf))
        for b in blocks:
            for p in pools:
                S, H = Fs[(b, p)].astype(np.float32), Fh[(b, p)].astype(np.float32)
                oof_s = np.zeros((len(gammas), ns))
                oof_h = np.zeros((len(gammas), nh))
                for f in range(k):
                    mu, sd, res = lda_gram(H[fh != f], S[fs != f], gammas, standardize, device)
                    for j, (w, e) in enumerate(res):
                        oof_s[j, fs == f] = ((S[fs == f] - mu) / sd) @ w / e
                        oof_h[j, fh == f] = ((H[fh == f] - mu) / sd) @ w / e
                for j, gam in enumerate(gammas):
                    r = {"block": b, "pool": int(p), "tau_f": int(tf), "gamma": float(gam), "dim": int(S.shape[1]),
                         "auc_cv": auroc(oof_s[j], oof_h[j])}
                    rows.append(r)
                    if best is None or r["auc_cv"] > best["auc_cv"]:
                        best, feats_best = r, (S, H)
        log(f"[lda] tau_f {tf}: best so far {best['block']} p{best['pool']} tau_f {best['tau_f']} "
            f"gamma {best['gamma']:g} -> CV AUROC {best['auc_cv']:.3f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    S, H = feats_best
    mu, sd, res = lda_gram(H, S, [best["gamma"]], standardize, device)
    w, e = res[0]
    est = LDAEstimator(model, best["block"], best["pool"], best["tau_f"], n_f, mu, sd, w, e)
    est.meta = {"n_source": ns, "n_healthy": nh, "folds": k, "selected": best, "standardize": bool(standardize)}
    return est, rows


# ============================================================================ loading
def load_estimator(path, model=None, device=None):
    """An estimator saved by Estimator.save.  The LDA estimator needs the diffusion model it was fitted on."""
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                   # torch < 1.13
        d = torch.load(path, map_location="cpu")
    if d["kind"] == "cnn":
        device = torch.device(device or (model.device if model is not None else
                                         ("cuda" if torch.cuda.is_available() else "cpu")))
        est = CNNEstimator.from_state(d["state"], device)
    elif d["kind"] == "lda":
        if model is None:
            raise ValueError("the LDA estimator reads the diffusion model's activations: pass the model")
        est = LDAEstimator.from_state(d["state"], model)
    else:
        raise ValueError(f"unknown estimator kind {d['kind']!r}")
    est.calibration, est.meta = d.get("calibration", {}), d.get("meta", {})
    return est
