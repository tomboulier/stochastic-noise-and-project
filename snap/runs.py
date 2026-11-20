# -*- coding: utf-8 -*-
"""The format of a run folder, written by scripts/project.py and read by the paper scripts.

    <run>/run.json          settings: method, sigma / tau or driven range, gamma, cycles, seed, model,
                            input, image names, calibration levels, monitors ...
    <run>/trajectory.npz    x0          uint8 (N, C, H, W)      the inputs
                            frame_cycles int (F,)               cycles at which images were kept
                            frames      uint8 (F, N, C, H, W)   the images at these cycles
                            mse         float32 (K+1, N)        MSE to x0, images in [0, 1], every cycle
                            monitor_cycles int (M,)             cycles at which the monitors were read
                            mon_<name>  float32 (M, N, ...)     e.g. mon_p: classifier probabilities
                            nfe, cycles_run (N,)                network calls and cycles per image
                            (driven)    scores (K+1, N), taus (K, N), stopped (N,)
                            names       (N,) str
    <run>/final/<name>.png  the image after the last cycle

A baseline (SDEdit, AnoDDPM, AutoDDPM, VAE, DIIP; snap/baselines) has one output per SETTING instead of
one per cycle (scripts/run_baseline.py):

    <run>/run.json          method, settings [{"label": "tau=100", ...parameters}], names, input, seed, model ...
    <run>/outputs.npz       x0 uint8 (N, C, H, W); outputs uint8 (S, N, C, H, W); labels (S,) str;
                            mse float32 (S, N) to x0 in [0, 1]; nfe float64 (S,); names (N,);
                            optional mon_<name> float32 (S, N, ...)
"""

import json
import os

import numpy as np

from .images import save_png

__all__ = ["save_run", "load_run", "Run", "save_settings_run", "SettingsRun"]

KEYS_ARRAYS = ("mse", "nfe", "cycles_run", "scores", "taus", "stopped", "monitor_cycles")


def save_run(out, res, meta, names, final_png=True):
    """Write a run folder (out must exist and be empty: see images.new_run_dir)."""
    ks = sorted(res["frames"])
    tab = {"x0": res["x0"], "frame_cycles": np.array(ks, np.int32),
           "frames": (np.stack([res["frames"][k] for k in ks]) if ks else np.zeros((0,) + res["x0"].shape, np.uint8)),
           "names": np.array(names)}
    for k in KEYS_ARRAYS:
        if k in res and res[k] is not None:
            tab[k] = np.asarray(res[k])
    for n, v in res.get("monitors", {}).items():
        tab[f"mon_{n}"] = np.asarray(v, np.float32)
    with open(os.path.join(out, "trajectory.npz"), "xb") as f:
        np.savez_compressed(f, **tab)
    meta = dict(meta)
    meta.update({"n_images": len(names), "frame_cycles": ks,
                 "mean_network_calls": float(np.mean(res["nfe"]))})
    for k in ("s_healthy", "s_source", "last_cycle"):
        if k in res:
            meta[k] = res[k]
    with open(os.path.join(out, "run.json"), "x") as f:
        json.dump(meta, f, indent=1, default=str)
    if final_png:
        from .images import to_uint8
        os.mkdir(os.path.join(out, "final"))
        fin = to_uint8(res["x"])
        for i, n in enumerate(names):
            save_png(fin[i], os.path.join(out, "final", f"{n}.png"))
    return out


class Run:
    """A run folder, read back.  run.meta (dict), run[key] (arrays of trajectory.npz), and helpers."""

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "run.json")) as f:
            self.meta = json.load(f)
        z = np.load(os.path.join(path, "trajectory.npz"), allow_pickle=False)
        self.data = {k: z[k] for k in z.files}
        self.names = [str(n) for n in self.data["names"]]

    def __getitem__(self, k):
        return self.data[k]

    def __contains__(self, k):
        return k in self.data

    @property
    def method(self):
        return self.meta["method"]

    @property
    def n(self):
        return len(self.names)

    def index(self, name_or_index):
        """Image index from a name or an integer position."""
        if isinstance(name_or_index, (int, np.integer)):
            return int(name_or_index)
        s = str(name_or_index)
        return self.names.index(s) if s in self.names else int(s)

    def frame(self, k, i):
        """Image i (C, H, W) uint8 at cycle k: a kept frame; for a driven run, the frozen image after the stop."""
        ks = list(self.data["frame_cycles"])
        if int(k) == 0:
            return self.data["x0"][i]
        if int(k) in ks:
            return self.data["frames"][ks.index(int(k))][i]
        raise KeyError(f"cycle {k} was not kept in {self.path} (kept: {ks})")

    def monitor(self, name, k=None):
        """mon_<name> at cycle k (or all monitored cycles)."""
        v = self.data[f"mon_{name}"]
        if k is None:
            return v
        mc = list(self.data["monitor_cycles"])
        return v[mc.index(int(k))]

    def stop_cycle(self, i):
        """Driven runs: the number of cycles image i ran (its stopping cycle)."""
        return int(self.data["cycles_run"][i])


def save_settings_run(out, x0_u8, outputs_u8, labels, mse, nfe, meta, names, monitors=None):
    """Write a baseline run folder: one output per setting (see the module docstring)."""
    tab = {"x0": np.asarray(x0_u8, np.uint8), "outputs": np.asarray(outputs_u8, np.uint8), "labels": np.array(labels),
           "mse": np.asarray(mse, np.float32), "nfe": np.asarray(nfe, np.float64), "names": np.array(names)}
    for n, v in (monitors or {}).items():
        tab[f"mon_{n}"] = np.asarray(v, np.float32)
    with open(os.path.join(out, "outputs.npz"), "xb") as f:
        np.savez_compressed(f, **tab)
    meta = dict(meta)
    meta.update({"n_images": len(names), "labels": list(labels)})
    with open(os.path.join(out, "run.json"), "x") as f:
        json.dump(meta, f, indent=1, default=str)
    return out


class SettingsRun:
    """A baseline run folder, read back: meta, labels, outputs (S, N, C, H, W), mse (S, N), nfe (S,)."""

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "run.json")) as f:
            self.meta = json.load(f)
        z = np.load(os.path.join(path, "outputs.npz"), allow_pickle=False)
        self.data = {k: z[k] for k in z.files}
        self.names = [str(n) for n in self.data["names"]]
        self.labels = [str(s) for s in self.data["labels"]]

    def __getitem__(self, k):
        return self.data[k]

    def __contains__(self, k):
        return k in self.data

    @property
    def method(self):
        return self.meta["method"]


def load_run(path):
    """A run folder: Run (trajectory.npz) or SettingsRun (outputs.npz)."""
    if os.path.exists(os.path.join(path, "outputs.npz")):
        return SettingsRun(path)
    return Run(path)
