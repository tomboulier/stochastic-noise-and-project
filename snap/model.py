# -*- coding: utf-8 -*-
"""The pre-trained diffusion model, as SNAP sees it.

SNAP needs three things from a pre-trained diffusion model:

    eps(x_t, t)        the noise prediction of the network at integer timestep t;
    alphas_cumprod     the noise schedule it was trained with (alpha_bar_t, t = 0..T-1);
    blocks()           (LDA estimator only) named sub-modules whose outputs serve as features.

Images live in [-1, 1], shape (B, C, H, W).  Noise levels are given either as a timestep tau
or as the variance-exploding level of the paper,

    sigma_tau = sqrt((1 - alpha_bar_tau) / alpha_bar_tau),   x_tau / sqrt(alpha_bar_tau) = x + sigma_tau xi.

`load_model` builds a `DiffusionModel` from
    - a diffusers `UNet2DModel` (folder written by `save_pretrained`, Hub id, or a state dict + JSON),
    - a guided-diffusion (ADM) checkpoint + JSON (the FFHQ model of the paper),
    - your own code: 'my_model.py:build'.
See ADDING_A_MODEL.md.
"""

import contextlib
import json
import math
import os

import numpy as np
import torch

__all__ = ["DiffusionModel", "load_model", "schedule_from_config", "linear_schedule"]


def linear_schedule(beta_start=1e-4, beta_end=0.02, num_train_timesteps=1000):
    """alpha_bar_t of the linear beta schedule of DDPM (Ho et al. 2020), in float64."""
    betas = np.linspace(beta_start, beta_end, int(num_train_timesteps), dtype=np.float64)
    return np.cumprod(1.0 - betas)


def schedule_from_config(cfg=None):
    """(alphas_cumprod float64, prediction_type) from a diffusers scheduler config (dict).

    No config: the linear schedule of DDPM, 1000 steps, epsilon prediction.  Linear and
    scaled-linear schedules are recomputed in float64; any other one is taken from diffusers."""
    cfg = dict(cfg or {})
    T = int(cfg.get("num_train_timesteps", 1000))
    b0, b1 = float(cfg.get("beta_start", 1e-4)), float(cfg.get("beta_end", 0.02))
    kind = cfg.get("beta_schedule", "linear")
    pred = cfg.get("prediction_type", "epsilon")
    if cfg.get("trained_betas") is None and kind == "linear":
        ac = linear_schedule(b0, b1, T)
    elif cfg.get("trained_betas") is None and kind == "scaled_linear":
        ac = np.cumprod(1.0 - np.linspace(b0 ** 0.5, b1 ** 0.5, T, dtype=np.float64) ** 2)
    else:
        from diffusers import DDPMScheduler
        keep = {k: v for k, v in cfg.items() if not k.startswith("_")}
        ac = DDPMScheduler.from_config(keep).alphas_cumprod.double().numpy()
    return ac, pred


class DiffusionModel:
    """A frozen noise-prediction network and its schedule.

    net              a torch module (eval mode, on its device)
    alphas_cumprod   1-D array, alpha_bar_t for t = 0..T-1
    forward_fn       forward_fn(net, x, t) -> raw output; t is a (B,) long tensor.
                     Default: net(x, t).sample (diffusers UNet2DModel).
    prediction_type  "epsilon" (default), "v_prediction" or "sample": converted to epsilon.
    eps_from_clipped_x0
                     False (diffusers' DDIM): each DDIM step uses the network's epsilon.
                     True (guided-diffusion's DDIM): epsilon is recomputed from the clipped x0.
    scalars          "float32": the schedule coefficients sqrt(alpha_bar), sqrt(1 - alpha_bar) are
                     float32 tensors (as in diffusers); "float64": Python floats (as in guided-diffusion).
                     Only the last bits of the results depend on it.
    blocks_fn        blocks_fn(net) -> {name: module}; default: the down / mid / up blocks of a UNet2DModel.
    half             fp16 autocast for the DDIM chain (default: on CUDA).  The one-step denoiser
                     whose Jacobian P-SNAP uses is always evaluated in float32.
    image_size, channels
                     read from net.config when it exists.
    """

    def __init__(self, net, alphas_cumprod, forward_fn=None, prediction_type="epsilon",
                 eps_from_clipped_x0=False, scalars="float32", blocks_fn=None, half=None,
                 image_size=None, channels=None, name=""):
        self.net = net
        self.alphas_cumprod = np.asarray(alphas_cumprod, dtype=np.float64)
        self.T = len(self.alphas_cumprod)
        self.forward_fn = forward_fn or (lambda n, x, t: n(x, t).sample)
        if prediction_type not in ("epsilon", "v_prediction", "sample"):
            raise ValueError(f"unknown prediction_type {prediction_type!r}")
        if scalars not in ("float32", "float64"):
            raise ValueError(f"scalars must be 'float32' or 'float64', not {scalars!r}")
        self.prediction_type = prediction_type
        self.eps_from_clipped_x0 = bool(eps_from_clipped_x0)
        self.scalars = scalars
        self.blocks_fn = blocks_fn
        self.device = next(net.parameters()).device
        self.half = (self.device.type == "cuda") if half is None else bool(half)
        cfg = getattr(net, "config", None)
        self.image_size = int(image_size if image_size is not None else getattr(cfg, "sample_size", 0) or 0)
        self.channels = int(channels if channels is not None else getattr(cfg, "in_channels", 0) or 0)
        self.name = name

    def __repr__(self):
        return (f"DiffusionModel({self.name or type(self.net).__name__}, {self.image_size}px, {self.channels} ch, "
                f"T={self.T}, {self.prediction_type}, {'fp16' if self.half else 'fp32'} on {self.device})")

    # ---------------------------------------------------------------- schedule
    def ab_value(self, t):
        """alpha_bar_t as a Python float (rounded to float32 when scalars == 'float32'); t < 0: 1."""
        t = int(t)
        if t < 0:
            return 1.0
        a = float(self.alphas_cumprod[t])
        return float(np.float32(a)) if self.scalars == "float32" else a

    def coef(self, t):
        """(sqrt(alpha_bar_t), sqrt(1 - alpha_bar_t)): float32 tensors or Python floats (see `scalars`)."""
        a = self.ab_value(t)
        if self.scalars == "float32":
            ab = torch.as_tensor(a, device=self.device, dtype=torch.float32)
            return ab.sqrt(), (1.0 - ab).sqrt()
        return a ** 0.5, (1.0 - a) ** 0.5

    def sigma(self, t):
        """sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t), the noise level of the paper."""
        a = float(self.alphas_cumprod[int(t)])
        return math.sqrt((1.0 - a) / a)

    def tau_for_sigma(self, sigma, stride=10):
        """The timestep, a multiple of `stride`, whose sigma is closest to `sigma` (in log scale)."""
        ts = np.arange(int(stride), self.T, int(stride))
        a = self.alphas_cumprod[ts]
        s = np.sqrt((1.0 - a) / a)
        return int(ts[np.argmin(np.abs(np.log(s) - math.log(float(sigma))))])

    def resolve_tau(self, tau=None, sigma=None, stride=10):
        """tau from either argument (tau wins), as a Python int."""
        if tau is not None:
            return int(tau)
        if sigma is not None:
            return self.tau_for_sigma(sigma, stride)
        raise ValueError("give a noise level: sigma or tau")

    # ---------------------------------------------------------------- network
    def forward(self, x, t, fp32=False):
        """Raw network output at timestep t (int), as float32."""
        tt = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
        ctx = (torch.autocast("cuda", dtype=torch.float16) if (x.is_cuda and self.half and not fp32)
               else contextlib.nullcontext())
        with ctx:
            out = self.forward_fn(self.net, x, tt)
        return out.float()

    def eps(self, x, t, fp32=False):
        """Epsilon prediction at timestep t (int), float32, shape of x."""
        out = self.forward(x, t, fp32)
        if out.shape[1] > x.shape[1]:                # learned variance: the first C channels are epsilon
            out = out[:, :x.shape[1]]
        if self.prediction_type == "epsilon":
            return out
        sa, sb = self.coef(t)
        if self.prediction_type == "v_prediction":
            return sa * out + sb * x.float()
        return (x.float() - sa * out) / sb                                # "sample": out is x0

    def blocks(self):
        """{name: module} whose outputs can serve as features (LDA estimator)."""
        if self.blocks_fn is not None:
            return dict(self.blocks_fn(self.net))
        n = self.net
        if not all(hasattr(n, a) for a in ("down_blocks", "mid_block", "up_blocks")):
            raise AttributeError("no down_blocks / mid_block / up_blocks: pass blocks_fn (see ADDING_A_MODEL.md)")
        out = {f"down{i}": b for i, b in enumerate(n.down_blocks)}
        out["mid"] = n.mid_block
        out.update({f"up{i}": b for i, b in enumerate(n.up_blocks)})
        return out


# -------------------------------------------------------------------- loading
_WEIGHT_EXT = (".pt", ".pth", ".bin", ".ckpt", ".safetensors")


def _read_state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                                 # torch < 1.13
        state = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "ema"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    return state


def _scheduler_config(path):
    """The scheduler config next to a diffusers model (folder or Hub id), or None."""
    from diffusers import DDPMScheduler
    for sub in (None, "scheduler"):
        try:
            return dict(DDPMScheduler.load_config(path, subfolder=sub))
        except Exception:                             # noqa: BLE001 - absent config: default schedule
            continue
    return None


def _from_python(spec, device, half):
    """'file.py:function' -> function(device=..., half=...), which must return a DiffusionModel."""
    import importlib.util
    fname, _, func = spec.rpartition(":")
    mod_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(fname))[0], fname)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    m = getattr(mod, func)(device=device, half=half)
    if not isinstance(m, DiffusionModel):
        raise TypeError(f"{spec} returned {type(m).__name__}, not a DiffusionModel")
    m.name = m.name or func
    return m


def _guided_diffusion():
    """The guided-diffusion package (pip install git+https://github.com/openai/guided-diffusion)."""
    for mod in ("guided_diffusion.script_util", "guided_diffusion.guided_diffusion.script_util"):
        try:
            import importlib
            return importlib.import_module(mod)
        except ImportError:
            continue
    raise ImportError("guided-diffusion is not installed: pip install git+https://github.com/openai/guided-diffusion")


def load_adm(path, cfg, device, half=None):
    """A guided-diffusion (ADM) checkpoint, e.g. the FFHQ 256 model of the paper (configs/ffhq_256_adm.json)."""
    su = _guided_diffusion()
    kw = su.model_and_diffusion_defaults()
    kw.update(cfg.get("model", {}))
    kw["timestep_respacing"] = ""
    kw["use_fp16"] = False                            # fp16 is applied by autocast instead
    net, _ = su.create_model_and_diffusion(**kw)
    net.load_state_dict(_read_state_dict(path))
    net.eval().requires_grad_(False).to(device)
    T = int(kw.get("diffusion_steps", 1000))
    if kw.get("noise_schedule", "linear") != "linear":
        raise ValueError("only the linear ADM schedule is wired here; build the DiffusionModel by hand otherwise")
    scale = 1000.0 / T
    ac = linear_schedule(scale * 0.0001, scale * 0.02, T)

    def blocks_fn(n):
        out = {f"in{i}": b for i, b in enumerate(n.input_blocks)}
        out["mid"] = n.middle_block
        out.update({f"out{i}": b for i, b in enumerate(n.output_blocks)})
        return out

    return DiffusionModel(net, ac, forward_fn=lambda n, x, t: n(x, t), prediction_type="epsilon",
                          eps_from_clipped_x0=True, scalars="float64", blocks_fn=blocks_fn, half=half,
                          image_size=int(kw["image_size"]), channels=3, name=os.path.basename(path))


def load_model(path, config=None, device=None, half=None):
    """A `DiffusionModel`.

    path     a folder written by `UNet2DModel.save_pretrained` (or a pipeline folder with `unet/`),
             a Hugging Face Hub id, a weight file (.pt, .pth, .bin, .ckpt, .safetensors) with --config,
             or 'my_model.py:build' (a function build(device, half) returning a DiffusionModel);
    config   JSON file.  {"type": "diffusers" (default), "unet": {UNet2DModel kwargs},
             "scheduler": {diffusers scheduler config}} or {"type": "guided_diffusion", "model":
             {create_model_and_diffusion kwargs}}; see configs/.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if ".py:" in str(path):
        return _from_python(path, device, half)
    cfg = {}
    if config is not None:
        with open(config) as f:
            cfg = json.load(f)
    if cfg.get("type", "diffusers") == "guided_diffusion":
        return load_adm(path, cfg, device, half)
    from diffusers import UNet2DModel
    if os.path.isfile(path) and path.endswith(_WEIGHT_EXT):
        if "unet" not in cfg:
            raise ValueError(f"{path} is a state dict: pass --config with the UNet2DModel architecture (configs/)")
        net = UNet2DModel(**cfg["unet"])
        net.load_state_dict(_read_state_dict(path))
        sched = cfg.get("scheduler")
    else:
        try:
            net = UNet2DModel.from_pretrained(path)
        except (OSError, EnvironmentError, ValueError):
            net = UNet2DModel.from_pretrained(path, subfolder="unet")
        sched = cfg.get("scheduler", _scheduler_config(path))
    net.eval()
    net.requires_grad_(False)
    net = net.to(device)
    ac, pred = schedule_from_config(sched)
    return DiffusionModel(net, ac, prediction_type=pred, half=half, name=os.path.basename(os.path.normpath(path)))
