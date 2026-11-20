# Adding a model, a classifier or a dataset

## A diffusion model

SNAP only asks three things of a pre-trained diffusion model:

1. **a noise-prediction network** `eps(x_t, t)` for integer timesteps t (images in [-1, 1]);
2. **its noise schedule** ᾱ_t, t = 0 … T − 1, the one it was trained with;
3. *(LDA estimator only, not used in the paper)* **named blocks** whose outputs serve as features.

Everything else (DDIM cycles, Jacobian-vector products, estimator-driven loop, baselines, figures)
is generic. Four ways to provide a model, from the easiest:

### 1. A diffusers `UNet2DModel` folder or Hub id — nothing to write

```bash
python scripts/project.py --model path/to/folder ...       # UNet2DModel.save_pretrained(folder)
python scripts/project.py --model user/model-id ...        # Hugging Face Hub (a pipeline repo with unet/ works too)
```

The architecture is read from the folder's `config.json`; the schedule from `scheduler_config.json`
(or `scheduler/`) when present, otherwise the linear DDPM schedule (1000 steps, β from 10⁻⁴ to 0.02).
`linear` and `scaled_linear` schedules are recomputed in float64; `prediction_type` `epsilon`,
`v_prediction` and `sample` are all converted to ε.

### 2. A state dict and an architecture file

```bash
python scripts/project.py --model weights/my_ddpm.pt --config configs/my_ddpm.json ...
```

```json
{
  "unet": {"sample_size": 64, "in_channels": 1, "out_channels": 1, "layers_per_block": 2,
           "block_out_channels": [64, 128, 256, 512],
           "down_block_types": ["DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"],
           "up_block_types": ["UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"]},
  "scheduler": {"num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02,
                "beta_schedule": "linear", "prediction_type": "epsilon"}
}
```

`"unet"` holds the keyword arguments of `diffusers.UNet2DModel` exactly as at training time;
`"scheduler"` a diffusers scheduler config. [`configs/chestmnist_224.json`](configs/chestmnist_224.json)
is the model of the paper (and `chestmnist_64.json` a 64-pixel version).

### 3. A guided-diffusion (ADM) checkpoint

```bash
pip install git+https://github.com/openai/guided-diffusion
python scripts/project.py --model weights/ffhq.pt --config configs/ffhq_256_adm.json ...
```

with `{"type": "guided_diffusion", "model": {...}}`: the keyword arguments of
`guided_diffusion.script_util.create_model_and_diffusion` (the rest takes its defaults). A learned
variance (`learn_sigma`) is handled: the first C output channels are ε. The DDIM steps follow
guided-diffusion's convention (ε recomputed from the clipped x̂₀) and its schedule coefficients are
Python floats (`scalars="float64"`): this is what the FFHQ results of the paper used.

### 4. Any other network — one Python function

```python
# my_model.py
import torch
from snap import DiffusionModel
from snap.model import linear_schedule


def build(device, half=None):
    net = MyNetwork(...)                                     # your code
    net.load_state_dict(torch.load("weights/my_model.pt", map_location="cpu"))
    net = net.eval().requires_grad_(False).to(device)
    return DiffusionModel(
        net,
        alphas_cumprod=linear_schedule(1e-4, 0.02, 1000),    # or any array of alpha_bar_t used at training
        forward_fn=lambda net, x, t: net(x, t),              # raw output for images x and timesteps t (B,) long
        prediction_type="epsilon",                           # or "v_prediction", "sample"
        eps_from_clipped_x0=False,                           # True for guided-diffusion-style DDIM
        image_size=256, channels=3, half=half,
    )
```

```bash
python scripts/project.py --model my_model.py:build --input ... --method psnap --sigma 1.24 --gamma 4.67
```

Things to get right:

| | |
|---|---|
| **timesteps** | `forward_fn` receives the training timestep index (0 … T − 1). Wrap the network if it expects a rescaled t, a continuous σ or a class label. |
| **range** | Images must be in [-1, 1] (x̂₀ is clipped to it inside DDIM). Convert in `forward_fn` if the model was trained on [0, 1]. |
| **precision** | The DDIM chain runs under fp16 autocast on CUDA (`half=False` to disable); the one-step denoiser of the Jacobian is always float32. |
| **channels** | Inputs are converted to `channels` (grey ↔ RGB) when they are read. |

### Check a new model, and choose σ and γ

```python
from snap import SNAPCycle
for s in (0.05, 0.34, 1.24):
    y = SNAPCycle(model, sigma=s)(x)                       # x: a few in-domain images in [-1, 1]
    print(s, float((y - x).pow(2).mean()))                 # small at small sigma, growing with sigma
```

At small σ one SNAP cycle must return almost the same image; otherwise the schedule, the timestep
convention or the image range is wrong. The paper's settings: ChestMNIST σ = 1.24, γ = 4.67, r = 0.01;
FFHQ σ ∈ {0.34, 0.72} (restoration) or 2.04 (paintings), γ = 3, r = 0.003. A larger γ lets more pixels
move as with SNAP; `PSNAPCycle(..., gamma)(x, info=d)` reports in `d["fraction_bounded"]` the share
of pixels at the bound |Δ|.

## A classifier

**For the estimator of SNAPED / P-SNAPED** nothing is needed: `scripts/train_estimator.py` trains the
paper's CNN ensemble on N images of the source domain and on images of the model's domain. Any other
score can be used from Python: `snap.FunctionEstimator(score_fn, calibration=...)`, where `score_fn(x)`
returns larger values for source-like images.

**For the evaluation** (detection rate of the pathology): a MedMNIST benchmark weight file or folder,
or your own classifier,

```python
# my_classifier.py
from snap.evaluation import Classifier


class Mine(Classifier):
    labels = ["healthy", "lesion"]
    def __init__(self, device): ...
    def probs(self, x):                                    # x (N, C, H, W) in [-1, 1] -> numpy (N, n_classes)
        ...


def build(device=None):
    return Mine(device)
```

```bash
python scripts/calibrate_classifier.py --classifier my_classifier.py:build \
    --positives lesion=data/lesion_test.npy --negatives data/healthy_test.npy --out calibration/mine.json
python scripts/project.py ... --classifier my_classifier.py:build --heads lesion
```

The threshold is calibrated once on real images (Youden's index) and never re-optimised on outputs.

## A dataset

Inputs are a folder of images (PNG, JPEG…, sorted by name) or a `.npy` / `.npz` array of uint8
images, (N, H, W), (N, H, W, C) or (N, C, H, W). `--n` and `--start` select images in file order,
`--resize` adapts the size to the model. For a new medical dataset, export single-label pathological
images and healthy images per split, as [`scripts/export_chestmnist.py`](scripts/export_chestmnist.py)
does for ChestMNIST.
