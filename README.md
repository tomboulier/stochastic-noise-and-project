# SNAP — Stochastic Noise-And-Project cycles

Code for *SNAP: Stochastic Noise-and-Project Cycles for Projecting Images onto the Manifold of a
Pre-trained Diffusion Model* (anonymous CVPR 2027 submission).

An image from outside the domain of a pre-trained diffusion model (a pathological chest radiograph
for a model of healthy ones, a degraded photograph or a painted portrait for a model of faces) is
brought onto that domain with the model alone: noise the image, denoise it, repeat. No retraining,
no paired images, no model of the deviation.

## Where the four methods are

| paper | code | what one cycle does |
|---|---|---|
| **SNAP** (Sec. 3) | [`snap/methods/snap.py`](snap/methods/snap.py) — `SNAPCycle`, `snap()` | noise to level σ, deterministic DDIM back to 0 |
| **P-SNAP** (Sec. 3.1, Alg. 1) | [`snap/methods/psnap.py`](snap/methods/psnap.py) — `PSNAPCycle`, `psnap()` | Δ = SNAP(x) − x, JΔ by central differences, x ← x + clip(γ(Δ − JΔ), ±\|Δ\|) |
| **SNAPED** (Sec. 3.2, Alg. 2) | [`snap/methods/estimator_driven.py`](snap/methods/estimator_driven.py) — `snaped()` | SNAP cycles, σ set per image and per cycle by a few-shot classifier, stop when the image reads as in-domain |
| **P-SNAPED** (Sec. 3.2, Alg. 2) | [`snap/methods/estimator_driven.py`](snap/methods/estimator_driven.py) — `psnaped()` | the same with P-SNAP cycles |

The rest of the repository:

| | |
|---|---|
| [`snap/estimators.py`](snap/estimators.py) | the few-shot estimator of SNAPED / P-SNAPED (ensemble of three small CNNs; an LDA on the U-Net's activations is also provided, not used in the paper) |
| [`snap/model.py`](snap/model.py) | `DiffusionModel` (noise-prediction network + schedule), `load_model` (diffusers, guided-diffusion, your own code) |
| [`snap/evaluation.py`](snap/evaluation.py) | the pathology classifier of the evaluation (MedMNIST ResNet-18), Youden threshold, LPIPS, bootstrap |
| [`snap/baselines/`](snap/baselines) | SDEdit, AnoDDPM (simplex noise), AutoDDPM, pseudo-healthy VAE, DIIP |
| [`scripts/`](scripts) | command lines: `project.py` (the four methods), `train_estimator.py`, `run_baseline.py`, `calibrate_classifier.py`, data preparation |
| [`paper/`](paper) | one script per figure and table of the paper |
| [`REPRODUCING.md`](REPRODUCING.md) | the exact runs and commands behind every figure and table |
| [`ADDING_A_MODEL.md`](ADDING_A_MODEL.md) | how to use another diffusion model, classifier or dataset |

## Install

With [uv](https://docs.astral.sh/uv/) (Python ≥ 3.10; `uv.lock` pins the versions the self-checks
were last run with):

```bash
uv sync                                   # creates .venv with the locked dependencies
uv run python tests/test_snap.py          # self-checks on a tiny random U-Net, CPU, ~1 min
```

The FFHQ model of the paper is a guided-diffusion network, and the CelebA-HQ photographs come as a
parquet file; both need the `faces` extra: `uv sync --extra faces`.

The commands below are written `python ...`: prefix them with `uv run`, or activate the environment
once (`source .venv/bin/activate`). Without uv, `pip install .` (or `pip install ".[faces]"`) installs
the same dependencies, unpinned.

## Quick start

```python
import torch
from snap import load_model, load_images, psnap, psnaped, load_estimator

model = load_model("weights/chestmnist_224.pt", "configs/chestmnist_224.json")
x, names = load_images("data/chestmnist224/cardiomegaly_test.npy", channels=1, size=224)
g = torch.Generator(device=model.device).manual_seed(0)

out = psnap(model, x[:8], sigma=1.24, gamma=4.67, n_cycles=150, generator=g)   # P-SNAP, Algorithm 1
out["x"]            # projected images in [-1, 1];  out["mse"]: MSE to the input at every cycle

est = load_estimator("estimators/cnn_cardiomegaly.pt")
out = psnaped(model, x[:8], est, gamma=4.67, tau_min=100, tau_max=400, max_cycles=500, generator=g)
out["cycles_run"]   # where each image stopped
```

The same from the command line, with a run folder as output (images, MSE at every cycle, optional
classifier readings, network calls):

```bash
python scripts/project.py --model weights/chestmnist_224.pt --config configs/chestmnist_224.json \
    --input data/chestmnist224/cardiomegaly_test.npy --n 64 --method psnap --sigma 1.24 --gamma 4.67 --cycles 600
python scripts/project.py ... --method snaped  --estimator estimators/cnn_cardiomegaly.pt
python scripts/project.py ... --method psnaped --estimator estimators/cnn_cardiomegaly.pt --gamma 4.67
```

Noise levels are given as σ, the variance-exploding level of the paper,
σ_τ = √((1 − ᾱ_τ)/ᾱ_τ) (`--tau` gives the timestep instead). For the linear DDPM schedule:

| τ | 100 | 150 | 200 | 300 | 400 |
|---|---|---|---|---|---|
| σ | 0.34 | 0.52 | 0.72 | 1.24 | 2.04 |

## Models and data

**No weights are in this repository.**

- ChestMNIST DDPM (healthy radiographs, 224 × 224, diffusers `UNet2DModel`, architecture in
  [`configs/chestmnist_224.json`](configs/chestmnist_224.json)): released with the camera-ready version.
- FFHQ 256 × 256 DDPM: the guided-diffusion checkpoint released with *Label-efficient semantic
  segmentation with diffusion models* (Baranchuk et al., ICLR 2022); configuration in
  [`configs/ffhq_256_adm.json`](configs/ffhq_256_adm.json).
- Evaluation classifier: the ResNet-18 of the MedMNIST v2 benchmark, 224 branch (published weights,
  three seeds averaged).
- Data: ChestMNIST from MedMNIST+ ([`scripts/export_chestmnist.py`](scripts/export_chestmnist.py)),
  SNGFaces paintings, face photographs (see REPRODUCING.md).

Any other model, classifier or dataset can be used: [ADDING_A_MODEL.md](ADDING_A_MODEL.md).

## Notes

- On CUDA the DDIM chain runs under fp16 autocast (`--fp32` to disable); the one-step denoiser whose
  Jacobian P-SNAP uses is always evaluated in float32 and not clipped.
- A run is reproducible given the seed, the batch size and the hardware: each batch draws its noise
  from the run's generator.
- Every script writes into a new dated folder and never overwrites a file.

## License

MIT (see [LICENSE](LICENSE)).
