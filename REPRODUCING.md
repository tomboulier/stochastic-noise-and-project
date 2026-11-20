# Reproducing the figures and tables of the paper

Every figure and table of the paper has one script in [`paper/`](paper). The scripts read run folders written
by [`scripts/project.py`](scripts/project.py) (SNAP, P-SNAP, SNAPED, P-SNAPED) and
[`scripts/run_baseline.py`](scripts/run_baseline.py) (competing methods); any model, classifier or dataset can
be used (see [ADDING_A_MODEL.md](ADDING_A_MODEL.md)). This page gives the runs of the paper, with their seeds,
then the command of each figure.

Each script draws the version printed in the paper (`--style printed`, the default); `--style da` gives the
later variant drawn at the printed width in a sans-serif font.

## What is reproduced exactly

The code paths of this repository were checked against the code that produced the paper: same inputs, model
and seeds give the same images bit for bit on the same hardware (SNAP, P-SNAP at a fixed gain, SNAPED,
P-SNAPED, the estimator training, the classifier, the baselines, the data export). Two limits:

- **GPU runs are not bit-reproducible from one process to another** when cuDNN picks its algorithms by
  benchmarking: re-running a submitted 224-pixel run differs from it by one grey level on about 1 % of the pixels
  after a few cycles. Figures and numbers are unaffected at the printed precision.
- **The gain of P-SNAP.** Algorithm 1 uses a fixed gain γ. Some P-SNAP results of the submitted version were
  computed with an earlier variant in which the gain was estimated per image and per cycle; its value was close
  to the γ given in the paper (about 4.7 on ChestMNIST at σ = 1.24, about 3 on FFHQ). Recomputed with Algorithm 1,
  these parts are close to the paper but not identical:

| | P-SNAP in the submitted version | recomputed with this repository |
|---|---|---|
| Fig. 1 | estimated gain (all three rows) | close |
| Fig. 2 | — (diagram) | identical |
| Fig. 3 | estimated gain (radiograph and photograph) | SNAP identical, P-SNAP close |
| Fig. 4 | radiographs: estimated gain; photographs: fixed γ = 4.67 | SNAP and the photographs identical, P-SNAP radiographs close |
| Fig. 5 | — (SNAP and SNAPED) | identical |
| Fig. 6 | P-SNAP and P-SNAPED: estimated gain | SNAP, SNAPED and the baselines identical, P-SNAP / P-SNAPED close |
| Table 1 | estimated gain | all rows identical except P-SNAP, close |
| Fig. 7 | estimated gain | close |

## 0. Weights, data, calibration (once)

Weights are not distributed with the code:

| | file | notes |
|---|---|---|
| ChestMNIST DDPM (healthy, 224 px) | `weights/chestmnist_224.pt` | [`configs/chestmnist_224.json`](configs/chestmnist_224.json); released with the camera-ready version |
| ChestMNIST DDPM fine-tuned on simplex noise (AnoDDPM) | `weights/chestmnist_224_simplex.pt` | `scripts/finetune_simplex_ddpm.py` from the healthy DDPM |
| Pseudo-healthy VAE, β = 1, 0.1, 0.01 (224 px) | `weights/vae/vae_b{1,0.1,0.01}.pt` | `scripts/train_vae.py` |
| FFHQ 256 DDPM (guided-diffusion) | `weights/ffhq.pt` | [`configs/ffhq_256_adm.json`](configs/ffhq_256_adm.json) |
| Classifier of the evaluation | `weights/medmnist/resnet18_224_{1,2,3}.pth` | MedMNIST v2 benchmark, ChestMNIST, ResNet-18 224 branch: the three published seeds, averaged |

```bash
# ChestMNIST 224 px, shuffled with seed 0 (the paper's image sets are the FIRST images of these arrays)
python scripts/export_chestmnist.py --size 224 --pathologies Cardiomegaly,Effusion,Edema \
    --max_healthy_train 5000 --max_healthy_val 2000 --out data/chestmnist224
D=data/chestmnist224

# detection thresholds: Youden's index on ALL real test images (single-label positives, healthy negatives)
python scripts/calibrate_classifier.py --classifier weights/medmnist --negatives $D/healthy_test.npy \
    --positives Cardiomegaly=$D/cardiomegaly_test.npy,Effusion=$D/effusion_test.npy,Edema=$D/edema_test.npy \
    --out calibration/r18_224_test.json
#   -> thresholds 0.0299 (Cardiomegaly), 0.1178 (Effusion), 0.0088 (Edema); 19 % of healthy test images detected

# the estimators of SNAPED / P-SNAPED: ensembles of three CNNs, N = all single-label training images
# (754 cardiomegalies, 2730 effusions, 479 edemas), 5000 healthy training images, 2000 healthy validation images
for P in cardiomegaly effusion edema; do
  python scripts/train_estimator.py --kind cnn --source $D/${P}_train.npy --healthy $D/healthy_train.npy \
      --n_healthy 5000 --healthy_val $D/healthy_val.npy --n_val 2000 --out estimators/cnn_${P}.pt
done

# faces (FFHQ model): photographs of the "plain" folder of "Glasses and Coverings", SNGFaces paintings, CelebA-HQ 256
python scripts/prepare_faces.py --input datasets/plain --name photo --color_only 1 --splits test=64,val=64,train=-1
python scripts/prepare_faces.py --input datasets/500paintings --name painting --recursive 1 --splits test=64,val=64,train=-1
python scripts/prepare_faces.py --input validation-00000-of-00001.parquet --name celebahq --preprocess none \
    --splits test=100,val=32,train=16
P=data/faces/<date-time>_photo
python scripts/degrade_faces.py --test $P/photo_test.npz:64,$P/photo_train.npz:36 --val $P/photo_val.npz:32 --level moderate
python scripts/degrade_faces.py --test $P/photo_test.npz:64,$P/photo_train.npz:36 --val $P/photo_val.npz:32 --level strong
```

## 1. Runs

Common options. The seeds are those of the submitted runs; the batch size changes the noise of each image, so
keep it.

```bash
M="--model weights/chestmnist_224.pt --config configs/chestmnist_224.json"
F="--model weights/ffhq.pt --config configs/ffhq_256_adm.json --h_rel 0.003 --batch 16"
C="--classifier weights/medmnist"
```

**ChestMNIST, fixed level σ = 1.24, 600 cycles** (Figs. 3–6). SNAP and P-SNAP of the same images share the seed,
hence the same noise at the first cycle.

| images (`--input`, `--n 64`) | seed | runs |
|---|---|---|
| `healthy_test.npy` | 973877553 | `--method snap` and `--method psnap --gamma 4.67` |
| `cardiomegaly_test.npy` | 1995644633 | idem, with `$C --heads Cardiomegaly` |
| `effusion_test.npy` | 740428020 | idem, with `$C --heads Effusion` |
| `edema_test.npy` | 1593127204 | idem, with `$C --heads Edema` |

```bash
python scripts/project.py $M --input $D/edema_test.npy --n 64 --method snap  --sigma 1.24 --cycles 600 \
    --batch 32 --seed 1593127204 $C --heads Edema --out outputs/chest
python scripts/project.py $M --input $D/edema_test.npy --n 64 --method psnap --sigma 1.24 --gamma 4.67 --cycles 600 \
    --batch 32 --seed 1593127204 $C --heads Edema --out outputs/chest
```

**ChestMNIST, SNAPED and P-SNAPED** (Figs. 5, 6): σ between 0.34 and 2.04, at most 500 cycles, stop below the
median of the held-out healthy scores (the defaults), batch 128. The cardiomegaly runs of the paper took all 243
test cardiomegalies (`--n 0`), the others the first 64; the figures use the first 64 images.

```bash
python scripts/project.py $M --input $D/edema_test.npy --n 64 --method snaped --estimator estimators/cnn_edema.pt \
    --batch 128 --seed 45659557 $C --heads Edema --out outputs/chest
python scripts/project.py $M --input $D/edema_test.npy --n 64 --method psnaped --gamma 4.67 --estimator estimators/cnn_edema.pt \
    --batch 128 --seed 1689606824 $C --heads Edema --out outputs/chest
```

**ChestMNIST, baselines** (Fig. 6), for each pathology P and its images (the default grids are those of the figure):

```bash
python scripts/run_baseline.py --method sdedit   $M --input $D/edema_test.npy --n 64 --group Edema --out outputs/chest
python scripts/run_baseline.py --method anoddpm  --model weights/chestmnist_224_simplex.pt --config configs/chestmnist_224.json \
    --input $D/edema_test.npy --n 64 --group Edema --out outputs/chest
python scripts/run_baseline.py --method autoddpm $M --input $D/edema_test.npy --n 64 --group Edema \
    --healthy $D/healthy_val.npy --n_healthy 32 --out outputs/chest
python scripts/run_baseline.py --method vae --vae weights/vae/vae_b1.pt,weights/vae/vae_b0.1.pt,weights/vae/vae_b0.01.pt \
    --input $D/edema_test.npy --n 64 --group Edema --out outputs/chest
python scripts/evaluate_runs.py --runs "outputs/chest/*" $C --heads Cardiomegaly,Effusion,Edema --name judge
```

**ChestMNIST, Fig. 1** (radiograph row): P-SNAP σ = 2.04 on the test cardiomegalies, seed 181964101.

```bash
python scripts/project.py $M --input $D/cardiomegaly_test.npy --n 64 --method psnap --sigma 2.04 --gamma 4.67 \
    --cycles 5 --batch 32 --seed 181964101 --out outputs/fig1
```

**FFHQ model** (`P` = the photographs, `A` = the paintings, `Q` = CelebA-HQ, `L` = a level of degraded photographs):

```bash
# Fig. 7 and Fig. 1 (painting row): the 64 test paintings, P-SNAP sigma 2.04, 4 cycles
python scripts/project.py $F --input $A/painting_test.npz --method psnap --sigma 2.04 --gamma 3 --cycles 4 --seed 643182507 --out outputs/paintings
# Fig. 3 (photograph): the first 8 test photographs, sigma 1.24, 600 cycles, SNAP and P-SNAP (the figure shows #4)
python scripts/project.py $F --input $P/photo_test.npz --n 8 --method snap  --sigma 1.24 --cycles 600 --seed 15658429 --out outputs/faces
python scripts/project.py $F --input $P/photo_test.npz --n 8 --method psnap --sigma 1.24 --gamma 4.67 --cycles 600 --seed 15658429 --out outputs/faces
# Fig. 4 (photographs): CelebA-HQ test photographs 84-99 in two runs of 8, sigma 1.24, 600 cycles, SNAP and P-SNAP
python scripts/project.py $F --input $Q/celebahq_test.npz --start 84 --n 8 --method snap  --sigma 1.24 --cycles 600 --seed 15658429 --out outputs/faces
python scripts/project.py $F --input $Q/celebahq_test.npz --start 84 --n 8 --method psnap --sigma 1.24 --gamma 4.67 --cycles 600 --seed 15658429 --out outputs/faces
python scripts/project.py $F --input $Q/celebahq_test.npz --start 92 --n 8 ...              # the same for photographs 92-99
# Table 1, for each level: SNAP and P-SNAP at sigma 0.34 and 0.72 (strong level: also 1.24), 10 cycles,
# on the four degradations together (all.npz), seeds 1009832855 (sigma 0.34), 163756827 (0.72), 1899634304 (1.24)
python scripts/project.py $F --input $L/all.npz --method snap  --sigma 0.34 --cycles 10 --save_cycles 1,2,3,4,5,6,7,8,9,10 --seed 1009832855 --out outputs/faces_moderate
python scripts/project.py $F --input $L/all.npz --method psnap --sigma 0.34 --gamma 3 --cycles 10 --save_cycles 1,2,3,4,5,6,7,8,9,10 --seed 1009832855 --out outputs/faces_moderate
# Table 1, baselines (see scripts/run_baseline.py --recipe faces), then LPIPS / MSE to the clean photographs
python scripts/run_baseline.py --method sdedit   --model weights/ffhq.pt --config configs/ffhq_256_adm.json --recipe faces \
    --input $L/blur.npz,$L/jpeg.npz,$L/pixelation.npz,$L/noise.npz --reference $L/clean.npz,$L/clean.npz,$L/clean.npz,$L/clean.npz --out outputs/faces_moderate
python scripts/evaluate_runs.py --runs "outputs/faces_moderate/*" --reference $L/clean.npz --name faces
# Fig. 1 (photograph row): the pixelated photographs (x8), P-SNAP sigma 0.72, 5 cycles
python scripts/project.py $F --input $L/pixelation.npz --method psnap --sigma 0.72 --gamma 3 --cycles 5 --out outputs/fig1
```

## 2. Figures and table

```bash
python paper/fig1_teaser.py --runs <radiograph run>,<painting run>,<photograph run> --images 29,31,4
pdflatex paper/fig2_diagrams.tex
python paper/fig3_drift_examples.py --snap <healthy SNAP run>,<photo SNAP run> --psnap <healthy P-SNAP run>,<photo P-SNAP run> --images 0,4
python paper/fig4_drift_mse.py --snap "<healthy SNAP run>;<CelebA-HQ SNAP run 84-91>,<run 92-99>" --psnap "<same for P-SNAP>"
python paper/fig5_snap_vs_snaped.py --snap Edema=<run>,Effusion=<run>,Cardiomegaly=<run> \
    --snaped Edema=<run>,Effusion=<run>,Cardiomegaly=<run> --calibration calibration/r18_224_test.json
python paper/fig6_benchmark.py --calibration calibration/r18_224_test.json \
    --snap "Cardiomegaly=<run>;Effusion=<run>;Edema=<run>" --psnap "..." --snaped "..." --psnaped "..." \
    --baselines "Cardiomegaly=<run>,<run>,...;Effusion=...;Edema=..."
python paper/table1_degraded_faces.py --level "moderate=outputs/faces_moderate/*" --level "strong=outputs/faces_strong/*"
python paper/fig7_paintings.py --run <painting run> --images 26,28,41
```

The docstring of each script details its inputs and options.

## 3. Where the paper text and its runs differ

To correct in the next version of the paper or the supplement:

- **P-SNAP's gain**: see the table at the top (estimated gain in part of the submitted results).
- **Table 1**: SNAP and P-SNAP ran for 10 cycles (the text says "at most 20"); the strong level also had σ = 1.24
  (the text gives σ ∈ {0.34, 0.72}), and that setting is the one chosen for strong blur and pixelation.
- **Fig. 6**: the SDEdit marks go up to τ = 900 and AutoDDPM's masking level up to 600 (the text gives τ ≤ 600 and
  {200, 300, 400}); AnoDDPM's grid is {100, 200, 250, 300, 400}. The simplex model of AnoDDPM was fine-tuned from
  the healthy DDPM, not trained from scratch.
- **Figs. 3 and 4**: Fig. 3's photograph is one of 8 "plain" photographs (not CelebA-HQ), while Fig. 4's face
  panel pools 16 CelebA-HQ photographs (84–99); the caption of Fig. 4 and the text say 8 photographs for both.
- **Fig. 1 and the text**: the pixelated photograph of Fig. 1 and the photograph of Fig. 3 are the same person
  ("plain" test photograph #4).
