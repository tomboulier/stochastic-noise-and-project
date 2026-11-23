#!/usr/bin/env bash
# Build everything the ChestMNIST experiments need besides the weights of the diffusion model
# (REPRODUCING.md, step 0).
#
#     scripts/build_chestmnist.sh          # or: uv run scripts/build_chestmnist.sh
#
#   1. data/chestmnist224/        the image sets of the paper, exported from ChestMNIST 224 x 224
#                                 (MedMNIST+; a 3.9 GB download to ~/.medmnist the first time)
#   2. estimators/cnn_<P>.pt      the estimators of SNAPED / P-SNAPED, P = cardiomegaly, effusion, edema
#   3. calibration/r18_224_test.json   the detection thresholds of the evaluation classifier; skipped
#                                 when its weights are not in weights/medmnist/
#
# A step whose output already exists is skipped: the script can be run again after an interruption,
# or once the classifier weights are there.  PYTHON selects the interpreter (default: python).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
D=data/chestmnist224

if [ -f "$D/healthy_test.npy" ]; then
    echo "[1/3] $D: already there"
else
    echo "[1/3] exporting ChestMNIST to $D"
    "$PY" scripts/export_chestmnist.py --size 224 --pathologies Cardiomegaly,Effusion,Edema \
        --max_healthy_train 5000 --max_healthy_val 2000 --out "$D"
fi

for P in cardiomegaly effusion edema; do
    if [ -f "estimators/cnn_$P.pt" ]; then
        echo "[2/3] estimators/cnn_$P.pt: already there"
    else
        echo "[2/3] training estimators/cnn_$P.pt"
        "$PY" scripts/train_estimator.py --kind cnn --source "$D/${P}_train.npy" \
            --healthy "$D/healthy_train.npy" --n_healthy 5000 --healthy_val "$D/healthy_val.npy" --n_val 2000 \
            --out "estimators/cnn_$P.pt"
    fi
done

if [ -f calibration/r18_224_test.json ]; then
    echo "[3/3] calibration/r18_224_test.json: already there"
elif ls weights/medmnist/*.pth >/dev/null 2>&1; then
    echo "[3/3] calibrating the classifier"
    "$PY" scripts/calibrate_classifier.py --classifier weights/medmnist --negatives "$D/healthy_test.npy" \
        --positives "Cardiomegaly=$D/cardiomegaly_test.npy,Effusion=$D/effusion_test.npy,Edema=$D/edema_test.npy" \
        --out calibration/r18_224_test.json
else
    echo "[3/3] skipped: no classifier weights in weights/medmnist/ (needed for Figs. 5 and 6 only)"
fi
