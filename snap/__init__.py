# -*- coding: utf-8 -*-
"""SNAP: stochastic noise-and-project cycles with a pre-trained diffusion model.

    from snap import load_model, snap, psnap, snaped, psnaped

    SNAP      snap.methods.snap               SNAPCycle, snap()      noise to sigma, DDIM back, repeat
    P-SNAP    snap.methods.psnap              PSNAPCycle, psnap()    Algorithm 1: clip(gamma (I - J) Delta, +-|Delta|)
    SNAPED    snap.methods.estimator_driven   snaped()               Algorithm 2 with SNAP cycles
    P-SNAPED  snap.methods.estimator_driven   psnaped()              Algorithm 2 with P-SNAP cycles
"""

from .model import DiffusionModel, load_model
from .methods import SNAPCycle, PSNAPCycle, snap, psnap, snaped, psnaped, estimator_driven, run_cycles
from .estimators import CNNEstimator, LDAEstimator, FunctionEstimator, train_cnn, fit_lda, load_estimator
from .images import load_images, first, new_run_dir
from .runs import save_run, load_run

__all__ = ["DiffusionModel", "load_model", "SNAPCycle", "PSNAPCycle", "snap", "psnap", "snaped", "psnaped",
           "estimator_driven", "run_cycles", "CNNEstimator", "LDAEstimator", "FunctionEstimator", "train_cnn",
           "fit_lda", "load_estimator", "load_images", "first", "new_run_dir", "save_run", "load_run"]
__version__ = "0.2.0"
