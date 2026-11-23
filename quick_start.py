import logging
import time

import torch
from snap import load_model, load_images, psnap, psnaped, load_estimator

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger("quick_start")

N = 8             # images to project
LOG_EVERY = 10    # progress line of the library every LOG_EVERY cycles
BATCH = 4         # images per network call (4 GB GPU shared with other processes)

log.debug("torch %s, CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    log.debug("GPU %s: %.1f GB free of %.1f GB", torch.cuda.get_device_name(0), free / 2**30, total / 2**30)

log.debug("loading the model weights/chestmnist_224.pt ...")
t = time.time()
model = load_model("weights/chestmnist_224.pt", "configs/chestmnist_224.json")
log.info("model loaded on %s in %.1f s (%.1f M parameters)", model.device, time.time() - t,
         sum(p.numel() for p in model.net.parameters()) / 1e6)

log.debug("loading the images data/chestmnist224/cardiomegaly_test.npy ...")
x, names = load_images("data/chestmnist224/cardiomegaly_test.npy", channels=1, size=224)
log.info("%d images loaded: shape %s, values in [%.2f, %.2f]; using the first %d",
         len(x), tuple(x.shape), float(x.min()), float(x.max()), N)

g = torch.Generator(device=model.device).manual_seed(0)
log.debug("generator on %s, seed 0", model.device)

log.info("P-SNAP (Algorithm 1): sigma=1.24, gamma=4.67, 150 cycles ...")
t = time.time()
out = psnap(model, x[:N], sigma=1.24, gamma=4.67, n_cycles=150, generator=g, batch=BATCH, log_every=LOG_EVERY)
# out["x"]: projected images in [-1, 1];  out["mse"]: MSE to the input at every cycle
log.info("P-SNAP done in %.0f s: output %s, values in [%.2f, %.2f]", time.time() - t, tuple(out["x"].shape),
         float(out["x"].min()), float(out["x"].max()))
log.info("P-SNAP: MSE to the input after the last cycle, per image: %s", out["mse"][-1].round(5).tolist())
log.debug("P-SNAP: network calls per image: %s", out["nfe"].tolist())

log.debug("loading the estimator estimators/cnn_cardiomegaly.pt ...")
est = load_estimator("estimators/cnn_cardiomegaly.pt")
log.info("estimator loaded: %s", type(est).__name__)

log.info("P-SNAPED (Algorithm 2): gamma=4.67, tau in [100, 400], at most 500 cycles ...")
t = time.time()
out = psnaped(model, x[:N], est, gamma=4.67, tau_min=100, tau_max=400, max_cycles=500, generator=g, batch=BATCH,
              log_every=LOG_EVERY)
log.info("P-SNAPED done in %.0f s: cycles run per image: %s", time.time() - t,
         out["cycles_run"].tolist())   # where each image stopped
log.debug("P-SNAPED: network calls per image: %s", out["nfe"].tolist())
if torch.cuda.is_available():
    log.debug("peak GPU memory used by this script: %.2f GB", torch.cuda.max_memory_allocated() / 2**30)
