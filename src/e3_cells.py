# %% [md]
# # E3: Which similarity metric predicts real cross-modal alignability?
#
# **Question.** The Aristotelian paper calibrates CKA with a permutation threshold; Rosetta (Schnaus, …, Isola, 2026) shows that unbiased CKA predicts how well two spaces can be aligned *without pairs*. Here we put the competing measurements against one objective outcome, on the same 84 model-and-dataset combinations, using **each paper's own code**:
#
# - **Outcome:** FOSCTTM of Rosetta's unpaired aligner (Wasserstein Procrustes with geometric initialization), averaged over the paper's **5 seeds**, on all 40,504 MS COCO val pairs. 7 vision models × 3 language models × 4 training corpora (MS COCO, SPC, DCI, DOCCI). Run with the authors' `run_alignment`, unchanged (commit `fdfad84`).
# - **Rosetta's own scores (their Fig. 3c protocol):** unbiased CKA (fp64), mutual k-NN, TSI, QSI on 10,000 paired training items. Reproducing their R² = 0.73 / ρ = −0.90 is the first check.
# - **The Aristotelian measurements (their code, commit `785054b`):** on n = 1,024 paired items, as in their paper. Features clipped at the 95th percentile and L2-normalized (`prepare_features`); linear CKA and mutual k-NN (`aristotelian.cka`, `aristotelian.mutual_knn`); calibrated with **`calibrated_similarity.calibrate`, K = 500 permutations, α = 0.05**; and their fp32 unbiased CKA. Plus unbiased CKA in fp64 on the same inputs. Each is averaged over **20 independent random subsets** (fixed seeds).
#
# **Pre-registered analysis** (written before any results):
# - **H0, replication:** Rosetta's CKA (n = 10,000) vs mean FOSCTTM gives R² ≈ 0.73, Spearman ≈ −0.90.
# - **H1, main:** at n = 1,024, unbiased CKA predicts FOSCTTM better than Aristotelian-calibrated CKA: |ρ(unbiased)| − |ρ(calibrated)| > 0, 95% cluster-bootstrap CI excluding 0. Clusters are the 21 model pairs, exactly as in Rosetta's `linear_fit`, 10,000 resamples.
# - **H2:** calibration does not change mutual k-NN's predictive power (its chance level is constant), so Δ|ρ| ≈ 0 with a CI covering 0.
# - **Secondary:** sample size (n = 1,024 vs 10,000); precision (fp32 vs fp64 unbiased CKA); robustness within each corpus.
# - **Only combinations with all 5 seeds complete enter the primary analysis.**
#
# **How to run.**
# 1. **Smoke test** (`SMOKE_TEST = True`; already passed 15/15 on 11 Oct 2026). Choose *Run all*. It runs 1 corpus (DCI) × 2 vision × 1 language model, with 1 seed, 2 subsets and K = 50, which takes roughly 20–40 minutes on a Colab CPU allotment (the first finished alignment prints an ETA). It ends with a **CHECKER** table (PASS/FAIL), including a null check and an exact-determinism re-run, then downloads `e3_smoke_results.zip`. Send that zip to Claude.
# 2. **Full run** (`SMOKE_TEST = False`, **the default in this version**). Choose *Run all*. Results go to `prh_audit/e3` on Drive. At the end the analysis cell writes `e3_results.zip`; send that to Claude. If the runtime disconnects, reconnect and *Run all* again: finished runs are skipped.
#
# **Compute.** The aligner is CPU-only, with 100 refinement steps of 10,000 × 10,000 Hungarian matching, measured at 2–3 minutes per run on DCI and ~77 minutes per run on MS COCO (one at a time, 48 cores). That's 420 runs; with ~12 alignments side by side, roughly a day. The progress line prints `eta_hours`. It runs in **background processes**, several alignments side by side (CPU cores // 4), and writes one JSON per run to Drive, seed by seed, so partial results are usable and a disconnect loses nothing (*Run all* again resumes). The GPU metrics run in the foreground on the GPU, taking well under an hour. The last cell waits for the background job.

# %%
# ============================== CONFIG ==============================
import os
SMOKE_TEST = False      # True: ~20–40 min check (1 corpus, 2 vision x 1 language, 1 seed, small K) + CHECKER cell.
                        # False: the full pre-registered run. Results go to separate folders.
ROSETTA_REPO_URL, ROSETTA_COMMIT = "https://github.com/dominik-schnaus/unpaired-rosetta", "fdfad84ea488f29702a61a455be71622c4ff5d49"
ARISTO_REPO_URL, ARISTO_COMMIT = "https://github.com/mlbio-epfl/Aristotelian", "785054b19c952dca0f605768552ddd50fa9286dc"
EMB_REPO = "schnaus/unpaired-rosetta-embeddings"
N_SUB = 1024            # Aristotelian paper's sample size
N_SUBSETS = 20          # independent random subsets per combination
K_PERM = 500            # permutations (Aristotelian paper's runs: 500)
ALPHA = 0.05
TOPK = 10
SUBSET_SEED_BASE = 20261011
B_BOOT = 10_000         # cluster-bootstrap resamples for the comparisons
TASK_PARALLEL = None    # alignments run side by side. None = automatic: CPU cores // 4, bounded by RAM (12 GB each).
                        # Each alignment's 100 refinement steps are sequential, so one-at-a-time leaves most cores idle.
RESTART_WORKERS = 4     # processes for each alignment's 30 restarts (the result does not depend on this; see smoke test)
DRYRUN = os.environ.get("E3_DRYRUN") == "1"   # local test only; never set on Colab
if SMOKE_TEST:          # smallest corpus (DCI, 7,805 items), so one aligner run takes a few CPU-minutes
    N_SUBSETS, K_PERM, B_BOOT = 2, 50, 200

# %%
# ============================== SETUP ==============================
import sys, subprocess, json, time, glob, shutil, math
IN_COLAB = "google.colab" in sys.modules
WORK = "/content" if IN_COLAB else os.environ.get("E3_WORK", "/tmp/e3")
os.makedirs(WORK, exist_ok=True)
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibmgm>=1.1.2,<2", "scikit-learn", "scipy", "seaborn",
                    "threadpoolctl", "loguru", "datasets", "huggingface_hub", "psutil"], check=True)
    from google.colab import drive; drive.mount("/content/drive")
    try:
        from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        pass

def clone(url, commit, path):
    if not os.path.exists(os.path.join(path, ".git")):
        subprocess.run(["git", "clone", "--quiet", url, path], check=True)
    subprocess.run(["git", "-C", path, "checkout", "--quiet", commit], check=True)
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == commit, (path, head)
    return path

ROSETTA = os.environ.get("E3_ROSETTA") or clone(ROSETTA_REPO_URL, ROSETTA_COMMIT, os.path.join(WORK, "unpaired-rosetta"))
ARISTO = os.environ.get("E3_ARISTO") or clone(ARISTO_REPO_URL, ARISTO_COMMIT, os.path.join(WORK, "Aristotelian"))
for p in (ROSETTA, ARISTO):
    if p not in sys.path: sys.path.insert(0, p)
if sys.version_info < (3, 14):     # Rosetta targets Python 3.14 (lazy annotations); one type hint names `pd`
    import builtins, pandas
    builtins.pd = pandas

STORAGE = os.environ.get("E3_STORAGE") or os.path.join(WORK, "rosetta_storage")      # embeddings (local disk)
os.environ["UNPAIRED_ROSETTA_ROOT"] = STORAGE
OUT = os.environ.get("E3_OUT") or (("/content/drive/MyDrive/prh_audit/" + ("e3_smoke" if SMOKE_TEST else "e3")) if IN_COLAB
                                   else os.path.join(WORK, "out"))
for d in ("align", "geometry", "metrics_gpu", "figs"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

import torch, numpy as np, pandas as pd, psutil, platform
torch.backends.cuda.matmul.allow_tf32 = False          # exact fp32 everywhere
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = os.path.join(OUT, "log.txt")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s); open(LOG, "a").write(s + "\n")

from modalities import image_captions as IC
from unpaired_rosetta.randomness import SEEDS
import scipy, sklearn, pylibmgm
versions = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, pylibmgm=getattr(pylibmgm, "__version__", "?"), rosetta=ROSETTA_COMMIT, aristotelian=ARISTO_COMMIT,
                cpus=os.cpu_count(), ram_gb=round(psutil.virtual_memory().total / 1e9), gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None)
json.dump(versions, open(os.path.join(OUT, "versions.json"), "w"), indent=1)
log("versions:", versions)

DATASETS, VISION, LANGUAGE = list(IC.TRAINING_DATASETS), list(IC.VISION_MODELS), list(IC.LANGUAGE_MODELS)
VALIDATION = IC.VALIDATION_DATASET
RUN_SEEDS = [int(s) for s in SEEDS]                      # the paper's 5 seeds
if SMOKE_TEST:
    DATASETS, VISION, LANGUAGE = ["DenselyCaptionedImages"], ["dinov2_vit-b14@224_mean", "ibot_vit-b16@224_mean"], ["mpnet"]
    RUN_SEEDS = RUN_SEEDS[:1]
if DRYRUN:
    from e3_dryrun_support import DRY_DATASETS, DRY_VISION, DRY_LANGUAGE, DRY_SIZES
    DATASETS, VISION, LANGUAGE = DRY_DATASETS, DRY_VISION, DRY_LANGUAGE
COMBOS = [(d, v, l) for d in DATASETS for v in VISION for l in LANGUAGE]
log(f"{'SMOKE TEST: ' if SMOKE_TEST else ''}{len(COMBOS)} combinations x {len(RUN_SEEDS)} seeds = "
    f"{len(COMBOS) * len(RUN_SEEDS)} alignment runs; seeds = {RUN_SEEDS}")

# %%
# ============================== EMBEDDINGS (download only what is needed) ==============================
from unpaired_rosetta.embeddings import embedding_path, load_rows, num_rows
need = [f"embeddings/{d}/vision/{v}.pt" for d in DATASETS + [VALIDATION] for v in VISION] + \
       [f"embeddings/{d}/language/{l}.pt" for d in DATASETS + [VALIDATION] for l in LANGUAGE]
missing = [p for p in need if not os.path.exists(os.path.join(STORAGE, p))]
if DRYRUN:
    from e3_dryrun_support import write_fake_storage
    write_fake_storage(STORAGE)
    missing = [p for p in need if not os.path.exists(os.path.join(STORAGE, p))]
elif missing:
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().dataset_info(EMB_REPO, files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    absent = [p for p in missing if p not in sizes]
    assert not absent, f"not in {EMB_REPO}: {absent[:5]}"
    gb = sum(sizes[p] for p in missing) / 1e9
    log(f"downloading {len(missing)} files, {gb:.1f} GB; free disk {shutil.disk_usage('/').free / 1e9:.0f} GB")
    snapshot_download(EMB_REPO, repo_type="dataset", local_dir=STORAGE, allow_patterns=missing, max_workers=8)
    missing = [p for p in need if not os.path.exists(os.path.join(STORAGE, p))]
assert not missing, missing[:5]
for d in DATASETS + [VALIDATION]:
    if not DRYRUN and d in IC.NUM_ITEMS:
        assert num_rows(embedding_path(d, "vision", VISION[0])) == IC.NUM_ITEMS[d], d   # file matches the paper's shape
log("embeddings ready")

# %%
# ============================== WORKER SCRIPT (runs the authors' code in a background process) ==============================
WORKER_SOURCE = r'''__WORKER__'''

# %%
# ============================== BACKGROUND JOB: Rosetta aligner (5 seeds) + Rosetta geometry scores ==============================
tasks = [dict(kind="geometry", dataset=d, vision=v, language=l) for d, v, l in COMBOS]
tasks += [dict(kind="align", dataset=d, vision=v, language=l, seed=int(s)) for s in RUN_SEEDS for d, v, l in COMBOS]  # seed-major
if DRYRUN:
    for t in tasks:
        t.update(DRY_SIZES)
json.dump(tasks, open(os.path.join(OUT, "tasks.json"), "w"))
import psutil
if TASK_PARALLEL is None:
    TASK_PARALLEL = max(1, min((os.cpu_count() or 4) // RESTART_WORKERS, int(psutil.virtual_memory().available / 1e9 // 12)))
if SMOKE_TEST or DRYRUN:
    TASK_PARALLEL = 1
worker = os.path.join(WORK, "e3_worker.py")
open(worker, "w").write(WORKER_SOURCE)            # defined in the WORKER cell above
# Stop any worker left over from an earlier run of this cell (finished tasks are on Drive and are skipped).
subprocess.run(["pkill", "-f", "e3_worker.py"]); time.sleep(2)
def _task_file(t):
    if t["kind"] == "align":
        return os.path.join(OUT, "align", f"{t['dataset']}__{t['vision']}__{t['language']}__seed{t['seed']}.json")
    return os.path.join(OUT, "geometry", f"{t['dataset']}__{t['vision']}__{t['language']}.json")
pending = [t for t in tasks if not os.path.exists(_task_file(t))]
# Round-robin shards keep the seed-major order: seed 1 of every combination finishes first.
# Each alignment depends only on its own seed, so running them side by side gives identical results.
threads = str(min(4, RESTART_WORKERS))             # same BLAS thread count as Rosetta's num_threads for every task
BG = []
for k in range(TASK_PARALLEL):
    shard = pending[k::TASK_PARALLEL]
    if not shard: continue
    tf = os.path.join(OUT, f"tasks_{k}.json"); json.dump(shard, open(tf, "w"))
    BG.append(subprocess.Popen([sys.executable, worker, tf, OUT, ROSETTA, str(RESTART_WORKERS), f"worker_log_{k}.txt"],
                               stdout=open(os.path.join(OUT, f"worker_stdout_{k}.txt"), "a"), stderr=subprocess.STDOUT,
                               env=dict(os.environ, UNPAIRED_ROSETTA_ROOT=STORAGE, PYTHONPATH=ROSETTA, OMP_NUM_THREADS=threads,
                                        MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)))
log(f"background job: {len(pending)} pending tasks on {len(BG)} parallel workers, each alignment's 30 restarts on "
    f"{RESTART_WORKERS} processes. Logs: {OUT}/worker_log_*.txt; progress bars: {OUT}/worker_stdout_*.txt")

def worker_logs():
    return "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(os.path.join(OUT, "worker_log*.txt"))))

def progress():
    na = len(glob.glob(os.path.join(OUT, "align", "*.json"))); ng = len(glob.glob(os.path.join(OUT, "geometry", "*.json")))
    done = [json.load(open(f))["seconds"] for f in glob.glob(os.path.join(OUT, "align", "*.json"))]
    total_a = len(COMBOS) * len(RUN_SEEDS)
    eta = (np.mean(done) * (total_a - na) / max(1, TASK_PARALLEL) / 3600) if done else float("nan")
    return dict(align_done=na, align_total=total_a, geometry_done=ng, geometry_total=len(COMBOS), eta_hours=round(eta, 1))
log("progress:", progress())

# %%
# ============================== GPU: the Aristotelian measurements at n = 1,024 (their code) ==============================
import aristotelian as A
from aristotelian.metrics.registry import MetricRegistry
from aristotelian.metrics.base import MetricConfig
torch.set_num_threads(4)   # leave the CPU cores to the aligners; the metrics below run on the GPU
# A.cka / A.mutual_knn / A.unbiased_cka build MetricConfig() with its default device="cpu", i.e. they copy every input
# to the CPU and compute there (that made this cell CPU-bound and ~10x slower). Same registry metrics, same settings,
# computed on the GPU (TF32 off, exact fp32):
def a_cka(a, b):  return MetricRegistry.compute_raw("cka", a, b, MetricConfig(kernel="ip", rbf_sigma=1.0, unbiased=False, device=DEV))
def a_mknn(a, b): return MetricRegistry.compute_raw("mutual_knn", a, b, MetricConfig(topk=TOPK, device=DEV))
def a_ucka(a, b): return MetricRegistry.compute_raw("cka_unbiased", a, b, MetricConfig(device=DEV))
from aristotelian.prh.preprocess import prepare_features
from calibrated_similarity import calibrate
from unpaired_rosetta.geometry import cka as rosetta_unbiased_cka
F = torch.nn.functional

def aristo_prep(x):        # Huh / Aristotelian pipeline: clip at the 95th percentile, then L2-normalize
    return F.normalize(prepare_features(x, q=0.95, exact=False), dim=-1)

def eff_rank(x):
    xc = x - x.mean(0, keepdim=True); G = (xc @ xc.T).double()
    return float(torch.trace(G) ** 2 / (G * G).sum())

def gen(seed):
    g = torch.Generator(device=DEV); g.manual_seed(seed); return g

def measure_combo(d, v, l):
    fp = os.path.join(OUT, "metrics_gpu", f"{d}__{v}__{l}.json")
    if os.path.exists(fp): return
    px, py = embedding_path(d, "vision", v), embedding_path(d, "language", l)
    n_items = num_rows(px); rows = []
    for s in range(N_SUBSETS):
        seed = SUBSET_SEED_BASE + s
        idx = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed))[:N_SUB]
        x, y = load_rows(px, idx), load_rows(py, idx)                       # Rosetta loader: unit-length fp32
        xp, yp = aristo_prep(x).to(DEV), aristo_prep(y).to(DEV)
        cka_raw = float(a_cka(xp, yp))
        cka_cal, cka_p, cka_tau = calibrate(xp, yp, a_cka, K=K_PERM, alpha=ALPHA, smax=1.0, generator=gen(seed))
        mk_raw = float(a_mknn(xp, yp))
        mk_cal, mk_p, mk_tau = calibrate(xp, yp, a_mknn, K=K_PERM, alpha=ALPHA, smax=1.0, generator=gen(seed + 1))
        rows.append(dict(
            subset=s, cka_raw=cka_raw, cka_cal=float(cka_cal), cka_tau=float(cka_tau), cka_p=float(cka_p),
            mknn_raw=mk_raw, mknn_cal=float(mk_cal), mknn_tau=float(mk_tau), mknn_p=float(mk_p),
            ucka_aristo_fp32=float(a_ucka(xp, yp)),                      # their unbiased CKA, as they run it
            ucka_fp64_aristoprep=float(rosetta_unbiased_cka(xp.double(), yp.double())),   # same inputs, fp64, no epsilon
            ucka_fp64_rosettaprep=float(rosetta_unbiased_cka((x - x.mean(0)).to(DEV), (y - y.mean(0)).to(DEV))),  # Rosetta's preprocessing at n=1,024
            reff_x=eff_rank(xp), reff_y=eff_rank(yp)))
    json.dump(dict(dataset=d, vision=v, language=l, n=N_SUB, K=K_PERM, subsets=rows), open(fp, "w"))

t0 = time.time()
for i, (d, v, l) in enumerate(COMBOS):
    measure_combo(d, v, l)
    if i % 10 == 0: log(f"metrics {i + 1}/{len(COMBOS)} ({time.time() - t0:.0f}s) | background: {progress()}")
log("GPU metrics done", progress())

# %%
# ============================== WAIT FOR THE BACKGROUND JOB ==============================
# Safe to re-run: if the runtime restarted, the SETUP and BACKGROUND cells resume the job first.
while True:
    p = progress()
    if p["align_done"] >= p["align_total"] and p["geometry_done"] >= p["geometry_total"]: break
    alive = subprocess.run(["pgrep", "-f", "e3_worker.py"], capture_output=True, text=True).stdout.strip()
    if not alive:
        log("background job is not running but tasks remain; see worker_log_*.txt for FAILED entries", p); break
    try:   # last progress-bar line of the running alignment (tqdm writes \r-separated updates)
        tail = open(os.path.join(OUT, "worker_stdout_0.txt"), errors="ignore").read()[-4000:].replace("\r", "\n").strip().splitlines()
        cur = tail[-1][-150:] if tail else ""
    except OSError:
        cur = ""
    log("waiting:", p, "| current:", cur); time.sleep(5 if DRYRUN else (60 if SMOKE_TEST else 600))

# %%
# ============================== CHECKER (runs in both modes; read the PASS/FAIL table) ==============================
checks = []
def check(name, ok, detail=""):
    checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=str(detail)[:160]))

wlog = worker_logs()
check("worker: no FAILED tasks", "FAILED" not in wlog, wlog[wlog.find("FAILED"):][:160] if "FAILED" in wlog else "")
A_FILES = glob.glob(os.path.join(OUT, "align", "*.json")); G_FILES = glob.glob(os.path.join(OUT, "geometry", "*.json"))
M_FILES = glob.glob(os.path.join(OUT, "metrics_gpu", "*.json"))
check("all alignment runs present", len(A_FILES) == len(COMBOS) * len(RUN_SEEDS), f"{len(A_FILES)}/{len(COMBOS) * len(RUN_SEEDS)}")
check("all geometry runs present", len(G_FILES) == len(COMBOS), f"{len(G_FILES)}/{len(COMBOS)}")
check("all metric files present", len(M_FILES) == len(COMBOS), f"{len(M_FILES)}/{len(COMBOS)}")

fos = [json.load(open(f))["metrics"] for f in A_FILES]
check("FOSCTTM within [0, 0.5]", all(0 <= m["foscttm"] <= 0.5 for m in fos), [round(m["foscttm"], 4) for m in fos][:8])
check("evaluated on all 40,504 COCO val pairs", all(m["num_validation_samples"] == IC.NUM_ITEMS["coco_val2014"] for m in fos) or DRYRUN,
      {m["num_validation_samples"] for m in fos})
check("informational: FOSCTTM per run (0.5 = chance; weak pairs may sit near it)", True,
      [round(m["foscttm"], 4) for m in fos][:8])
geo_m = [json.load(open(f))["metrics"] for f in G_FILES]
check("Rosetta geometry scores in range", all(-1 <= g["cka"] <= 1 and 0 <= g["mutual_knn"] <= 1 and 0 <= g["tsi"] <= 1 for g in geo_m),
      [{k: round(v, 3) for k, v in g.items()} for g in geo_m][:2])
subs = [s for f in M_FILES for s in json.load(open(f))["subsets"]]
check("calibrated CKA in [0, 1] and <= raw CKA", all(0 <= s["cka_cal"] <= 1 and s["cka_cal"] <= s["cka_raw"] + 1e-6 for s in subs))
check("calibrated mutual k-NN in [0, 1] and <= raw", all(0 <= s["mknn_cal"] <= 1 and s["mknn_cal"] <= s["mknn_raw"] + 1e-6 for s in subs))
check("permutation p-values in (0, 1]", all(0 < s["cka_p"] <= 1 and 0 < s["mknn_p"] <= 1 for s in subs))
gap = max(abs(s["ucka_aristo_fp32"] - s["ucka_fp64_aristoprep"]) for s in subs) if subs else float("nan")
check("their fp32 unbiased CKA vs fp64 on identical inputs (informational)", True, f"max |difference| = {gap:.2e}")

# Null check: break the pairing; their calibrated CKA should be 0 and p-values roughly uniform
d0, v0, l0 = COMBOS[0]
n0 = num_rows(embedding_path(d0, "vision", v0))
idx = torch.randperm(n0, generator=torch.Generator().manual_seed(7))[:min(N_SUB, n0)]
x0 = aristo_prep(load_rows(embedding_path(d0, "vision", v0), idx)).to(DEV)
y0 = aristo_prep(load_rows(embedding_path(d0, "language", l0), idx)).to(DEV)
null_cal, null_p = [], []
for r in range(20):
    sh = torch.randperm(x0.shape[0], generator=torch.Generator().manual_seed(100 + r)).to(DEV)
    c_, p_, _ = calibrate(x0, y0[sh], a_cka, K=100, alpha=ALPHA, generator=gen(200 + r))
    null_cal.append(float(c_)); null_p.append(float(p_))
check("null: calibrated CKA = 0 on shuffled pairs (>= 18/20)", sum(c == 0 for c in null_cal) >= 18, f"{sum(c == 0 for c in null_cal)}/20 zero")
check("null: p < 0.05 in <= 3/20 shuffles", sum(p < 0.05 for p in null_p) <= 3, f"{sum(p < 0.05 for p in null_p)}/20")

# Determinism: re-run one alignment in this process with the authors' code; the FOSCTTM must match exactly
if SMOKE_TEST or DRYRUN:
    sys.path.insert(0, WORK); import importlib, e3_worker; importlib.reload(e3_worker); e3_worker._setup(ROSETTA)
    from experiments.alignment import AlignmentRun, run_alignment
    t0_ = json.load(open(sorted(A_FILES)[0]))["task"]
    cfg = dict(dataset=t0_["dataset"], model_x=t0_["vision"], model_y=t0_["language"], seed=t0_["seed"], method="ours", classification=())
    for k in ("train_size", "validation_size"):
        if t0_.get(k) is not None: cfg[k] = t0_[k]
    # Different number of restart processes than the background run: checks determinism AND worker-count invariance
    w2 = max(1, RESTART_WORKERS // 2)
    _, m2 = run_alignment(AlignmentRun(**cfg), num_workers=w2, num_threads=min(4, w2))
    m1 = json.load(open(sorted(A_FILES)[0]))["metrics"]
    check(f"determinism: re-run with {w2} vs {RESTART_WORKERS} restart processes gives identical FOSCTTM", m1["foscttm"] == m2["foscttm"], f"{m1['foscttm']} vs {m2['foscttm']}")

checks = pd.DataFrame(checks); checks.to_csv(os.path.join(OUT, "checker.csv"), index=False)
print(checks.to_string(index=False))
log("checker:", (checks.result == "PASS").sum(), "PASS /", len(checks))
if SMOKE_TEST:                                   # package the smoke outputs for Claude
    import zipfile
    zp = os.path.join(OUT, "e3_smoke_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["checker.csv", "versions.json", "log.txt"] + [os.path.basename(g) for g in glob.glob(os.path.join(OUT, "worker_*.txt"))]:
            if os.path.exists(os.path.join(OUT, f)): z.write(os.path.join(OUT, f), f)
        for sub in ("align", "geometry", "metrics_gpu"):
            for f in glob.glob(os.path.join(OUT, sub, "*.json")): z.write(f, os.path.join(sub, os.path.basename(f)))
    if IN_COLAB:
        from google.colab import files; files.download(zp)

# %%
# ============================== PRE-REGISTERED ANALYSIS ==============================
assert not SMOKE_TEST, "Smoke test finished. Read the CHECKER table above, then set SMOKE_TEST = False and Run all."
from scipy import stats
from experiments import plotting   # Rosetta's own linear_fit (cluster bootstrap over model pairs)
import matplotlib; import matplotlib.pyplot as plt

# outcome: mean FOSCTTM over the 5 seeds; keep only complete combinations
al = pd.DataFrame([dict(**json.load(open(f))["task"], foscttm=json.load(open(f))["metrics"]["foscttm"])
                   for f in glob.glob(os.path.join(OUT, "align", "*.json"))])
cnt = al.groupby(["dataset", "vision", "language"]).seed.nunique()
complete = cnt[cnt == len(RUN_SEEDS)].index
outcome = al.set_index(["dataset", "vision", "language"]).loc[complete].groupby(level=[0, 1, 2]).foscttm.agg(["mean", "std"]).rename(
    columns={"mean": "foscttm", "std": "foscttm_sd_over_seeds"}).reset_index()
log(f"{len(complete)}/{len(COMBOS)} combinations have all {len(RUN_SEEDS)} seeds")

geo = pd.DataFrame([dict(dataset=r["task"]["dataset"], vision=r["task"]["vision"], language=r["task"]["language"],
                         **{f"rosetta_{k}": v for k, v in r["metrics"].items()})
                    for r in (json.load(open(f)) for f in glob.glob(os.path.join(OUT, "geometry", "*.json")))])
met_rows = []
for f in glob.glob(os.path.join(OUT, "metrics_gpu", "*.json")):
    r = json.load(open(f)); sub = pd.DataFrame(r["subsets"])
    met_rows.append(dict(dataset=r["dataset"], vision=r["vision"], language=r["language"],
                         **sub.drop(columns="subset").mean().to_dict(),
                         **{f"{c}_sd": v for c, v in sub.drop(columns="subset").std().to_dict().items()}))
met = pd.DataFrame(met_rows)
df = outcome.merge(geo, on=["dataset", "vision", "language"]).merge(met, on=["dataset", "vision", "language"])
df["cluster"] = df.vision + " / " + df.language                               # Rosetta's clusters: model pairs
df.to_csv(os.path.join(OUT, "e3_table.csv"), index=False)

PREDICTORS = {
    "rosetta_cka": "Rosetta CKA (unbiased, fp64, n=10k)", "rosetta_mutual_knn": "Rosetta mutual k-NN (n=10k)",
    "rosetta_tsi": "Rosetta TSI (n=10k)", "rosetta_qsi": "Rosetta QSI (n=10k)",
    "cka_raw": "Raw CKA (Aristotelian pipeline, n=1024)", "cka_cal": "Calibrated CKA (Aristotelian, n=1024)",
    "ucka_fp64_aristoprep": "Unbiased CKA fp64 (same inputs, n=1024)", "ucka_aristo_fp32": "Unbiased CKA, Aristotelian fp32 code (n=1024)",
    "ucka_fp64_rosettaprep": "Unbiased CKA fp64 (Rosetta prep, n=1024)",
    "mknn_raw": "Mutual k-NN raw (n=1024)", "mknn_cal": "Mutual k-NN calibrated (n=1024)",
}
fits = []
for k, lab in PREDICTORS.items():
    fit = plotting.linear_fit(df[k], df.foscttm, df.cluster)
    fits.append(dict(predictor=k, label=lab, r2=fit["r2"], pearson=fit["pearson"], spearman=fit["spearman"], n=fit["num_points"]))
fits = pd.DataFrame(fits); fits.to_csv(os.path.join(OUT, "e3_fits.csv"), index=False)
print(fits.round(3).to_string(index=False))
log("H0 replication (paper: R2 0.73, Spearman -0.90):", fits[fits.predictor == "rosetta_cka"][["r2", "spearman"]].round(3).to_dict("records"))

def cluster_boot_diff(a, b, B=B_BOOT, seed=0):
    """|Spearman(a)| - |Spearman(b)| with a cluster bootstrap over model pairs (paired: same resample for both)."""
    rng = np.random.default_rng(seed); groups = [np.flatnonzero(df.cluster.values == c) for c in pd.unique(df.cluster)]
    obs = abs(stats.spearmanr(df[a], df.foscttm).statistic) - abs(stats.spearmanr(df[b], df.foscttm).statistic)
    diffs = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        ra = stats.spearmanr(df[a].values[idx], df.foscttm.values[idx]).statistic
        rb = stats.spearmanr(df[b].values[idx], df.foscttm.values[idx]).statistic
        diffs.append(abs(ra) - abs(rb))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(a=a, b=b, delta_abs_spearman=float(obs), ci_low=float(lo), ci_high=float(hi))

tests = [
    ("H1", "ucka_fp64_aristoprep", "cka_cal"),            # main: unbiased vs Aristotelian-calibrated, same inputs, same n
    ("H1b", "cka_raw", "cka_cal"),                        # does calibration lose predictive signal?
    ("H2", "mknn_raw", "mknn_cal"),                       # calibration should not matter for mutual k-NN
    ("S1", "rosetta_cka", "ucka_fp64_rosettaprep"),       # sample size 10,000 vs 1,024 (Rosetta preprocessing)
    ("S2", "ucka_fp64_aristoprep", "ucka_aristo_fp32"),   # precision: fp64 vs their fp32 code
]
tests = pd.DataFrame([dict(test=t, **cluster_boot_diff(a, b)) for t, a, b in tests])
tests.to_csv(os.path.join(OUT, "e3_tests.csv"), index=False); print(tests.round(4).to_string(index=False))

within = pd.DataFrame([dict(dataset=dset, predictor=k, spearman=stats.spearmanr(g[k], g.foscttm).statistic, n=len(g))
                       for dset, g in df.groupby("dataset") for k in PREDICTORS])
within.to_csv(os.path.join(OUT, "e3_within_dataset.csv"), index=False)
print(within.pivot(index="predictor", columns="dataset", values="spearman").round(3))

# Figure: each predictor vs FOSCTTM
keys = ["rosetta_cka", "cka_raw", "cka_cal", "ucka_fp64_aristoprep", "mknn_raw", "mknn_cal"]
fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2), sharey=True)
for ax, k in zip(axs, keys):
    for dset, g in df.groupby("dataset"): ax.scatter(g[k], g.foscttm, s=10, label=dset)
    r = fits.set_index("predictor").loc[k]; ax.set_title(f"{PREDICTORS[k]}\nρ = {r.spearman:.2f}, R² = {r.r2:.2f}", fontsize=7.5)
    ax.grid(alpha=.3)
axs[0].set_ylabel("FOSCTTM of unpaired alignment (lower = better)"); axs[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "fig_e3_predictors.png"), dpi=140); plt.show()

# %%
# ============================== SUMMARY + PACKAGE FOR CLAUDE ==============================
import zipfile
summary = dict(versions=versions, progress=progress(), complete_combos=int(len(complete)), fits=fits.round(4).to_dict("records"),
               tests=tests.round(4).to_dict("records"), failures=[l for l in worker_logs().splitlines() if "FAILED" in l][:20])
json.dump(summary, open(os.path.join(OUT, "e3_summary.json"), "w"), indent=1, default=str)
zp = os.path.join(OUT, "e3_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["e3_table.csv", "e3_fits.csv", "e3_tests.csv", "e3_within_dataset.csv", "e3_summary.json", "versions.json", "log.txt", "checker.csv"] + [os.path.basename(g) for g in glob.glob(os.path.join(OUT, "worker_log*.txt"))]:
        if os.path.exists(os.path.join(OUT, f)): z.write(os.path.join(OUT, f), f)
    for sub in ("align", "geometry", "metrics_gpu", "figs"):
        for f in glob.glob(os.path.join(OUT, sub, "*")): z.write(f, os.path.join(sub, os.path.basename(f)))
log("packaged", zp, f"{os.path.getsize(zp) / 1e6:.1f} MB")
if IN_COLAB:
    from google.colab import files; files.download(zp)
