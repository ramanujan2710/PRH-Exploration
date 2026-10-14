# %% [md]
# # E7 (Rosetta–prompt): does a prompted readout make decoder LLMs work for *unpaired* alignment?
#
# **Why.** In E3, Rosetta's unpaired aligner did much worse with the generative decoder (Qwen3-8B, Rosetta's "generative token pooling": FOSCTTM 0.40) than with embedding models (Qwen3-Embedding-8B 0.21, mpnet 0.21). E2-prompt showed that in the raw, unclipped geometry, a summarising prompt (PromptEOL) gives decoders a much better-organised global geometry than mean pooling (+0.13 calibrated CKA), and that mean pooling degrades with size. Rosetta's aligner works on exactly that raw geometry.
#
# **Design.** Rosetta's aligner and evaluation are used unchanged (commit `fdfad84`): Wasserstein Procrustes with geometric initialization, FOSCTTM on all 40,504 MS COCO val pairs, the same seeds as E3.
# - **New language embeddings** (final layer, no paired data used anywhere):
#   - Qwen3-8B (the checkpoint behind Rosetta's "qwen3-8b-gen") and Llama-3-8B (a second family).
#   - Each read out two ways: `avg` (mean over tokens) and `eol` (last token of `This sentence : "<caption>" means in one word:"`).
#   - Captions come from Rosetta's own corpus loaders, in their row order, and are written in their file format.
# - **Baselines from E3** (same seeds, read from `prh_audit/e3/align`): Qwen3-8B generative, Qwen3-Embedding-8B, mpnet.
# - **Grid:** DCI and DOCCI (where the decoder gap is largest and runs are fastest) × 7 vision models × 4 new language embeddings × 3 seeds = 168 alignments, plus Rosetta's geometry scores.
#
# **Pre-registered predictions** (written before any E7 result; outcome = mean FOSCTTM over the 3 seeds, 14 (corpus × vision) cells per comparison, 95% CI from a bootstrap over the 7 vision models):
# - **Q1:** Qwen3-8B `eol` < Qwen3-8B generative (Rosetta's baseline). Mean Δ < 0, CI excluding 0.
# - **Q2:** Qwen3-8B `eol` closes at least half of the gap to Qwen3-Embedding-8B: median of (gen − eol)/(gen − emb) ≥ 0.5.
# - **Q3:** `eol` < `avg` for both decoders (CI excluding 0).
# - **Q4 (secondary):** Llama-3-8B `eol` is within 0.05 FOSCTTM of the embedding models (mean over cells).
#
# **How to run.** `SMOKE_TEST = True` first: Qwen3-0.6B, DCI, 2 vision models, 1 seed, ~15 min. It ends with a CHECKER table and `e7_smoke_results.zip`. Then set `SMOKE_TEST = False` and *Run all*: ~1 h of text embedding, then ~2.5 h of alignments with ~12 running side by side. It ends with `e7_results.zip`. Everything is saved to Drive and resumes after a disconnect.

# %%
# ============================== CONFIG ==============================
import os
SMOKE_TEST = True
ROSETTA_REPO_URL, ROSETTA_COMMIT = "https://github.com/dominik-schnaus/unpaired-rosetta", "fdfad84ea488f29702a61a455be71622c4ff5d49"
EMB_REPO = "schnaus/unpaired-rosetta-embeddings"
DECODERS = {"qwen3-8b": "Qwen/Qwen3-8B", "llama3-8b": "NousResearch/Meta-Llama-3-8B"}   # file prefix -> checkpoint
READOUTS = ("avg", "eol")
CORPORA = ["DenselyCaptionedImages", "DOCCIDataset"]
N_SEEDS = 3                     # the first 3 of the paper's 5 seeds (the same seeds as E3)
TASK_PARALLEL = None            # None = CPU cores // 4, bounded by RAM (12 GB per alignment)
RESTART_WORKERS = 4
B_BOOT = 10_000
if SMOKE_TEST:
    DECODERS = {"qwen3-0.6b": "Qwen/Qwen3-0.6B"}
    CORPORA = ["DenselyCaptionedImages"]
    N_SEEDS, B_BOOT = 1, 500

# %%
# ============================== SETUP ==============================
import sys, subprocess, json, time, glob, shutil, math
IN_COLAB = "google.colab" in sys.modules
WORK = "/content" if IN_COLAB else os.environ.get("E7_WORK", "/tmp/e7")
os.makedirs(WORK, exist_ok=True)
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibmgm>=1.1.2,<2", "scikit-learn", "scipy", "seaborn",
                    "threadpoolctl", "loguru", "datasets", "huggingface_hub", "psutil", "-U", "transformers", "accelerate"], check=True)
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

ROSETTA = clone(ROSETTA_REPO_URL, ROSETTA_COMMIT, os.path.join(WORK, "unpaired-rosetta"))
if ROSETTA not in sys.path: sys.path.insert(0, ROSETTA)
if sys.version_info < (3, 14):     # Rosetta targets Python 3.14; one type hint names `pd`
    import builtins, pandas
    builtins.pd = pandas
STORAGE = os.path.join(WORK, "rosetta_storage"); os.environ["UNPAIRED_ROSETTA_ROOT"] = STORAGE
DRIVE = "/content/drive/MyDrive/prh_audit" if IN_COLAB else os.path.join(WORK, "drive")
OUT = os.path.join(DRIVE, "e7_smoke" if SMOKE_TEST else "e7")
E3_OUT = os.path.join(DRIVE, "e3")                                   # E3 baselines (same seeds)
EMB_KEEP = os.path.join(OUT, "embeddings")                            # our new language files, kept on Drive
for d in ("align", "geometry", "figs", "embeddings"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
import torch, numpy as np, pandas as pd, psutil
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = os.path.join(OUT, "log.txt")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s); open(LOG, "a").write(s + "\n")
from modalities import image_captions as IC
from modalities.image_captions import datasets as RD
from unpaired_rosetta.randomness import SEEDS
from unpaired_rosetta.embeddings import embedding_path, num_rows, load_rows
RUN_SEEDS = [int(s) for s in SEEDS][:N_SEEDS]
VISION = list(IC.VISION_MODELS)
if SMOKE_TEST: VISION = ["dinov2_vit-b14@224_mean", "ibot_vit-b16@224_mean"]
VAL = IC.VALIDATION_DATASET
NEW_LANG = [f"{p}-{r}" for p in DECODERS for r in READOUTS]
BASE_LANG = ["qwen3-8b-gen_69445574b3021b8a2733e87d8603d769", "qwen3-embedding-8b", "mpnet"]
log("device:", torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "| cpus", os.cpu_count(), "| seeds", RUN_SEEDS,
    "| new language embeddings:", NEW_LANG)

# %%
# ============================== DOWNLOAD (vision for the corpora + COCO val; mpnet for the order check) ==============================
need = [f"embeddings/{d}/vision/{v}.pt" for d in CORPORA + [VAL] for v in VISION] + \
       [f"embeddings/{d}/language/mpnet.pt" for d in CORPORA + [VAL]]
missing = [p for p in need if not os.path.exists(os.path.join(STORAGE, p))]
if missing:
    from huggingface_hub import snapshot_download
    log(f"downloading {len(missing)} embedding files")
    snapshot_download(EMB_REPO, repo_type="dataset", local_dir=STORAGE, allow_patterns=missing, max_workers=8)
assert all(os.path.exists(os.path.join(STORAGE, p)) for p in need)
log("Rosetta embeddings ready")

# %%
# ============================== TEXT EMBEDDINGS (new language files) ==============================
__E7_TEXT__

TEXTS = {d: RD.texts(d) for d in CORPORA + [VAL]}                     # Rosetta's own caption loaders, row order
for d, t in TEXTS.items():
    assert len(t) == IC.NUM_ITEMS[d], (d, len(t), IC.NUM_ITEMS[d])
    log(f"{d}: {len(t)} items, {sum(len(c) for c in t)} captions")
for prefix, ckpt in DECODERS.items():
    for d in CORPORA + [VAL]:
        names = {r: f"{prefix}-{r}" for r in READOUTS}
        kept = {r: os.path.join(EMB_KEEP, d, f"{n}.pt") for r, n in names.items()}
        if all(os.path.exists(p) for p in kept.values()):
            for r, n in names.items():
                dst = os.path.join(STORAGE, "embeddings", d, "language", f"{n}.pt")
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(kept[r], dst)
            log("exists:", prefix, d); continue
        t0 = time.time()
        out = embed_corpus(ckpt, TEXTS[d], READOUTS)
        for r, x in out.items():
            write_file(STORAGE, d, names[r], x)
            os.makedirs(os.path.join(EMB_KEEP, d), exist_ok=True); torch.save(x, kept[r])
        log(f"embedded {prefix} on {d} ({time.time() - t0:.0f}s)")
torch.cuda.empty_cache()

# %%
# ============================== WORKER SCRIPT (Rosetta's code, unchanged; as in E3) ==============================
WORKER_SOURCE = r'''__WORKER__'''

# %%
# ============================== ALIGN (background workers, ~12 side by side) ==============================
tasks = [dict(kind="geometry", dataset=d, vision=v, language=l) for d in CORPORA for v in VISION for l in NEW_LANG]
tasks += [dict(kind="align", dataset=d, vision=v, language=l, seed=s) for s in RUN_SEEDS for d in CORPORA for v in VISION for l in NEW_LANG]
def task_file(t):
    if t["kind"] == "align":
        return os.path.join(OUT, "align", f"{t['dataset']}__{t['vision']}__{t['language']}__seed{t['seed']}.json")
    return os.path.join(OUT, "geometry", f"{t['dataset']}__{t['vision']}__{t['language']}.json")
pending = [t for t in tasks if not os.path.exists(task_file(t))]
if TASK_PARALLEL is None:
    TASK_PARALLEL = max(1, min((os.cpu_count() or 4) // RESTART_WORKERS, int(psutil.virtual_memory().available / 1e9 // 12)))
worker = os.path.join(WORK, "e3_worker.py"); open(worker, "w").write(WORKER_SOURCE)
subprocess.run(["pkill", "-f", "e3_worker.py"]); time.sleep(2)
threads = str(min(4, RESTART_WORKERS)); BG = []
# Shard by task index so the seeds of one combination spread over workers (no worker holds a whole combination).
for k in range(TASK_PARALLEL):
    shard = pending[k::TASK_PARALLEL]
    if not shard: continue
    tf = os.path.join(OUT, f"tasks_{k}.json"); json.dump(shard, open(tf, "w"))
    BG.append(subprocess.Popen([sys.executable, worker, tf, OUT, ROSETTA, str(RESTART_WORKERS), f"worker_log_{k}.txt"],
                               stdout=open(os.path.join(OUT, f"worker_stdout_{k}.txt"), "a"), stderr=subprocess.STDOUT,
                               env=dict(os.environ, UNPAIRED_ROSETTA_ROOT=STORAGE, PYTHONPATH=ROSETTA, OMP_NUM_THREADS=threads,
                                        MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)))
log(f"{len(pending)} pending tasks on {len(BG)} workers")
def worker_logs():
    return "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(os.path.join(OUT, "worker_log*.txt"))))
def progress():
    na = len(glob.glob(os.path.join(OUT, "align", "*.json"))); ng = len(glob.glob(os.path.join(OUT, "geometry", "*.json")))
    done = [json.load(open(f))["seconds"] for f in glob.glob(os.path.join(OUT, "align", "*.json"))]
    total = sum(t["kind"] == "align" for t in tasks)
    eta = (np.mean(done) * (total - na) / max(1, TASK_PARALLEL) / 3600) if done else float("nan")
    return dict(align_done=na, align_total=total, geometry_done=ng, eta_hours=round(float(eta), 1))
while True:
    p = progress()
    if p["align_done"] >= p["align_total"] and p["geometry_done"] >= sum(t["kind"] == "geometry" for t in tasks): break
    if not subprocess.run(["pgrep", "-f", "e3_worker.py"], capture_output=True, text=True).stdout.strip():
        log("workers stopped with tasks remaining; see worker_log_*.txt", p); break
    log("waiting:", p); time.sleep(60 if SMOKE_TEST else 600)
log("alignment finished:", progress())

# %%
# ============================== CHECKER ==============================
checks = []
def check(name, ok, detail=""):
    checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=str(detail)[:200]))
wl = worker_logs(); check("worker: no FAILED tasks", "FAILED" not in wl, wl[wl.find("FAILED"):][:200] if "FAILED" in wl else "")
check("all alignment and geometry runs present", all(os.path.exists(task_file(t)) for t in tasks),
      sum(not os.path.exists(task_file(t)) for t in tasks))
for d in CORPORA + [VAL]:
    ref = torch.load(embedding_path(d, "language", "mpnet"), mmap=True)
    ref_mask = torch.isnan(ref.float()).all(-1)                       # which (item, caption) slots are empty in Rosetta's file
    for n in NEW_LANG:
        x = torch.load(embedding_path(d, "language", n), mmap=True)
        mask = torch.isnan(x.float()).all(-1)
        ok = (x.shape[0] == ref.shape[0]) and (x.shape[1] == ref.shape[1]) and bool(torch.equal(mask, ref_mask)) \
             and bool(torch.isfinite(x.float()[~mask]).all())
        check(f"{n} on {d}: Rosetta's shape and caption-slot pattern, finite", ok, f"{tuple(x.shape)} vs {tuple(ref.shape)}")
# Content check: unbiased CKA with a vision model on paired rows is clearly > 0, and ~0 after shuffling the rows
from unpaired_rosetta.geometry import cka as r_cka
idx = torch.arange(min(4000, IC.NUM_ITEMS[CORPORA[0]]))
xv = load_rows(embedding_path(CORPORA[0], "vision", VISION[0]), idx).double()
for n in NEW_LANG:
    yl = load_rows(embedding_path(CORPORA[0], "language", n), idx).double()
    c = float(r_cka(xv - xv.mean(0), yl - yl.mean(0)))
    perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))
    c0 = float(r_cka(xv - xv.mean(0), yl[perm] - yl[perm].mean(0)))
    check(f"{n}: CKA with {VISION[0]} > 0.05 paired, |CKA| < 0.02 shuffled", c > 0.05 and abs(c0) < 0.02, f"paired {c:.3f}, shuffled {c0:.4f}")
fos = [json.load(open(f))["metrics"] for f in glob.glob(os.path.join(OUT, "align", "*.json"))]
check("FOSCTTM in [0, 1] on all 40,504 COCO val pairs", all(0 <= m["foscttm"] <= 1 and m["num_validation_samples"] == 40504 for m in fos),
      sorted(round(m["foscttm"], 3) for m in fos)[:8])
checks = pd.DataFrame(checks); checks.to_csv(os.path.join(OUT, "checker.csv"), index=False)
print(checks.to_string(index=False)); log("checker:", (checks.result == "PASS").sum(), "PASS /", len(checks))
if SMOKE_TEST:
    import zipfile
    zp = os.path.join(OUT, "e7_smoke_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [LOG, os.path.join(OUT, "checker.csv")] + glob.glob(os.path.join(OUT, "worker_log*.txt")) + \
                 glob.glob(os.path.join(OUT, "align", "*.json")) + glob.glob(os.path.join(OUT, "geometry", "*.json")):
            z.write(f, os.path.relpath(f, OUT))
    if IN_COLAB:
        from google.colab import files; files.download(zp)
    raise SystemExit("Smoke test finished: send e7_smoke_results.zip, then set SMOKE_TEST = False and Run all.")

# %%
# ============================== PRE-REGISTERED ANALYSIS ==============================
from scipy import stats
def load_align(folder, langs):
    rows = []
    for f in glob.glob(os.path.join(folder, "align", "*.json")):
        j = json.load(open(f)); t = j["task"]
        if t["dataset"] in CORPORA and t["vision"] in VISION and t["language"] in langs and t["seed"] in RUN_SEEDS:
            rows.append(dict(dataset=t["dataset"], vision=t["vision"], language=t["language"], seed=t["seed"], foscttm=j["metrics"]["foscttm"]))
    return pd.DataFrame(rows)
al = pd.concat([load_align(OUT, NEW_LANG), load_align(E3_OUT, BASE_LANG)], ignore_index=True)
cnt = al.groupby(["dataset", "vision", "language"]).size()
assert (cnt == len(RUN_SEEDS)).all(), cnt[cnt != len(RUN_SEEDS)]
tab = al.groupby(["dataset", "vision", "language"]).foscttm.mean().unstack("language")
short = {"qwen3-8b-gen_69445574b3021b8a2733e87d8603d769": "qwen3-8b-gen"}
tab = tab.rename(columns=short); tab.to_csv(os.path.join(OUT, "e7_table.csv"))
print(tab.round(3).to_string())
rng = np.random.default_rng(0); vis = tab.index.get_level_values("vision")
def boot(vals, stat=np.mean):
    vals = pd.Series(vals, index=vis); vs = vis.unique(); bs = []
    for _ in range(B_BOOT):
        pick = rng.choice(vs, len(vs)); bs.append(stat(np.concatenate([vals[vals.index == v].values for v in pick])))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
rows = []
def contrast(name, a, b):
    d = (tab[a] - tab[b]).values; lo, hi = boot(d)
    rows.append(dict(test=name, a=a, b=b, mean_a=tab[a].mean(), mean_b=tab[b].mean(), delta=d.mean(), ci_low=lo, ci_high=hi,
                     a_better=f"{int((d < 0).sum())}/{len(d)}", wilcoxon_p=float(stats.wilcoxon(d).pvalue)))
contrast("Q1 Qwen3-8B eol vs Rosetta's generative pooling", "qwen3-8b-eol", "qwen3-8b-gen")
contrast("Q3 Qwen3-8B eol vs avg", "qwen3-8b-eol", "qwen3-8b-avg")
contrast("Q3 Llama-3-8B eol vs avg", "llama3-8b-eol", "llama3-8b-avg")
contrast("Q4 Llama-3-8B eol vs Qwen3-Embedding-8B", "llama3-8b-eol", "qwen3-embedding-8b")
contrast("Q4 Llama-3-8B eol vs mpnet", "llama3-8b-eol", "mpnet")
contrast("ref: Qwen3-8B eol vs Qwen3-Embedding-8B", "qwen3-8b-eol", "qwen3-embedding-8b")
contrast("ref: Qwen3-8B avg vs Rosetta's generative pooling", "qwen3-8b-avg", "qwen3-8b-gen")
gap = ((tab["qwen3-8b-gen"] - tab["qwen3-8b-eol"]) / (tab["qwen3-8b-gen"] - tab["qwen3-embedding-8b"])).values
lo, hi = boot(gap, np.median)
rows.append(dict(test="Q2 share of the gen -> embedding gap closed by Qwen3-8B eol (median over cells)", a="qwen3-8b-eol", b="gap",
                 delta=float(np.median(gap)), ci_low=lo, ci_high=hi, a_better=f"{int((gap >= 0.5).sum())}/{len(gap)} cells >= 0.5"))
tests = pd.DataFrame(rows); tests.to_csv(os.path.join(OUT, "e7_tests.csv"), index=False)
print(tests.round(4).to_string(index=False))
geo = pd.DataFrame([dict(**json.load(open(f))["task"], **json.load(open(f))["metrics"]) for f in glob.glob(os.path.join(OUT, "geometry", "*.json"))])
geo.to_csv(os.path.join(OUT, "e7_geometry.csv"), index=False)

import matplotlib; import matplotlib.pyplot as plt
order = ["qwen3-8b-gen", "qwen3-8b-avg", "qwen3-8b-eol", "llama3-8b-avg", "llama3-8b-eol", "qwen3-embedding-8b", "mpnet"]
fig, axs = plt.subplots(1, len(CORPORA), figsize=(6 * len(CORPORA), 3.6), sharey=True)
for ax, d in zip(np.atleast_1d(axs), CORPORA):
    sub = tab.loc[d][order]
    for v, row in sub.iterrows(): ax.plot(range(len(order)), row.values, "o-", color="#c3c2b7", lw=1, ms=3)
    ax.plot(range(len(order)), sub.mean().values, "o-", color="#2a78d6", lw=2.5, label="mean over vision models")
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right"); ax.set_title(d); ax.grid(alpha=.3)
np.atleast_1d(axs)[0].set_ylabel("FOSCTTM (lower = better)"); np.atleast_1d(axs)[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figs", "fig_e7.png"), dpi=140); plt.show()

# %%
# ============================== PACKAGE ==============================
import zipfile
zp = os.path.join(OUT, "e7_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in [LOG, os.path.join(OUT, "checker.csv")] + glob.glob(os.path.join(OUT, "e7_*.csv")) + glob.glob(os.path.join(OUT, "figs", "*.png")) + \
             glob.glob(os.path.join(OUT, "worker_log*.txt")) + glob.glob(os.path.join(OUT, "align", "*.json")) + glob.glob(os.path.join(OUT, "geometry", "*.json")):
        z.write(f, os.path.relpath(f, OUT))
log("packaged", zp)
if IN_COLAB:
    from google.colab import files; files.download(zp)
