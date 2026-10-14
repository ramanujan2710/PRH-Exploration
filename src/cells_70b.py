# %% [md]
# # PRH audit v4: newer LLMs + exact bf16 everywhere
#
# **What this adds to the 12-LLM audit you already ran (same Drive folder, same engine v3, so existing results are reused):**
# 1. **The 8 LLMs of Huh et al.'s own "test" set:** OLMo-1B/7B, Gemma-2B/7B, Mistral-7B, Mixtral-8x7B, Llama-3-8B and Llama-3-70B.
# 2. **No quantization anywhere.** Every model runs in bf16. Models larger than the H100 (Mixtral, Llama-3-70B, LLaMA-65B) are loaded with accelerate CPU/disk offload: the bf16 weights are streamed to the GPU layer by layer, so the arithmetic is identical to an all-GPU run, just slower. A pre-flight check decides GPU / GPU+CPU / GPU+CPU+disk from the exact checkpoint size. If a model cannot fit, it is **skipped with the reason logged, never quantized**.
# 3. **LLaMA-65B is redone in bf16.** The earlier 4-bit features and results are moved to `archive_nf4/` and compared with the bf16 ones.
# 4. **A pre-registered saturation test** (written before seeing the new results). For the unbiased held-out CKA and mutual k-NN, we fit a hinge regression of score on capability (−bits per byte), with the hinge fixed in advance at the median capability of all LLMs. Prediction: positive slope below the hinge, slope ≤ 0 (CI covering 0) above it. Fitted with ViT fixed effects, with and without family fixed effects, and on log2(parameters) as a second axis. CIs by cluster bootstrap over ViTs (2,000 resamples).
#
# **Run the smoke test first** (`SMOKE_TEST = True`, the default). Choose *Run all*; it takes about 10 minutes. It extracts OLMo-1B, a new family, and checks:
# - that loading with GPU+CPU offload gives **bit-identical** features to loading fully on the GPU, on the real H100;
# - Hugging Face access to all 20 models;
# - the disk/RAM placement plan for each big model (metadata only, nothing large is downloaded).
#
# It then prints a PASS/FAIL table, downloads `v4_smoke_results.zip` and stops. Send the zip to Claude, then set `SMOKE_TEST = False` and *Run all*.
#
# **Before running:**
# - **Hugging Face access.** Gemma and Mistral/Mixtral are gated. On huggingface.co, accept the licence on the pages for `google/gemma-2b`, `google/gemma-7b`, `mistralai/Mistral-7B-v0.1` and `mistralai/Mixtral-8x7B-v0.1`, then add an `HF_TOKEN` secret in Colab (key icon on the left). Without it, those 4 are skipped and logged.
# - **Runtime.** H100 + High-RAM. Llama-3-70B (141 GB) and LLaMA-65B bf16 (~130 GB) each need that much free disk to download, and enough CPU RAM to hold whatever does not fit on the GPU. The pre-flight log prints exactly what fits.
# - **Run all.** Expect roughly 3–6 hours, dominated by downloading ~450 GB of weights (one model at a time; each is deleted before the next) and the offloaded forward passes. Finished models are skipped on re-run.
# - **Send back** `prh_audit_results.zip` (full folder) when done.

# %%
# ============================== CONFIG ==============================
import os
SMOKE_TEST = False           # True: ~10 min check of the NEW code paths + CHECKER table. False: full run (reuses the
                             # 12-LLM features already on Drive; smoke results go to a separate folder).
INCLUDE_NEW_LLMS = True      # Huh et al.'s "test" LLMs (OLMo, Gemma, Mistral/Mixtral, Llama-3)
REDO_LLAMA_65B_BF16 = True   # replace the earlier 4-bit LLaMA-65B with exact bf16 (CPU/disk offload)
N_PERMS = 200                # permutations for the calibration null (Aristotelian paper: 200)
ALPHA = 0.05
TOPK = 10                    # mutual k-NN neighbours (Huh et al.: 10)
VARIANTS = ("huh", "noclip", "std")
SAVE_TO_DRIVE = True         # keep features & results on Drive so a disconnect loses nothing
SEED = 0
RUN_ONLY = "NousResearch/Meta-Llama-3-70B"          # None: extract every LLM that is still missing, in size order.
                             # Or name the model(s) to extract in THIS session, e.g. "huggyllama/llama-65b" or
                             # ["NousResearch/Meta-Llama-3-70B"]. Everything after extraction (alignment, tests,
                             # figures) always uses every model whose features are on Drive, so you can run the big
                             # models one per session and the final session produces the complete analysis.
RUN_EXCEPT = None            # Or the opposite: extract every missing LLM EXCEPT these, e.g.
                             # ["huggyllama/llama-65b", "NousResearch/Meta-Llama-3-70B"]. Can be combined with RUN_ONLY.

LLMS_FULL = ["bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7", "bigscience/bloomz-3b",
             "bigscience/bloomz-7b1", "openlm-research/open_llama_3b", "openlm-research/open_llama_7b",
             "openlm-research/open_llama_13b", "huggyllama/llama-7b", "huggyllama/llama-13b",
             "huggyllama/llama-30b", "huggyllama/llama-65b"]
VISION_FULL = ["vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k",
               "vit_base_patch16_224.augreg_in21k", "vit_large_patch16_224.augreg_in21k",
               "vit_base_patch16_224.mae", "vit_large_patch16_224.mae", "vit_huge_patch14_224.mae",
               "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
               "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m",
               "vit_base_patch16_clip_224.laion2b", "vit_large_patch14_clip_224.laion2b",
               "vit_huge_patch14_clip_224.laion2b", "vit_base_patch16_clip_224.laion2b_ft_in12k",
               "vit_large_patch14_clip_224.laion2b_ft_in12k", "vit_huge_patch14_clip_224.laion2b_ft_in12k"]
LLMS_NEW = ["allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf", "google/gemma-2b", "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1", "mistralai/Mixtral-8x7B-v0.1",
            "NousResearch/Meta-Llama-3-8B", "NousResearch/Meta-Llama-3-70B"]   # = Huh et al.'s "test" modelset
LEGACY_NF4 = {"huggyllama/llama-65b"}   # extracted in 4-bit by notebook v3; re-extracted here in bf16

if SMOKE_TEST:
    LLMS = ["bigscience/bloomz-560m", "bigscience/bloomz-1b1", "allenai/OLMo-1B-hf"]   # OLMo = a new family, ungated
    VISION = ["vit_tiny_patch16_224.augreg_in21k", "vit_small_patch14_dinov2.lvd142m"]
    N_PERMS = 50
else:
    LLMS = LLMS_FULL + (LLMS_NEW if INCLUDE_NEW_LLMS else [])
    VISION = VISION_FULL

if os.environ.get("PRH_DRYRUN") == "1":          # local test only; never set on Colab
    from dryrun_support import DRY_LLMS, DRY_VISION
    LLMS, VISION, N_PERMS = DRY_LLMS, DRY_VISION, 20

def llm_family(m):
    m = m.lower()
    for key, fam in (("bloomz", "BLOOMZ"), ("open_llama", "OpenLLaMA"), ("huggyllama", "LLaMA"), ("olmo", "OLMo"),
                     ("gemma", "Gemma"), ("mistral", "Mistral"), ("mixtral", "Mistral"), ("llama-3", "Llama-3")):
        if key in m: return fam
    return "other"

# %%
# ============================== SETUP ==============================
import os, sys, subprocess, json, time, glob, shutil
DRYRUN = os.environ.get("PRH_DRYRUN") == "1"     # local test mode (tiny random models); ignore on Colab
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm", "accelerate", "psutil", "huggingface_hub",
                    "datasets", "sentencepiece", "protobuf"], check=False)
    if SAVE_TO_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # optional: add HF_TOKEN in Colab secrets
    except Exception:
        pass

ROOT = os.environ.get("PRH_ROOT") or ("/content/drive/MyDrive/prh_audit" if (IN_COLAB and SAVE_TO_DRIVE) else "/content/prh_audit")
RUN = os.path.join(ROOT, "smoke" if SMOKE_TEST else "full")
FEAT_DIR, RES_DIR, MAT_DIR, FIG_DIR = (os.path.join(RUN, d) for d in ("features", "results", "matrices", "figs"))
for d in (FEAT_DIR, RES_DIR, MAT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

import torch, numpy as np, pandas as pd
torch.backends.cuda.matmul.allow_tf32 = False      # exact fp32 for the metrics
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
print("run dir:", RUN)
import psutil
print("free disk (GB):", round(shutil.disk_usage("/").free / 1e9, 1), "| free CPU RAM (GB):", round(psutil.virtual_memory().available / 1e9, 1),
      "| GPU memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if DEV == "cuda" else 0)
LOG = os.path.join(RUN, "log.txt")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s); open(LOG, "a").write(s + "\n")

# %%
# ============================== ENGINE ==============================
__ENGINE__

# %%
# ============================== EXTRACTION CODE ==============================
__EXTRACT_V4__

# %%
# ============================== DATA: WIT-1024 (Huh et al.) ==============================
if DRYRUN:
    from dryrun_support import make_dataset
    images, texts = make_dataset()
else:
    from datasets import load_dataset
    ds = load_dataset("minhuh/prh", revision="wit_1024", split="train")
    texts = [str(x["text"][0]) for x in ds]
    images = [x["image"] for x in ds]
N = len(texts)
log(f"dataset: {N} image-caption pairs; example caption: {texts[0][:100]!r}")

def feat_path(name):
    return os.path.join(FEAT_DIR, name.replace("/", "_") + ".pt")

# %%
# ============================== EXTRACT VISION FEATURES ==============================
for name in VISION:
    fp = feat_path(name)
    if os.path.exists(fp):
        log("vision exists, skipping:", name); continue
    t0 = time.time()
    try:
        r = extract_vision(name, images, batch_size=64, pretrained=not DRYRUN)
        torch.save(r, fp)
        log(f"vision done: {name} feats={tuple(r['feats'].shape)} ({time.time()-t0:.0f}s)")
    except Exception as e:
        log("VISION FAILED:", name, repr(e)[:300])

# %%
# ============================== EXTRACT LLM FEATURES (exact bf16, offload if needed) ==============================
# 1) Archive the earlier 4-bit LLaMA-65B (features + its alignment results) so it is recomputed in bf16.
ARCH = os.path.join(RUN, "archive_nf4"); os.makedirs(ARCH, exist_ok=True)
for name in LEGACY_NF4:
    fp = feat_path(name)
    if REDO_LLAMA_65B_BF16 and os.path.exists(fp) and torch.load(fp, mmap=True).get("precision") != "bf16":
        shutil.move(fp, os.path.join(ARCH, os.path.basename(fp)))
        tag = name.replace("/", "_")
        for f in glob.glob(os.path.join(RES_DIR, f"align_*_{tag}.csv")) + glob.glob(os.path.join(MAT_DIR, f"*_{tag}.npz")):
            shutil.move(f, os.path.join(ARCH, os.path.basename(f)))
        log(f"archived 4-bit {name} to {ARCH}; will re-extract in bf16")
# 2) Free the disk used by cached ViT weights (their features are already extracted).
for d in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--timm--*")):
    shutil.rmtree(d, ignore_errors=True)
# 3) Extract. Smallest checkpoints first, so a failure on the largest never costs the rest.
def _size(m):
    try: return weights_gb(m)[0]
    except Exception: return 1e9
def free_disk_gb():
    return shutil.disk_usage("/").free / 1e9

# Each LLM runs in its own Python process. When it exits, every byte of GPU memory, CPU RAM and every open
# (memory-mapped) weight file is released, so the deleted download really frees its disk before the next model.
# The process runs the same extraction code in the same environment, so the features are identical.
WORK_DIR = "/content/prh_work"; os.makedirs(WORK_DIR, exist_ok=True)
open(os.path.join(WORK_DIR, "prh_extract_v4.py"), "w").write(__EXTRACT_V4_STR__)
LLM_WORKER = os.path.join(WORK_DIR, "llm_worker.py")
open(LLM_WORKER, "w").write("""import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prh_extract_v4 import extract_llm
name, out_fp, plan_fp, texts_fp = sys.argv[1:5]
r = extract_llm(name, json.load(open(texts_fp)), json.load(open(plan_fp)), offload_dir="/content/offload")
torch.save(r, out_fp + ".tmp"); os.replace(out_fp + ".tmp", out_fp)   # atomic: never a half-written file
""")
TEXTS_FP = os.path.join(WORK_DIR, "texts.json"); json.dump(texts, open(TEXTS_FP, "w"))

todo = [m for m in LLMS if not os.path.exists(feat_path(m))]
if RUN_ONLY:
    only = [RUN_ONLY] if isinstance(RUN_ONLY, str) else list(RUN_ONLY)
    unknown = [m for m in only if m not in LLMS]
    assert not unknown, f"RUN_ONLY names models not in the model list: {unknown}"
    todo = [m for m in todo if m in only]
    log("RUN_ONLY:", only, "| still to extract:", todo or "nothing (features already on Drive)")
if RUN_EXCEPT:
    excl = [RUN_EXCEPT] if isinstance(RUN_EXCEPT, str) else list(RUN_EXCEPT)
    unknown = [m for m in excl if m not in LLMS]
    assert not unknown, f"RUN_EXCEPT names models not in the model list: {unknown}"
    todo = [m for m in todo if m not in excl]
    log("RUN_EXCEPT:", excl, "| still to extract:", todo or "nothing (features already on Drive)")
# Start from a clean disk: remove any LLM weights left by an earlier session (their features are on Drive).
if not DRYRUN and todo:
    d0 = free_disk_gb(); clear_hf_cache(None)
    log(f"cleared old downloads: free disk {d0:.0f} -> {free_disk_gb():.0f} GB")
for name in sorted(todo, key=_size):
    try:
        plan = plan_placement(name) if not DRYRUN else dict(mode="gpu", weights_gb=0.0)
    except Exception as e:      # gated repo without accepted licence / token, or repo moved
        log("LLM FAILED (pre-flight):", name, repr(e)[:300]); continue
    log("pre-flight:", name, {k: v for k, v in plan.items() if k != "max_memory"}, "max_memory:", plan.get("max_memory"))
    if plan["mode"] == "skip":
        log(f"SKIPPING {name}: {plan['reason']}"); continue
    gc.collect(); torch.cuda.empty_cache()          # leave the GPU to the worker process
    t0, disk0 = time.time(), free_disk_gb()
    plan_fp = os.path.join(WORK_DIR, "plan.json"); json.dump(plan, open(plan_fp, "w"))
    wlog = os.path.join(RUN, "llm_worker_" + name.replace("/", "_") + ".txt")
    with open(wlog, "w") as fh:
        rc = subprocess.run([sys.executable, LLM_WORKER, name, feat_path(name), plan_fp, TEXTS_FP],
                            stdout=fh, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")).returncode
    # The process has exited, so nothing holds the weight files open: deleting them frees the disk now.
    if not DRYRUN:
        clear_hf_cache(name)
    shutil.rmtree("/content/offload", ignore_errors=True)
    disk1 = free_disk_gb()
    if rc == 0 and os.path.exists(feat_path(name)):
        r = torch.load(feat_path(name), mmap=True)
        log(f"llm done: {name} feats={tuple(r['feats'].shape)} bpb={r['bpb']:.3f} placement={r['placement']} "
            f"batch={r['batch_size']} seq={r['seq_len']} ({time.time()-t0:.0f}s) | free disk {disk0:.0f} -> {disk1:.0f} GB")
        del r
    else:
        tail = open(wlog, errors="ignore").read().replace("\r", "\n").strip().splitlines()[-12:]
        log(f"LLM FAILED: {name} (exit code {rc}); last lines of {wlog}:\n    " + "\n    ".join(tail))
    if disk1 < disk0 - 5:
        log(f"WARNING: disk not recovered after {name} ({disk0:.0f} -> {disk1:.0f} GB free). Largest folders:")
        log(subprocess.run("du -xh --max-depth=3 /root /content /tmp 2>/dev/null | sort -rh | head -12",
                           shell=True, capture_output=True, text=True).stdout)

# %%
# ============================== OUTLIER / SPECTRUM DIAGNOSTICS ==============================
diag_fp = os.path.join(RES_DIR, "diagnostics.csv")
rows = []
for name, modality in [(m, "language") for m in LLMS] + [(m, "vision") for m in VISION]:
    if not os.path.exists(feat_path(name)): continue
    f = torch.load(feat_path(name))["feats"].to(DEV).float()
    for r in layer_diagnostics(f):
        rows.append(dict(model=name, modality=modality, n_layers=f.shape[1], **r))
    for v in VARIANTS:                                   # effective rank of what the metrics actually see
        for l, X in enumerate(preprocess(f, v)):
            Xc = X - X.mean(0, keepdim=True); G = Xc @ Xc.T
            rows.append(dict(model=name, modality=modality, layer=l, variant=v,
                             reff=(torch.trace(G) ** 2 / (G * G).sum()).item()))
    del f; torch.cuda.empty_cache()
diag = pd.DataFrame(rows); diag.to_csv(diag_fp, index=False)
log("diagnostics saved:", diag_fp, len(diag), "rows")

# %%
# ============================== ALIGNMENT: every pair, every layer pair, 3 preprocessings ==============================
g = torch.Generator().manual_seed(SEED)
PERMS = [torch.randperm(N, generator=g).to(DEV) for _ in range(N_PERMS)]
_split = torch.randperm(N, generator=g)
HALF = (_split[: N // 2].to(DEV), _split[N // 2:].to(DEV))
PERMS_HALF = [torch.randperm(N - N // 2, generator=g).to(DEV) for _ in range(N_PERMS)]
ENGINE_VERSION = "v3"   # v3: fp64 unbiased CKA, NaN for degenerate layers, held-out calibrated CKA (selected by unbiased CKA)
avail_llm = [m for m in LLMS if os.path.exists(feat_path(m))]
avail_vis = [m for m in VISION if os.path.exists(feat_path(m))]
log(f"aligning {len(avail_llm)} LLMs x {len(avail_vis)} vision models x {len(VARIANTS)} variants")

def side_stats(name, variant):
    f = torch.load(feat_path(name))["feats"].to(DEV).float()
    s = SideStats(preprocess(f, variant), k=TOPK, half_idx=HALF)
    del f; return s

for variant in VARIANTS:
    todo = [m for m in avail_llm if not os.path.exists(os.path.join(RES_DIR, f"align_{ENGINE_VERSION}_{variant}_{m.replace('/', '_')}.csv"))]
    if not todo:
        log("variant done:", variant); continue
    t0 = time.time()
    vis_stats = {v: side_stats(v, variant) for v in avail_vis}
    log(f"[{variant}] vision stats built ({time.time()-t0:.0f}s)")
    for lm in todo:
        t1 = time.time(); xs = side_stats(lm, variant); rows, mats = [], {}
        for vm, ys in vis_stats.items():
            out, m = align_pair(xs, ys, PERMS, alpha=ALPHA, perms_half=PERMS_HALF)
            rows.append(dict(variant=variant, llm=lm, vision=vm, **out))
            for k_, a in m.items():
                mats[f"{vm}__{k_}"] = a
        pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, f"align_{ENGINE_VERSION}_{variant}_{lm.replace('/', '_')}.csv"), index=False)
        np.savez_compressed(os.path.join(MAT_DIR, f"{ENGINE_VERSION}_{variant}_{lm.replace('/', '_')}.npz"), **mats)
        log(f"[{variant}] {lm}: {len(rows)} pairs ({time.time()-t1:.0f}s)")
        del xs; torch.cuda.empty_cache()
    del vis_stats; torch.cuda.empty_cache()

# %%
# ============================== ANALYSIS ==============================
import matplotlib; import matplotlib.pyplot as plt
res = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES_DIR, f"align_{ENGINE_VERSION}_*.csv"))], ignore_index=True)
meta = []
for m in avail_llm:
    d = torch.load(feat_path(m))
    meta.append(dict(llm=m, bpb=d["bpb"], loss=d["loss"], llm_params=d["num_params"], llm_family=llm_family(m),
                     precision=d.get("precision", "bf16"), placement=d.get("placement", "gpu")))
meta = pd.DataFrame(meta); meta.to_csv(os.path.join(RES_DIR, "llm_meta.csv"), index=False)
def vfam(v):
    return ("clip_ft" if "ft_in12k" in v else "clip" if "clip" in v else "dinov2" if "dinov2" in v
            else "mae" if ".mae" in v else "in21k")
res = res.merge(meta, on="llm"); res["vision_family"] = res.vision.map(vfam)
# predicted aggregate threshold: chance offset of the highest-effective-rank layer on each side
_rmax = diag[diag.variant.notna()].groupby(["model", "variant"]).reff.max().reset_index()
res = (res.merge(_rmax.rename(columns={"model": "llm", "reff": "rmax_x"}), on=["llm", "variant"])
          .merge(_rmax.rename(columns={"model": "vision", "reff": "rmax_y"}), on=["vision", "variant"]))
res["tau_pred"] = np.sqrt(res.rmax_x * res.rmax_y) / N
res.to_csv(os.path.join(RES_DIR, "alignment_all.csv"), index=False)

METRICS = ["cka_max", "cka_cal", "cka_heldout_cal", "ucka_max", "ucka_heldout", "mknn_max", "mknn_cal"]
LABEL = dict(cka_max="CKA (raw, Huh)", cka_cal="CKA calibrated (Aristotelian)", cka_heldout_cal="CKA calibrated (held-out layer)", ucka_max="Unbiased CKA (max)",
             ucka_heldout="Unbiased CKA (held-out layer)", mknn_max="Mutual k-NN (raw)", mknn_cal="Mutual k-NN calibrated")
from scipy.stats import pearsonr, spearmanr
trend = []
for v in VARIANTS:
    for met in METRICS:
        d = res[res.variant == v]
        if met not in d or len(d.llm.unique()) < 3: continue
        rs = [pearsonr(-g_.bpb, g_[met])[0] for _, g_ in d.groupby("vision") if g_[met].std() > 0]
        ss = [spearmanr(-g_.bpb, g_[met])[0] for _, g_ in d.groupby("vision") if g_[met].std() > 0]
        mean_by_llm = d.groupby("llm")[met].mean().reindex(meta.sort_values("bpb", ascending=False).llm)
        trend.append(dict(variant=v, metric=met, pearson_vs_capability=np.nanmean(rs) if rs else np.nan,
                          spearman_vs_capability=np.nanmean(ss) if ss else np.nan,
                          mean_score=d[met].mean(), weakest_llm_score=mean_by_llm.iloc[0], strongest_llm_score=mean_by_llm.iloc[-1]))
trend = pd.DataFrame(trend); trend.to_csv(os.path.join(RES_DIR, "trend_table.csv"), index=False)
print(trend.round(3).to_string())

# Figure 1: mean score (over vision models) vs LLM capability, every metric x variant
fig, axs = plt.subplots(len(VARIANTS), len(METRICS), figsize=(3.1 * len(METRICS), 2.6 * len(VARIANTS)), squeeze=False)
for i, v in enumerate(VARIANTS):
    d = res[res.variant == v].groupby(["llm", "bpb", "llm_family"])[METRICS].mean().reset_index()
    for j, met in enumerate(METRICS):
        ax = axs[i, j]
        for fam, gf in d.groupby("llm_family"):
            gf = gf.sort_values("bpb"); ax.plot(gf.bpb, gf[met], "o-", ms=3, label=fam)
        ax.invert_xaxis(); ax.grid(alpha=.3)
        if i == 0: ax.set_title(LABEL[met], fontsize=8)
        if j == 0: ax.set_ylabel(f"{v}\nscore", fontsize=8)
        if i == len(VARIANTS) - 1: ax.set_xlabel("LLM bits/byte (better →)", fontsize=8)
axs[0, 0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_trends.png"), dpi=130); plt.show()

# Figure 2: effective rank / n and outlier share per LLM (what decides the E1 calibration penalty)
dl = diag[(diag.modality == "language") & diag.variant.isna()].merge(meta, left_on="model", right_on="llm")
dr = diag[diag.variant.notna()].copy()
fig, axs = plt.subplots(1, 4, figsize=(17, 3.3))
for v in VARIANTS:
    s = dr[(dr.variant == v) & (dr.modality == "language")].groupby("model").reff.median().reindex(meta.llm)
    axs[0].plot(meta.bpb, s / N, "o", label=v)
axs[0].set_title("LLM effective rank / n (median over layers)", fontsize=9); axs[0].invert_xaxis(); axs[0].legend(fontsize=7)
s = dl.groupby("model").top1_var_share.max().reindex(meta.llm)
axs[1].plot(meta.bpb, s, "o"); axs[1].set_title("LLM: max over layers of top-1 dim variance share", fontsize=9); axs[1].invert_xaxis()
d = res[res.variant == "huh"]
dm = d.groupby(["llm", "bpb"])[["cka_tau", "cka_tau_fixedpair", "cka_heldout_tau"]].mean().reset_index()
for c_, lab in [("cka_tau", "aggregate (max over all layer pairs)"), ("cka_tau_fixedpair", "selected pair only"), ("cka_heldout_tau", "held-out pair (n/2)")]:
    axs[2].plot(dm.bpb, dm[c_], "o", label=lab)
axs[2].set_xlabel("LLM bits/byte (better →)"); axs[2].invert_xaxis(); axs[2].set_ylabel("95% chance threshold τ")
axs[2].set_title("Calibration thresholds (huh variant)", fontsize=9); axs[2].legend(fontsize=7)
axs[3].scatter(res.tau_pred, res.cka_tau, s=8); lim = [0, max(res.tau_pred.max(), res.cka_tau.max()) * 1.05]
axs[3].plot(lim, lim, "k:"); axs[3].set_xlabel("√(max-rank LLM layer × max-rank vision layer) / n"); axs[3].set_ylabel("observed aggregate τ")
axs[3].set_title("Threshold set by highest-rank layers?", fontsize=9)
for a in axs[:2]: a.set_xlabel("LLM bits/byte (better →)")
for a in axs: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_effrank_outliers.png"), dpi=130); plt.show()

# %%
# ============================== CHECKER (smoke test; read the PASS/FAIL table) ==============================
if SMOKE_TEST:
    checks = []
    def check(name, ok, detail=""):
        checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=str(detail)[:200]))
    lg = open(LOG).read()
    check("no FAILED extraction in this run", "FAILED" not in lg, lg[lg.find("FAILED"):][:200] if "FAILED" in lg else "")
    check("all smoke LLMs extracted", all(os.path.exists(feat_path(m)) for m in LLMS), [m for m in LLMS if not os.path.exists(feat_path(m))])
    check("no NaN/inf in any alignment score", bool(np.isfinite(res[["cka_max", "cka_cal", "ucka_max", "ucka_heldout", "mknn_max"]].values).all()))
    check("threshold formula still exact (r > 0.99)", float(np.corrcoef(res.tau_pred, res.cka_tau)[0, 1]) > 0.99,
          f"r = {np.corrcoef(res.tau_pred, res.cka_tau)[0, 1]:.4f}")
    olmo = meta[meta.llm == "allenai/OLMo-1B-hf"]
    check("new family loads and gives a sane bpb (0.5 < bpb < 3)", len(olmo) == 1 and 0.5 < float(olmo.bpb.iloc[0]) < 3,
          olmo[["bpb", "llm_params", "placement"]].to_dict("records"))

    # Offload exactness on the real GPU: OLMo-1B fully on GPU vs forced GPU+CPU offload, 64 captions
    plan_gpu = plan_placement("allenai/OLMo-1B-hf")
    plan_off = dict(plan_gpu, mode="gpu+cpu", max_memory={0: "1GiB", "cpu": "60GiB"})
    a_ = extract_llm("allenai/OLMo-1B-hf", texts[:64], plan_gpu)
    b_ = extract_llm("allenai/OLMo-1B-hf", texts[:64], plan_off)
    diff = (a_["feats"].float() - b_["feats"].float()).abs().max().item()
    check("offload is exact: GPU vs GPU+CPU features identical", diff == 0.0 and a_["bpb"] == b_["bpb"],
          f"max |diff| = {diff:.3e}; bpb {a_['bpb']:.6f} vs {b_['bpb']:.6f}; placements {a_['placement']} / {b_['placement']}")
    clear_hf_cache("allenai/OLMo-1B-hf")

    # Access and placement plan for every model of the full run (metadata only, nothing is downloaded)
    from huggingface_hub import auth_check
    plan_rows = []
    for m in LLMS_FULL + LLMS_NEW:
        row = dict(model=m)
        try:
            auth_check(m); row["access"] = "ok"
        except Exception as e:
            row["access"] = "NO ACCESS: " + type(e).__name__
        try:
            p_ = plan_placement(m); row.update(mode=p_["mode"], weights_gb=p_["weights_gb"], reason=p_.get("reason", ""))
        except Exception as e:
            row.update(mode="?", reason=repr(e)[:80])
        plan_rows.append(row)
    plan_df = pd.DataFrame(plan_rows); plan_df.to_csv(os.path.join(RES_DIR, "preflight_plan.csv"), index=False)
    print(plan_df.to_string(index=False))
    check("Hugging Face access to every model", (plan_df.access == "ok").all(),
          "accept the licence + set HF_TOKEN for: " + ", ".join(plan_df[plan_df.access != "ok"].model))
    check("every model fits (GPU, GPU+CPU or +disk) on this runtime", (plan_df["mode"] != "skip").all(),
          "; ".join(f"{r.model}: {r.reason}" for r in plan_df[plan_df["mode"] == "skip"].itertuples()))

    checks = pd.DataFrame(checks); checks.to_csv(os.path.join(RES_DIR, "checker.csv"), index=False)
    print(checks.to_string(index=False))
    import zipfile
    zp = os.path.join(RUN, "v4_smoke_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["checker.csv", "preflight_plan.csv", "alignment_all.csv", "llm_meta.csv", "diagnostics.csv"]:
            if os.path.exists(os.path.join(RES_DIR, f)): z.write(os.path.join(RES_DIR, f), f)
        z.write(LOG, "log.txt")
    if IN_COLAB:
        from google.colab import files; files.download(zp)
    raise SystemExit("Smoke test finished: read the CHECKER table, send v4_smoke_results.zip, then set SMOKE_TEST = False and Run all.")

# %%
# ============================== PRE-REGISTERED TESTS ==============================
# Fixed before the new models were run. Unit of resampling = ViT (cluster bootstrap), B = 2000, seed 0.
B_BOOT = 200 if (SMOKE_TEST or DRYRUN) else 2000
SAT_METRICS = ["ucka_heldout", "mknn_max", "cka_max", "cka_cal"]
h = res[(res.variant == "huh") & (res.precision == "bf16")].copy()
h["cap"] = -h.bpb
h["log2p"] = np.log2(h.llm_params.astype(float))
cap_hinge = float(np.median(-meta[meta.precision == "bf16"].bpb))           # hinge fixed in advance: median capability
p_hinge = float(np.median(np.log2(meta[meta.precision == "bf16"].llm_params.astype(float))))

def _design(df, x, hinge, family_fe):
    cols = [df[x].values, np.maximum(df[x].values - hinge, 0.0)]
    dummies = [pd.get_dummies(df.vision).values.astype(float)]                 # ViT fixed effects
    if family_fe:
        fam = pd.get_dummies(df.llm_family).values.astype(float)[:, 1:]       # drop one level (collinear with ViT FE)
        dummies.append(fam)
    return np.column_stack(cols + dummies)

def hinge_fit(df, y, x, hinge, family_fe):
    X = _design(df, x, hinge, family_fe)
    beta = np.linalg.lstsq(X, df[y].values, rcond=None)[0]
    return beta[0], beta[0] + beta[1]                                         # slope below, slope above the hinge

def hinge_test(df, y, x, hinge, family_fe, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed); vits = df.vision.unique()
    lo, hi = hinge_fit(df, y, x, hinge, family_fe); bl, bh = [], []
    for _ in range(B):
        pick = rng.choice(vits, len(vits))
        bd = pd.concat([df[df.vision == v].assign(vision=f"{v}#{k}") for k, v in enumerate(pick)])
        a_, b_ = hinge_fit(bd, y, x, hinge, family_fe); bl.append(a_); bh.append(b_)
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(slope_below=lo, ci_below=ci(bl), slope_above=hi, ci_above=ci(bh))

sat = []
for x, hinge in (("cap", cap_hinge), ("log2p", p_hinge)):
    for fe in (False, True):
        for y in SAT_METRICS:
            r_ = hinge_test(h, y, x, hinge, fe)
            sat.append(dict(axis=x, family_fe=fe, metric=y, **r_))
sat = pd.DataFrame(sat); sat.to_csv(os.path.join(RES_DIR, "saturation_tests.csv"), index=False)
print(f"hinge: capability {cap_hinge:.3f} (bpb {-cap_hinge:.3f}), log2(params) {p_hinge:.2f}")
print(sat.round(4).to_string())

# Within-family slopes (families with >= 2 bf16 members), per doubling of parameters, CI over ViTs
fam_rows = []
rng = np.random.default_rng(1)
for fam, g in h.groupby("llm_family"):
    if g.llm.nunique() < 2: continue
    for y in SAT_METRICS:
        s0 = np.polyfit(g.log2p, g[y], 1)[0]; vits = g.vision.unique(); bs = []
        for _ in range(B_BOOT):
            bd = pd.concat([g[g.vision == v] for v in rng.choice(vits, len(vits))]); bs.append(np.polyfit(bd.log2p, bd[y], 1)[0])
        fam_rows.append(dict(family=fam, n_llms=g.llm.nunique(), metric=y, slope_per_doubling=s0,
                             ci_low=float(np.percentile(bs, 2.5)), ci_high=float(np.percentile(bs, 97.5))))
fam_slopes = pd.DataFrame(fam_rows); fam_slopes.to_csv(os.path.join(RES_DIR, "family_slopes.csv"), index=False)
print(fam_slopes.round(4).to_string())

# Replication of the Aristotelian statistic on their 12 LLMs (Pearson with LLM size rank, mean over ViTs)
orig = h[h.llm.isin(LLMS_FULL)].copy()
orig["size_rank"] = orig.llm.map(orig.groupby("llm").llm_params.first().rank())
replication = {m_: float(np.mean([pearsonr(g_[m_], g_.size_rank)[0] for _, g_ in orig.groupby("vision")]))
               for m_ in ["cka_max", "cka_cal", "ucka_heldout", "mknn_max", "mknn_cal"]} if orig.llm.nunique() >= 3 else {}
print("replication (paper: CKA 0.86 -> 0.45, mKNN ~0.85):", {k: round(v, 3) for k, v in replication.items()})

# 4-bit vs bf16 LLaMA-65B
nf4_cmp = None
arch_csv = glob.glob(os.path.join(RUN, "archive_nf4", "align_v3_huh_*llama-65b.csv"))
if arch_csv and "huggyllama/llama-65b" in set(h.llm):
    old = pd.read_csv(arch_csv[0]); new = h[h.llm == "huggyllama/llama-65b"]
    cols = ["cka_max", "cka_cal", "ucka_heldout", "mknn_max", "cka_tau"]
    nf4_cmp = pd.DataFrame({"nf4": old[cols].mean(), "bf16": new[cols].mean()})
    nf4_cmp.loc["bpb"] = [np.nan, float(new.bpb.iloc[0])]
    print(nf4_cmp.round(4))

# Figure 3: all LLMs, unbiased held-out CKA and mutual k-NN vs capability, with the hinge
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
mm = h.groupby(["llm", "llm_family", "bpb", "llm_params"])[SAT_METRICS].mean().reset_index()
for ax, y in zip(axs, ["ucka_heldout", "mknn_max"]):
    for fam, g in mm.groupby("llm_family"):
        g = g.sort_values("bpb"); ax.plot(g.bpb, g[y], "o-", ms=4, label=fam)
    ax.axvline(-cap_hinge, color="k", ls=":", lw=1); ax.invert_xaxis(); ax.grid(alpha=.3)
    ax.set_xlabel("LLM bits per byte on WIT captions (better →)"); ax.set_title(LABEL[y] + " (mean over ViTs); dotted = pre-registered hinge", fontsize=9)
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_saturation.png"), dpi=130); plt.show()

# %%
# ============================== SUMMARY + PACKAGE FOR CLAUDE ==============================
summary = dict(
    run="smoke" if SMOKE_TEST else "full", n=N, n_perms=N_PERMS, llms=avail_llm, vision=avail_vis,
    engine=ENGINE_VERSION, trend=trend.round(4).to_dict("records"),
    tau_pred_fit=dict(corr=float(np.corrcoef(res.tau_pred, res.cka_tau)[0, 1]), mae=float((res.tau_pred - res.cka_tau).abs().mean())),
    tau_vs_capability=res[res.variant == "huh"].groupby("llm")[["bpb", "cka_tau", "tau_pred", "rmax_x"]].mean().round(4).reset_index().to_dict("records"),
    thresholds=res.groupby("variant")[["cka_tau", "cka_tau_fixedpair", "cka_heldout_tau", "nullmax_pair_freq", "nullmax_reff_x", "nullmax_reff_y", "reff_x", "reff_y"]].median().round(4).reset_index().to_dict("records"),
    reff_over_n=dr.groupby(["modality", "variant"]).reff.describe()[["min", "50%", "max"]].div(N).round(4).reset_index().to_dict("records"),
    llm_top1_var_share=dl.groupby("model").top1_var_share.max().round(4).to_dict(),
    pred_shrink=res.groupby("variant").pred_shrink.describe()[["min", "50%", "max"]].round(4).reset_index().to_dict("records"),
    failures=[l.strip() for l in open(LOG) if "FAILED" in l or "SKIPPING" in l],
    preflight=[l.strip() for l in open(LOG) if "pre-flight" in l],
    saturation=sat.round(5).to_dict("records"), family_slopes=fam_slopes.round(5).to_dict("records"),
    replication_size_rank=replication, nf4_vs_bf16=(nf4_cmp.round(5).to_dict() if nf4_cmp is not None else None),
    llm_meta=meta.to_dict("records"),
)
json.dump(summary, open(os.path.join(RES_DIR, "summary.json"), "w"), indent=1, default=str)
zip_base = os.path.join(RUN, "prh_audit_results")
import zipfile
with zipfile.ZipFile(zip_base + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("results", "matrices", "figs"):
        for f in glob.glob(os.path.join(RUN, sub, "*")):
            if sub != "figs" and not os.path.basename(f).startswith(("diagnostics", "llm_meta", "alignment_all", "trend", "summary", "saturation", "family_slopes", ENGINE_VERSION, f"align_{ENGINE_VERSION}")): continue   # current engine only
            z.write(f, os.path.join(sub, os.path.basename(f)))
    z.write(LOG, "log.txt")
log("packaged:", zip_base + ".zip", f"{os.path.getsize(zip_base + '.zip')/1e6:.1f} MB")
if IN_COLAB:
    from google.colab import files
    files.download(zip_base + ".zip")
print(json.dumps({k: summary[k] for k in ("failures",)}, indent=1))
