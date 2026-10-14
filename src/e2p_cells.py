# %% [md]
# # E2-prompt: does a global (prompted) readout restore convergence with scale across 20 LLMs?
#
# **Motivation.** In E6 (Qwen3, 0.6B → 8B), reading the decoder with a PromptEOL prompt raised calibrated CKA with DINOv2 from 0.23 to 0.37 under the Aristotelian pipeline. Mean pooling (the Huh / Aristotelian readout) stayed flat. This notebook asks whether that holds across the 20 LLMs of the E2 audit. It reuses the E2 pipeline unchanged: WIT-1024, the 17 ViTs (their E2 features on Drive), all layer pairs, a permutation null on the max, 200 permutations, exact bf16, CPU offload for the largest models.
#
# **Readouts.** All are the last token of a template, at every layer:
# - `eol` (main): `This sentence : "<caption>" means in one word:"`
# - `eol_alt`: a second summarising prompt, `This image caption : "<caption>" can be summarized in one word:"`
# - `neutral` (control): the same frame without a summary request, `This sentence : "<caption>" was written down and then`
# - `repeat` (control): the caption twice, `<caption>\n<caption>`, so the last token has seen the whole caption, with no instruction
#
# The mean-pooled baseline is the E2 result itself (`prh_audit/full/results/alignment_all.csv`). Layer 0 of a fixed-final-token readout is the same vector for every caption, so zero-variance layers are dropped from the layer search, as in E6 v2.
#
# **Pre-registered predictions** (written before any E2-prompt result; primary metric = calibrated CKA, Huh preprocessing, layer search):
# - **R1 (scale):** the slope over log2(parameters), with family and ViT fixed effects, is larger for `eol` than for `mean`. Two-way bootstrap (LLMs × ViTs) CI of the difference excludes 0.
# - **R2 (capability):** the Pearson correlation of LLM means with capability (−bits per byte) is larger for `eol` than for `mean`, two-way CI excluding 0.
# - **R3 (it is the summary request):** `eol` − `neutral` > 0 (mean over all pairs, CI excludes 0), and `eol_alt` ≈ `eol` (|Δ| < 0.02).
# - **R4 (descriptive):** inside the families with a wide size range (LLaMA 7→65B, Llama-3 8→70B, Mistral 7B→Mixtral), `eol` rises where `mean` was flat.
#
# **How to run.**
# 1. `SMOKE_TEST = True` (default): BLOOMZ-560m and OLMo-1B × 2 ViTs, all readouts, 50 permutations, ~10 min. It ends with a CHECKER table and `e2p_smoke_results.zip`.
# 2. Full run: set `SMOKE_TEST = False`. Small and mid-size models first (`RUN_EXCEPT` the two biggest), then LLaMA-65B and Llama-3-70B one per session with `RUN_ONLY`, as in v4. The last session writes `e2p_results.zip`. Accept the Gemma and Mistral licences with the same HF token as before.
# 3. To save time on the 65B/70B sessions, `CONTROLS_MAX_GB` limits the control readouts to models up to that size (default 40 GB of weights). The main `eol` readout runs on all 20.

# %%
# ============================== CONFIG ==============================
import os
SMOKE_TEST = True
READOUTS = ("eol", "eol_alt", "neutral", "repeat")
CONTROLS_MAX_GB = 40.0       # eol on every model; eol_alt / neutral / repeat only on models with <= this many GB of weights
N_PERMS = 200
ALPHA = 0.05
TOPK = 10
VARIANTS_P = ("huh", "noclip")
B_BOOT = 2000
SEED = 0
RUN_ONLY = None              # e.g. "huggyllama/llama-65b": extract only this model in this session
RUN_EXCEPT = None            # e.g. ["huggyllama/llama-65b", "NousResearch/Meta-Llama-3-70B"]
LLMS_ALL = ["bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7", "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1", "openlm-research/open_llama_3b", "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b", "huggyllama/llama-7b", "huggyllama/llama-13b",
            "huggyllama/llama-30b", "huggyllama/llama-65b", "allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf",
            "google/gemma-2b", "google/gemma-7b", "mistralai/Mistral-7B-v0.1", "mistralai/Mixtral-8x7B-v0.1",
            "NousResearch/Meta-Llama-3-8B", "NousResearch/Meta-Llama-3-70B"]
VISION_ALL = ["vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k",
              "vit_base_patch16_224.augreg_in21k", "vit_large_patch16_224.augreg_in21k",
              "vit_base_patch16_224.mae", "vit_large_patch16_224.mae", "vit_huge_patch14_224.mae",
              "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
              "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m",
              "vit_base_patch16_clip_224.laion2b", "vit_large_patch14_clip_224.laion2b",
              "vit_huge_patch14_clip_224.laion2b", "vit_base_patch16_clip_224.laion2b_ft_in12k",
              "vit_large_patch14_clip_224.laion2b_ft_in12k", "vit_huge_patch14_clip_224.laion2b_ft_in12k"]
if SMOKE_TEST:
    LLMS = ["bigscience/bloomz-560m", "allenai/OLMo-1B-hf"]
    VISION = ["vit_tiny_patch16_224.augreg_in21k", "vit_small_patch14_dinov2.lvd142m"]
    N_PERMS, B_BOOT = 50, 200
else:
    LLMS, VISION = LLMS_ALL, VISION_ALL

def llm_family(m):
    m = m.lower()
    for key, fam in (("bloomz", "BLOOMZ"), ("open_llama", "OpenLLaMA"), ("huggyllama", "LLaMA"), ("olmo", "OLMo"),
                     ("gemma", "Gemma"), ("mixtral", "Mistral"), ("mistral", "Mistral"), ("llama-3", "Llama-3")):
        if key in m: return fam
    return "other"

# %%
# ============================== SETUP ==============================
import sys, subprocess, json, time, glob, shutil, math
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "timm", "accelerate", "psutil",
                    "huggingface_hub", "datasets", "sentencepiece", "protobuf"], check=False)
    from google.colab import drive; drive.mount("/content/drive")
    try:
        from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        pass
ROOT = os.environ.get("PRH_ROOT") or "/content/drive/MyDrive/prh_audit"
E2_FEAT = os.path.join(ROOT, "full", "features")                  # E2 vision features (CLS per block), reused
E2_RES = os.path.join(ROOT, "full", "results")                    # E2 mean-pooled results + bpb, reused
RUN = os.path.join(ROOT, "e2prompt_smoke" if SMOKE_TEST else "e2prompt")
FEAT_DIR, RES_DIR, FIG_DIR = (os.path.join(RUN, d) for d in ("features", "results", "figs"))
for d in (FEAT_DIR, RES_DIR, FIG_DIR): os.makedirs(d, exist_ok=True)
import torch, numpy as np, pandas as pd
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = os.path.join(RUN, "log.txt")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s); open(LOG, "a").write(s + "\n")
import transformers, psutil
log("device:", torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "| transformers", transformers.__version__,
    "| free disk GB", round(shutil.disk_usage("/").free / 1e9), "| run dir:", RUN)
assert all(os.path.exists(os.path.join(E2_FEAT, v.replace("/", "_") + ".pt")) for v in VISION), "E2 vision features not found on Drive"
assert os.path.exists(os.path.join(E2_RES, "alignment_all.csv")), "E2 results (alignment_all.csv) not found on Drive"

# %%
# ============================== ENGINE (identical to E2, v3) ==============================
__ENGINE__

# %%
# ============================== EXTRACTION CODE (E2 v4 loader + prompted readouts) ==============================
__EXTRACT_V4__

# %%
# ============================== DATA (WIT-1024, identical to E2) ==============================
from datasets import load_dataset
ds = load_dataset("minhuh/prh", revision="wit_1024", split="train")
texts = [str(x["text"][0]) for x in ds]
N = len(texts); assert N == 1024
log(f"dataset: {N} captions; first: {texts[0][:80]!r}")
def fpath(model, readout):
    return os.path.join(FEAT_DIR, f"{model.replace('/', '_')}__{readout}.pt")

# %%
# ============================== EXTRACT (one Python process per LLM, as in v4) ==============================
WORK_DIR = "/content/prh_work"; os.makedirs(WORK_DIR, exist_ok=True)
open(os.path.join(WORK_DIR, "prh_extract_v4.py"), "w").write(__EXTRACT_V4_STR__)
open(os.path.join(WORK_DIR, "e2p_extract.py"), "w").write(__E2P_STR__)
WORKER = os.path.join(WORK_DIR, "e2p_worker.py")
open(WORKER, "w").write("""import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2p_extract import extract_prompted
name, feat_dir, plan_fp, texts_fp, readouts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")
out = extract_prompted(name, json.load(open(texts_fp)), json.load(open(plan_fp)), readouts)
for r, d in out.items():
    fp = os.path.join(feat_dir, name.replace("/", "_") + "__" + r + ".pt")
    torch.save(d, fp + ".tmp"); os.replace(fp + ".tmp", fp)
""")
TEXTS_FP = os.path.join(WORK_DIR, "texts.json"); json.dump(texts, open(TEXTS_FP, "w"))
def free_disk_gb(): return shutil.disk_usage("/").free / 1e9
def _size(m):
    try: return weights_gb(m)[0]
    except Exception: return 1e9
def wanted(m):
    return [r for r in READOUTS if r == "eol" or _size(m) <= CONTROLS_MAX_GB]
todo = [m for m in LLMS if any(not os.path.exists(fpath(m, r)) for r in wanted(m))]
if RUN_ONLY:
    only = [RUN_ONLY] if isinstance(RUN_ONLY, str) else list(RUN_ONLY); todo = [m for m in todo if m in only]
if RUN_EXCEPT:
    excl = [RUN_EXCEPT] if isinstance(RUN_EXCEPT, str) else list(RUN_EXCEPT); todo = [m for m in todo if m not in excl]
log("to extract:", todo)
if todo:
    d0 = free_disk_gb(); clear_hf_cache(None); log(f"cleared old downloads: free disk {d0:.0f} -> {free_disk_gb():.0f} GB")
for name in sorted(todo, key=_size):
    rs = [r for r in wanted(name) if not os.path.exists(fpath(name, r))]
    try:
        plan = plan_placement(name)
    except Exception as e:
        log("LLM FAILED (pre-flight):", name, repr(e)[:300]); continue
    log("pre-flight:", name, {k: v for k, v in plan.items() if k != "max_memory"}, "| readouts:", rs)
    if plan["mode"] == "skip":
        log(f"SKIPPING {name}: {plan['reason']}"); continue
    import gc; gc.collect(); torch.cuda.empty_cache()
    t0 = time.time(); plan_fp = os.path.join(WORK_DIR, "plan.json"); json.dump(plan, open(plan_fp, "w"))
    wlog = os.path.join(RUN, "worker_" + name.replace("/", "_") + ".txt")
    with open(wlog, "w") as fh:
        rc = subprocess.run([sys.executable, WORKER, name, FEAT_DIR, plan_fp, TEXTS_FP, ",".join(rs)], stdout=fh, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")).returncode
    clear_hf_cache(name); shutil.rmtree("/content/offload", ignore_errors=True)
    if rc == 0 and all(os.path.exists(fpath(name, r)) for r in rs):
        d = torch.load(fpath(name, rs[0]), mmap=True)
        log(f"llm done: {name} readouts={rs} feats={tuple(d['feats'].shape)} placement={d['placement']} batch={d['batch_size']} "
            f"({time.time()-t0:.0f}s) | free disk {free_disk_gb():.0f} GB")
    else:
        tail = open(wlog, errors="ignore").read().replace("\r", "\n").strip().splitlines()[-12:]
        log(f"LLM FAILED: {name} (exit code {rc}):\n    " + "\n    ".join(tail))

# %%
# ============================== ALIGN (E2 engine, zero-variance layers dropped) ==============================
g = torch.Generator().manual_seed(SEED)
PERMS = [torch.randperm(N, generator=g).to(DEV) for _ in range(N_PERMS)]
_split = torch.randperm(N, generator=g)
HALF = (_split[: N // 2].to(DEV), _split[N // 2:].to(DEV))
PERMS_HALF = [torch.randperm(N - N // 2, generator=g).to(DEV) for _ in range(N_PERMS)]
DROPPED = {}
def stats(path, variant, tag):
    f = torch.load(path)["feats"].to(DEV).float()
    layers = preprocess(f, variant)
    keep = [l for l, X in enumerate(layers) if float((X - X.mean(0)).pow(2).sum()) > 1e-6 * X.shape[0]]
    if len(keep) < len(layers): DROPPED[f"{tag}|{variant}"] = [l for l in range(len(layers)) if l not in keep]
    s = SideStats([layers[l] for l in keep], k=TOPK, half_idx=HALF); del f, layers; return s

for variant in VARIANTS_P:
    vs = None
    for r in READOUTS:
        out_csv = os.path.join(RES_DIR, f"align_{r}_{variant}.csv")
        done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["llm"])
        todo_a = [m for m in LLMS if os.path.exists(fpath(m, r)) and m not in set(done.llm)]
        if not todo_a: continue
        if vs is None:
            vs = {v: stats(os.path.join(E2_FEAT, v.replace("/", "_") + ".pt"), variant, v) for v in VISION}
        for m in todo_a:
            t1 = time.time(); xs = stats(fpath(m, r), variant, f"{m}/{r}"); rows = []
            for v, ys in vs.items():
                o, _ = align_pair(xs, ys, PERMS, alpha=ALPHA, perms_half=PERMS_HALF)
                rows.append(dict(variant=variant, readout=r, llm=m, vision=v, **{a: (str(b) if isinstance(b, tuple) else b) for a, b in o.items()}))
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(out_csv, index=False)
            log(f"[{variant}/{r}] {m}: {len(rows)} pairs ({time.time()-t1:.0f}s)")
            del xs; torch.cuda.empty_cache()
    del vs; torch.cuda.empty_cache()
log("zero-variance layers dropped:", DROPPED)

# %%
# ============================== CHECKER ==============================
checks = []
def check(name, ok, detail=""):
    checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=str(detail)[:200]))
wl = open(LOG).read()
check("no extraction failures", "FAILED" not in wl, [l for l in wl.splitlines() if "FAILED" in l][:3])
exp = [(m, r) for m in LLMS for r in wanted(m)]
missing = [f"{m}/{r}" for m, r in exp if not os.path.exists(fpath(m, r))]
check("feature files present (this session's models)", not missing, missing[:6])
bad, lt = [], {}
for m, r in exp:
    if os.path.exists(fpath(m, r)):
        d = torch.load(fpath(m, r), mmap=True)
        if d["feats"].shape[0] != N or not torch.isfinite(d["feats"].float()).all(): bad.append(f"{m}/{r}")
        lt[f"{m.split('/')[-1]}/{r}"] = (d["last_token"], d["last_token_unique"])
check("features: 1024 rows, all finite", not bad, bad[:5])
fixed_end = {k: v for k, v in lt.items() if not k.endswith("/repeat")}
expected_end = {"eol": '"', "eol_alt": '"', "neutral": "then"}
ok_end = all(u == 1 and v[0].strip().endswith(expected_end[k.split("/")[-1]]) for k, v in fixed_end.items() for u in [v[1]])
check("prompted readouts sit on the template's final token for every caption", ok_end, {k: v for k, v in list(fixed_end.items())[:6]})
for m in LLMS:
    rs = [r for r in wanted(m) if os.path.exists(fpath(m, r))]
    if len(rs) >= 2:
        a, b = (torch.load(fpath(m, r))["feats"][:, -1].float() for r in rs[:2])
        check(f"readouts differ ({m.split('/')[-1]}: {rs[0]} vs {rs[1]})", not torch.allclose(a, b), float((a - b).abs().max()))
res_p = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES_DIR, "align_*.csv"))], ignore_index=True)
check("calibrated CKA in [0, 1] and <= raw, no NaN", bool(((res_p.cka_cal >= 0) & (res_p.cka_cal <= res_p.cka_max + 1e-9)).all()) and not res_p.cka_max.isna().any())
tp = np.sqrt(res_p.nullmax_reff_x * res_p.nullmax_reff_y) / N
check("threshold = chance level of the highest-rank layer pair (r > 0.99)", float(np.corrcoef(tp, res_p.cka_tau)[0, 1]) > 0.99,
      round(float(np.corrcoef(tp, res_p.cka_tau)[0, 1]), 5))
m0 = next(m for m in LLMS if os.path.exists(fpath(m, "eol"))); v0 = VISION[0]
xs = stats(fpath(m0, "eol"), "huh", "null-check"); zero = 0
fv = torch.load(os.path.join(E2_FEAT, v0.replace("/", "_") + ".pt"))["feats"].to(DEV).float()
for s_ in range(10):
    p = torch.randperm(N, generator=torch.Generator().manual_seed(100 + s_)).to(DEV)
    ys = SideStats(preprocess(fv[p], "huh"), k=TOPK, half_idx=HALF)
    o, _ = align_pair(xs, ys, PERMS, alpha=ALPHA); zero += int(o["cka_cal"] == 0)
check("null: shuffled pairs give calibrated CKA = 0 (>= 9/10)", zero >= 9, f"{zero}/10")
e2 = pd.read_csv(os.path.join(E2_RES, "alignment_all.csv"))
check("E2 mean-pooled baseline available for these LLMs", set(LLMS) <= set(e2.llm), sorted(set(LLMS) - set(e2.llm)))
checks = pd.DataFrame(checks); checks.to_csv(os.path.join(RUN, "checker.csv"), index=False)
print(checks.to_string(index=False)); log("checker:", (checks.result == "PASS").sum(), "PASS /", len(checks))
if SMOKE_TEST:
    import zipfile
    zp = os.path.join(RUN, "e2p_smoke_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [LOG, os.path.join(RUN, "checker.csv")] + glob.glob(os.path.join(RES_DIR, "*.csv")) + glob.glob(os.path.join(RUN, "worker_*.txt")):
            z.write(f, os.path.relpath(f, RUN))
    if IN_COLAB:
        from google.colab import files; files.download(zp)
    raise SystemExit("Smoke test finished: send e2p_smoke_results.zip, then set SMOKE_TEST = False.")

# %%
# ============================== PRE-REGISTERED ANALYSIS (two-way bootstrap: LLMs x ViTs) ==============================
import matplotlib; import matplotlib.pyplot as plt
meta = pd.read_csv(os.path.join(E2_RES, "llm_meta.csv"))[["llm", "bpb", "llm_params"]]
base_cols = ["variant", "llm", "vision", "cka_max", "cka_cal", "cka_tau", "ucka_heldout", "mknn_max"]
mean_rows = e2[(e2.variant.isin(VARIANTS_P)) & (e2.precision == "bf16")][base_cols].assign(readout="mean")
allr = pd.concat([mean_rows, res_p[base_cols + ["readout"]]], ignore_index=True).merge(meta, on="llm")
allr["family"] = allr.llm.map(llm_family); allr["log2p"] = np.log2(allr.llm_params.astype(float)); allr["cap"] = -allr.bpb
allr.to_csv(os.path.join(RES_DIR, "e2p_all.csv"), index=False)
MET = ["cka_cal", "ucka_heldout", "mknn_max", "cka_max", "cka_tau"]
rng = np.random.default_rng(SEED)

def mat(variant, readout, y, llms, vits):
    d = allr[(allr.variant == variant) & (allr.readout == readout)].pivot_table(index="llm", columns="vision", values=y)
    return d.reindex(index=llms, columns=vits).values
lmeta = allr.groupby("llm").agg(log2p=("log2p", "first"), cap=("cap", "first"), family=("family", "first"))

def slope_fe_arr(Y, Li, Vj, logp, fam):
    """slope of Y on log2(params) with ViT (column) and family fixed effects; rows Li, columns Vj (copies = own levels)."""
    nL, nV = len(Li), len(Vj)
    y = Y[np.ix_(Li, Vj)].ravel()
    x = np.repeat(logp[Li], nV)
    vfe = np.tile(np.eye(nV), (nL, 1))
    fl = fam[Li]; levels = np.unique(fl)[1:]
    ffe = np.repeat(np.stack([(fl == lv).astype(float) for lv in levels], 1), nV, 0) if len(levels) else np.zeros((nL * nV, 0))
    X = np.column_stack([x, vfe, ffe]); ok = ~np.isnan(y)
    return float(np.linalg.lstsq(X[ok], y[ok], rcond=None)[0][0])
def cap_r_arr(Y, Li, Vj, cap):
    m = np.nanmean(Y[np.ix_(Li, Vj)], 1); return float(np.corrcoef(m, cap[Li])[0, 1])

def boot_pair(stat, nL, nV, B=B_BOOT):
    out = []
    for _ in range(B):
        Li, Vj = rng.integers(0, nL, nL), rng.integers(0, nV, nV)
        try: out.append(stat(Li, Vj))
        except Exception: out.append(np.nan)
    return np.array(out)
def ci(a): return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))

rows = []
vits = sorted(allr.vision.unique())
for variant in VARIANTS_P:
    # R1 / R2: eol vs mean on the LLMs that have the eol readout
    llms = sorted(set(allr[(allr.readout == "eol") & (allr.variant == variant)].llm) & set(allr[allr.readout == "mean"].llm))
    if len(llms) >= 3:
        logp, cap = lmeta.loc[llms, "log2p"].values, lmeta.loc[llms, "cap"].values
        fam = lmeta.loc[llms, "family"].values; full = (np.arange(len(llms)), np.arange(len(vits)))
        for y in MET:
            Ye, Ym = mat(variant, "eol", y, llms, vits), mat(variant, "mean", y, llms, vits)
            s_e, s_m = slope_fe_arr(Ye, *full, logp, fam), slope_fe_arr(Ym, *full, logp, fam)
            c_e, c_m = cap_r_arr(Ye, *full, cap), cap_r_arr(Ym, *full, cap)
            bs_s = boot_pair(lambda Li, Vj: slope_fe_arr(Ye, Li, Vj, logp, fam) - slope_fe_arr(Ym, Li, Vj, logp, fam), len(llms), len(vits))
            bs_c = boot_pair(lambda Li, Vj: cap_r_arr(Ye, Li, Vj, cap) - cap_r_arr(Ym, Li, Vj, cap), len(llms), len(vits))
            rows.append(dict(test=f"R1 slope per doubling (family + ViT FE), {len(llms)} LLMs", variant=variant, metric=y, a="eol", b="mean",
                             value_a=s_e, value_b=s_m, delta=s_e - s_m, ci_low=ci(bs_s)[0], ci_high=ci(bs_s)[1],
                             primary=(variant == "huh" and y == "cka_cal")))
            rows.append(dict(test=f"R2 Pearson with capability (LLM means), {len(llms)} LLMs", variant=variant, metric=y, a="eol", b="mean",
                             value_a=c_e, value_b=c_m, delta=c_e - c_m, ci_low=ci(bs_c)[0], ci_high=ci(bs_c)[1],
                             primary=(variant == "huh" and y == "cka_cal")))
    # R3: controls (mean over all LLM x ViT pairs where both readouts exist)
    for a, b in (("eol", "neutral"), ("eol", "repeat"), ("eol_alt", "eol"), ("eol", "mean"), ("neutral", "mean"), ("repeat", "mean")):
        both = sorted(set(allr[(allr.readout == a) & (allr.variant == variant)].llm) & set(allr[(allr.readout == b) & (allr.variant == variant)].llm))
        if len(both) < 2: continue
        for y in ["cka_cal", "ucka_heldout", "mknn_max"]:
            Ya, Yb = mat(variant, a, y, both, vits), mat(variant, b, y, both, vits)
            dm = lambda Li, Vj: float(np.nanmean(Ya[np.ix_(Li, Vj)]) - np.nanmean(Yb[np.ix_(Li, Vj)]))
            d0 = dm(np.arange(len(both)), np.arange(len(vits))); bs = boot_pair(dm, len(both), len(vits))
            rows.append(dict(test=f"R3 {a} - {b} (mean over pairs), {len(both)} LLMs", variant=variant, metric=y, a=a, b=b,
                             value_a=float(np.nanmean(Ya)), value_b=float(np.nanmean(Yb)), delta=d0, ci_low=ci(bs)[0], ci_high=ci(bs)[1],
                             primary=(variant == "huh" and y == "cka_cal" and (a, b) == ("eol", "neutral"))))
tests = pd.DataFrame(rows); tests.to_csv(os.path.join(RES_DIR, "e2p_tests.csv"), index=False)
print(tests.round(4).to_string(index=False))

# R4: per-LLM means (huh), mean vs eol, for every family
per = allr[allr.variant == "huh"].groupby(["family", "llm", "llm_params", "readout"])[MET].mean().reset_index()
per.to_csv(os.path.join(RES_DIR, "e2p_per_llm.csv"), index=False)
fams = ["BLOOMZ", "OpenLLaMA", "LLaMA", "Llama-3", "Mistral", "OLMo", "Gemma"]
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
for ax, (rd, y, ttl) in zip(axs, [("mean", "cka_cal", "mean pooling: calibrated CKA"), ("eol", "cka_cal", "PromptEOL: calibrated CKA"),
                                  ("mean", "cka_tau", "mean pooling: threshold τ"), ("eol", "cka_tau", "PromptEOL: threshold τ")]):
    for f_, c_ in zip(fams, cols):
        gg = per[(per.family == f_) & (per.readout == rd)].sort_values("llm_params")
        if len(gg): ax.plot(gg.llm_params / 1e9, gg[y], "o-", color=c_, label=f_)
    ax.set_xscale("log"); ax.set_title(ttl); ax.set_xlabel("LLM parameters (B)"); ax.grid(alpha=.3)
axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_e2p_families.png"), dpi=150); plt.show()

# %%
# ============================== PACKAGE ==============================
import zipfile
zp = os.path.join(RUN, "e2p_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in [LOG, os.path.join(RUN, "checker.csv")] + glob.glob(os.path.join(RES_DIR, "*.csv")) + glob.glob(os.path.join(FIG_DIR, "*.png")):
        if os.path.exists(f): z.write(f, os.path.relpath(f, RUN))
log("packaged", zp, f"{os.path.getsize(zp) / 1e6:.1f} MB")
if IN_COLAB:
    from google.colab import files; files.download(zp)
