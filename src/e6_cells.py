# %% [md]
# # E6: Is global cross-modal structure there when both sides have a *global* representation?
#
# **Question (Mayug's hypothesis).** The Aristotelian paper finds that global alignment (CKA) disappears after calibration while local alignment (mutual k-NN) survives. They, like Huh et al., measured **mean-pooled hidden states of decoder LLMs**. Hypothesis: global structure is shared once the language side has a global sentence representation, e.g. an encoder, or a decoder turned into one.
#
# **Design (2 × 2 + pooling).** WIT-1024 (Huh et al.), the same pipeline and engine as the E2 audit (v3 engine, all layer pairs, permutation null on the max).
# - **Language:** Qwen3 decoders (0.6B / 4B / 8B **Base**) vs the Qwen3-Embedding models of the same sizes (decoder architecture turned into an embedding model), plus mpnet (a classic encoder).
#   - Readouts: `mean` (E2 / Huh protocol), `last` token, and for decoders `prompt` (PromptEOL, a training-free decoder→encoder readout). For mpnet: `mean` (its native pooling) and `cls`.
# - **Vision:** MAE (masked reconstruction, "predictive") vs DINOv2 (self-distillation with a global objective), 3 sizes each. Readouts: `cls` (E2 / Huh) and `mean` over patch tokens.
# - **Measurements** per (language readout × vision readout), for 2 preprocessings (`huh` = 95th-percentile clip + L2, the Aristotelian setting; `noclip` = L2 only):
#   - (a) the Aristotelian layer search: raw max CKA, calibrated CKA, held-out unbiased CKA, mutual k-NN;
#   - (b) the final layer only (the model's own output): calibrated CKA of that fixed pair, unbiased CKA, mutual k-NN.
#
# **Pre-registered predictions** (written before any E6 result):
# - **P1 (Mayug):** at each size, Qwen3-Embedding (`last`, its native readout) > Qwen3-Base (`mean`, the E2 readout) on calibrated CKA and held-out unbiased CKA, against DINOv2 `cls`. Positive in ≥ 8 of 9 (size × DINOv2) cells.
# - **P2 (convergence where it should be):** the slope over log2(parameters) of held-out unbiased CKA is larger for the embedding family than for the decoder family.
# - **P3 (global organisation vs "like with like"):** the encoder − decoder gap is ≥ 0 with MAE as well. "Like with like" predicts it to be < 0 with MAE.
# - **P4 (pooling):** for decoders, `prompt` or `last` ≥ `mean` on the final-layer measurements. Exploratory; no direction pre-registered for MAE `cls` vs `mean`.
#
# **How to run.** `SMOKE_TEST = True` first (~10 min: Qwen3-0.6B-Base, Qwen3-Embedding-0.6B, mpnet × MAE-B, DINOv2-B, 50 permutations). It ends with a **CHECKER** table and `e6_smoke_results.zip`. Then set `SMOKE_TEST = False` and *Run all* (~1.5–2 h on a G4/H100). It ends with `e6_results.zip`. Everything is saved to Drive and resumes after a disconnect.

# %%
# ============================== CONFIG ==============================
import os
SMOKE_TEST = False      # smoke and first full run done (12 Oct); this version only re-aligns (features are on Drive)
N_PERMS = 200                # Aristotelian paper's PRH runs: 200
ALPHA = 0.05
TOPK = 10
VARIANTS_E6 = ("huh", "noclip")
B_BOOT = 2000
SAVE_TO_DRIVE = True
SEED = 0

LANG_FULL = [   # (model, kind, readouts)
    ("Qwen/Qwen3-0.6B-Base", "decoder", ("mean", "last", "prompt")),
    ("Qwen/Qwen3-4B-Base", "decoder", ("mean", "last", "prompt")),
    ("Qwen/Qwen3-8B-Base", "decoder", ("mean", "last", "prompt")),
    ("Qwen/Qwen3-Embedding-0.6B", "embedding", ("last", "mean")),
    ("Qwen/Qwen3-Embedding-4B", "embedding", ("last", "mean")),
    ("Qwen/Qwen3-Embedding-8B", "embedding", ("last", "mean")),
    ("sentence-transformers/all-mpnet-base-v2", "encoder", ("mean", "cls")),
]
VIS_FULL = ["vit_base_patch16_224.mae", "vit_large_patch16_224.mae", "vit_huge_patch14_224.mae",
            "vit_base_patch14_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m"]
NATIVE = {"decoder": "mean", "embedding": "last", "encoder": "mean"}   # E2 readout for decoders; own pooling for the others
if SMOKE_TEST:
    LANG = [LANG_FULL[0], LANG_FULL[3], LANG_FULL[6]]
    VIS = [VIS_FULL[0], VIS_FULL[3]]
    N_PERMS, B_BOOT = 50, 200
else:
    LANG, VIS = LANG_FULL, VIS_FULL

# %%
# ============================== SETUP ==============================
import sys, subprocess, json, time, glob, shutil, math
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "timm", "accelerate", "psutil", "huggingface_hub",
                    "datasets", "sentencepiece", "protobuf", "sentence-transformers"], check=False)
    if SAVE_TO_DRIVE:
        from google.colab import drive; drive.mount("/content/drive")
    try:
        from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        pass
ROOT = os.environ.get("PRH_ROOT") or ("/content/drive/MyDrive/prh_audit" if (IN_COLAB and SAVE_TO_DRIVE) else "/content/prh_audit")
RUN = os.path.join(ROOT, "e6_smoke" if SMOKE_TEST else "e6")
FEAT_DIR, RES_DIR, FIG_DIR = (os.path.join(RUN, d) for d in ("features", "results", "figs"))
for d in (FEAT_DIR, RES_DIR, FIG_DIR): os.makedirs(d, exist_ok=True)
import torch, numpy as np, pandas as pd
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = os.path.join(RUN, "log.txt")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s); open(LOG, "a").write(s + "\n")
import transformers, timm
log("device:", torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "| torch", torch.__version__,
    "| transformers", transformers.__version__, "| timm", timm.__version__, "| run dir:", RUN)

# %%
# ============================== ENGINE (identical to the E2 audit, v3) ==============================
__ENGINE__

# %%
# ============================== E6 EXTRACTION CODE ==============================
__E6_EXTRACT__

# %%
# ============================== DATA: WIT-1024 (Huh et al.), identical to E2 ==============================
from datasets import load_dataset
ds = load_dataset("minhuh/prh", revision="wit_1024", split="train")
texts = [str(x["text"][0]) for x in ds]; images = [x["image"] for x in ds]
N = len(texts); assert N == 1024, N
log(f"dataset: {N} pairs; first caption: {texts[0][:80]!r}")
def fpath(model, readout):
    return os.path.join(FEAT_DIR, f"{model.replace('/', '_')}__{readout}.pt")

# %%
# ============================== EXTRACT ==============================
for name in VIS:
    if all(os.path.exists(fpath(name, r)) for r in ("cls", "mean")):
        log("vision exists:", name); continue
    t0 = time.time()
    try:
        out = extract_vision2(name, images)
        for r, d in out.items(): torch.save(d, fpath(name, r))
        log(f"vision done: {name} {tuple(out['cls']['feats'].shape)} ({time.time()-t0:.0f}s)")
    except Exception as e:
        log("VISION FAILED:", name, repr(e)[:300])
    finally:
        hf_cache_clear("timm/" + name)
for name, kind, rs in LANG:
    if all(os.path.exists(fpath(name, r)) for r in rs):
        log("language exists:", name); continue
    t0 = time.time()
    try:
        out = extract_text(name, texts, kind, rs)
        for r, d in out.items(): torch.save(d, fpath(name, r))
        log(f"language done: {name} [{kind}] readouts={list(out)} {tuple(next(iter(out.values()))['feats'].shape)} "
            f"batch={next(iter(out.values()))['batch_size']} ({time.time()-t0:.0f}s)")
    except Exception as e:
        log("LANGUAGE FAILED:", name, repr(e)[:300])
    finally:
        hf_cache_clear(name)

# %%
# ============================== ALIGN: every language readout x every vision readout ==============================
g = torch.Generator().manual_seed(SEED)
PERMS = [torch.randperm(N, generator=g).to(DEV) for _ in range(N_PERMS)]
_split = torch.randperm(N, generator=g)
HALF = (_split[: N // 2].to(DEV), _split[N // 2:].to(DEV))
PERMS_HALF = [torch.randperm(N - N // 2, generator=g).to(DEV) for _ in range(N_PERMS)]
L_SETS = [(m, k, r) for m, k, rs in LANG for r in rs if os.path.exists(fpath(m, r))]
V_SETS = [(m, r) for m in VIS for r in ("cls", "mean") if os.path.exists(fpath(m, r))]
log(f"aligning {len(L_SETS)} language feature sets x {len(V_SETS)} vision feature sets x {len(VARIANTS_E6)} preprocessings")

DROPPED = {}
def stats(path, variant, final_only):
    f = torch.load(path)["feats"].to(DEV).float()
    if final_only: f = f[:, -1:, :]
    layers = preprocess(f, variant)
    # A layer whose readout is the same vector for every caption (e.g. layer 0 of a last-token / prompt / [CLS] readout:
    # the input embedding of one fixed token) has zero variance; its CKA is 0/0. Drop such layers from the layer search.
    keep = [l for l, X in enumerate(layers) if float((X - X.mean(0)).pow(2).sum()) > 1e-6 * X.shape[0]]
    if len(keep) < len(layers): DROPPED[(os.path.basename(path), variant)] = [l for l in range(len(layers)) if l not in keep]
    s = SideStats([layers[l] for l in keep], k=TOPK, half_idx=HALF); s.kept_layers = keep; del f, layers; return s

for variant in VARIANTS_E6:
    for final_only in (False, True):
        tag = f"{variant}_{'final' if final_only else 'layers'}"
        out_csv = os.path.join(RES_DIR, f"align2_{tag}.csv")   # v2: constant layers dropped
        done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["lang", "lang_readout"])
        todo = [x for x in L_SETS if not ((done.lang == x[0]) & (done.lang_readout == x[2])).any()]
        if not todo:
            log("done:", tag); continue
        vs = {(m, r): stats(fpath(m, r), variant, final_only) for m, r in V_SETS}
        for m, k, r in todo:
            t1 = time.time(); xs = stats(fpath(m, r), variant, final_only); rows = []
            for (vm, vr), ys in vs.items():
                o, _ = align_pair(xs, ys, PERMS, alpha=ALPHA, perms_half=PERMS_HALF)
                rows.append(dict(variant=variant, scope="final" if final_only else "layers", lang=m, lang_kind=k, lang_readout=r,
                                 vision=vm, vision_readout=vr, **{a: (str(b) if isinstance(b, tuple) else b) for a, b in o.items()}))
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(out_csv, index=False)
            log(f"[{tag}] {m} / {r}: {len(rows)} pairs ({time.time()-t1:.0f}s)")
            del xs; torch.cuda.empty_cache()
        del vs; torch.cuda.empty_cache()
log("constant layers dropped from the layer search:", {f"{k[0]}|{k[1]}": v for k, v in DROPPED.items()})

# %%
# ============================== CHECKER (read the PASS/FAIL table) ==============================
checks = []
def check(name, ok, detail=""):
    checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=str(detail)[:200]))
wl = open(LOG).read()
check("no extraction failures", "FAILED" not in wl, [l for l in wl.splitlines() if "FAILED" in l][:3])
exp_sets = [(m, r) for m, k, rs in LANG for r in rs] + [(m, r) for m in VIS for r in ("cls", "mean")]
missing = [f"{m}/{r}" for m, r in exp_sets if not os.path.exists(fpath(m, r))]
check("all feature files present", not missing, missing[:5])
bad = []
for m, r in exp_sets:
    if os.path.exists(fpath(m, r)):
        d = torch.load(fpath(m, r), mmap=True); f = d["feats"]
        if f.shape[0] != N or not torch.isfinite(f.float()).all(): bad.append(f"{m}/{r}")
check("features: N = 1024 rows and all finite", not bad, bad[:5])
fo = [torch.load(fpath(m, r), mmap=True).get("final_is_model_output", True) for m, k, rs in LANG for r in rs if os.path.exists(fpath(m, r))]
check("last layer read out = the model's own output (last_hidden_state)", all(fo), fo)
# readouts must differ from each other (catches a pooling bug that returns the same tensor twice)
for m, k, rs in LANG:
    if len(rs) > 1 and all(os.path.exists(fpath(m, r)) for r in rs):
        a, b = (torch.load(fpath(m, r))["feats"][:, -1].float() for r in rs[:2])
        check(f"readouts differ ({m.split('/')[-1]}: {rs[0]} vs {rs[1]})", not torch.allclose(a, b), float((a - b).abs().max()))
# mpnet mean readout reproduces sentence-transformers' own embedding
try:
    from sentence_transformers import SentenceTransformer
    st_name = "sentence-transformers/all-mpnet-base-v2"
    if os.path.exists(fpath(st_name, "mean")):
        st = SentenceTransformer(st_name, device=DEV)
        e_st = torch.tensor(st.encode(texts[:64], normalize_embeddings=True))
        ours = torch.nn.functional.normalize(torch.load(fpath(st_name, "mean"))["feats"][:64, -1].float(), dim=-1)
        cos = float((e_st * ours).sum(-1).min())
        check("mpnet mean readout = sentence-transformers embedding (min cosine > 0.999)", cos > 0.999, round(cos, 6))
        del st; hf_cache_clear(st_name)
except Exception as e:
    check("mpnet vs sentence-transformers (could not run)", False, repr(e)[:150])
res_all = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES_DIR, "align2_*.csv"))], ignore_index=True)
check("alignment rows complete", len(res_all) == len(L_SETS) * len(V_SETS) * len(VARIANTS_E6) * 2,
      f"{len(res_all)} vs {len(L_SETS) * len(V_SETS) * len(VARIANTS_E6) * 2}")
check("calibrated CKA in [0, 1] and <= raw", bool(((res_all.cka_cal >= 0) & (res_all.cka_cal <= res_all.cka_max + 1e-9)).all()))
lay = res_all[res_all.scope == "layers"]
tau_pred = np.sqrt(lay.nullmax_reff_x * lay.nullmax_reff_y) / N
check("threshold = chance level of the highest-rank layer pair (r > 0.99, as in E2)", float(np.corrcoef(tau_pred, lay.cka_tau)[0, 1]) > 0.99,
      round(float(np.corrcoef(tau_pred, lay.cka_tau)[0, 1]), 5))
# null: a shuffled pairing must give calibrated CKA = 0
m0, k0, r0 = L_SETS[0]; v0, vr0 = V_SETS[0]
xs = stats(fpath(m0, r0), "huh", True); zero = 0
for s_ in range(10):
    p = torch.randperm(N, generator=torch.Generator().manual_seed(100 + s_)).to(DEV)
    f = torch.load(fpath(v0, vr0))["feats"].to(DEV).float()[:, -1:, :][p]
    ys = SideStats(preprocess(f, "huh"), k=TOPK, half_idx=HALF)
    o, _ = align_pair(xs, ys, PERMS, alpha=ALPHA); zero += int(o["cka_cal"] == 0)
check("null: shuffled pairs give calibrated CKA = 0 (>= 9/10)", zero >= 9, f"{zero}/10")
checks = pd.DataFrame(checks); checks.to_csv(os.path.join(RUN, "checker.csv"), index=False)
print(checks.to_string(index=False)); log("checker:", (checks.result == "PASS").sum(), "PASS /", len(checks))
if SMOKE_TEST:
    import zipfile
    zp = os.path.join(RUN, "e6_smoke_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [LOG, os.path.join(RUN, "checker.csv")] + glob.glob(os.path.join(RES_DIR, "*.csv")):
            z.write(f, os.path.relpath(f, RUN))
    if IN_COLAB:
        from google.colab import files; files.download(zp)
    raise SystemExit("Smoke test finished: read the CHECKER table, send e6_smoke_results.zip, then set SMOKE_TEST = False and Run all.")

# %%
# ============================== PRE-REGISTERED ANALYSIS ==============================
import matplotlib; import matplotlib.pyplot as plt
res = res_all.copy()
psize = {m: torch.load(fpath(m, rs[0]), mmap=True)["num_params"] for m, k, rs in LANG if os.path.exists(fpath(m, rs[0]))}
psize.update({m: torch.load(fpath(m, "cls"), mmap=True)["num_params"] for m in VIS if os.path.exists(fpath(m, "cls"))})
res["lang_params"] = res.lang.map(psize); res["vision_params"] = res.vision.map(psize)
res["size"] = res.lang.str.extract(r"(0\.6B|4B|8B)")[0]
res["vfam"] = np.where(res.vision.str.contains("mae"), "MAE", "DINOv2")
res.to_csv(os.path.join(RES_DIR, "e6_all.csv"), index=False)
PRIMARY = ["cka_cal", "ucka_heldout"]                  # pre-registered (layer search, huh)
ALLM = ["cka_max", "cka_cal", "ucka_heldout", "ucka_max", "mknn_max", "mknn_cal"]
rng = np.random.default_rng(SEED)
def boot_ci(df, value, cluster, B=B_BOOT):
    """Mean of `value` and a 95% CI resampling clusters (vision models)."""
    cl = df[cluster].unique(); est = df[value].mean(); bs = []
    for _ in range(B):
        pick = rng.choice(cl, len(cl)); bs.append(pd.concat([df[df[cluster] == c] for c in pick])[value].mean())
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def sel(scope="layers", variant="huh", kind=None, readout=None, vfam=None, vreadout=None):
    d = res[(res.scope == scope) & (res.variant == variant)]
    if kind: d = d[d.lang_kind == kind]
    if readout: d = d[d.lang_readout == readout]
    if vfam: d = d[d.vfam == vfam]
    if vreadout: d = d[d.vision_readout == vreadout]
    return d

rows = []
# P1: embedding(last) - decoder(mean), same size, DINOv2 cls
for scope in ("layers", "final"):
    for variant in VARIANTS_E6:
        for met in ALLM if scope == "layers" else ["cka_max", "cka_cal", "ucka_max", "mknn_max"]:
            e = sel(scope, variant, "embedding", "last", "DINOv2", "cls")[["size", "vision", met]]
            dcd = sel(scope, variant, "decoder", "mean", "DINOv2", "cls")[["size", "vision", met]]
            j = e.merge(dcd, on=["size", "vision"], suffixes=("_enc", "_dec")); j["delta"] = j[f"{met}_enc"] - j[f"{met}_dec"]
            if len(j):
                est, lo, hi = boot_ci(j, "delta", "vision")
                rows.append(dict(test="P1 Qwen3-Embedding(last) - Qwen3-Base(mean), DINOv2 cls", scope=scope, variant=variant, metric=met,
                                 delta=est, ci_low=lo, ci_high=hi, positive=f"{int((j.delta > 0).sum())}/{len(j)}",
                                 primary=(scope == "layers" and variant == "huh" and met in PRIMARY)))
# P3: encoder - decoder gap by vision family (2x2), each vision readout
for variant in VARIANTS_E6:
    for vr in ("cls", "mean"):
        for vf in ("DINOv2", "MAE"):
            for met in PRIMARY + ["mknn_max"]:
                e = sel("layers", variant, "embedding", "last", vf, vr)[["size", "vision", met]]
                dcd = sel("layers", variant, "decoder", "mean", vf, vr)[["size", "vision", met]]
                j = e.merge(dcd, on=["size", "vision"], suffixes=("_enc", "_dec")); j["delta"] = j[f"{met}_enc"] - j[f"{met}_dec"]
                if len(j):
                    est, lo, hi = boot_ci(j, "delta", "vision")
                    rows.append(dict(test=f"P3 encoder - decoder gap | {vf} {vr}", scope="layers", variant=variant, metric=met,
                                     delta=est, ci_low=lo, ci_high=hi, positive=f"{int((j.delta > 0).sum())}/{len(j)}",
                                     primary=(variant == "huh" and vr == "cls" and met in PRIMARY)))
# P4: decoder readouts vs mean (final layer and layer search), DINOv2 cls
for scope in ("final", "layers"):
    for rd in ("last", "prompt"):
        for met in ["cka_cal", "ucka_max", "mknn_max"]:
            a = sel(scope, "huh", "decoder", rd, "DINOv2", "cls")[["size", "vision", met]]
            b = sel(scope, "huh", "decoder", "mean", "DINOv2", "cls")[["size", "vision", met]]
            j = a.merge(b, on=["size", "vision"], suffixes=("_a", "_b")); j["delta"] = j[f"{met}_a"] - j[f"{met}_b"]
            if len(j):
                est, lo, hi = boot_ci(j, "delta", "vision")
                rows.append(dict(test=f"P4 decoder {rd} - mean", scope=scope, variant="huh", metric=met, delta=est, ci_low=lo,
                                 ci_high=hi, positive=f"{int((j.delta > 0).sum())}/{len(j)}", primary=False))
tests = pd.DataFrame(rows); tests.to_csv(os.path.join(RES_DIR, "e6_tests.csv"), index=False)
print(tests.round(4).to_string(index=False))

# P2: scale slopes (held-out unbiased CKA and calibrated CKA vs log2 params), DINOv2 cls, per family
slopes = []
for kind, rd in (("embedding", "last"), ("decoder", "mean"), ("decoder", "prompt")):
    for met in PRIMARY + ["mknn_max"]:
        d = sel("layers", "huh", kind, rd, "DINOv2", "cls").copy(); d["lp"] = np.log2(d.lang_params.astype(float))
        if d.lang.nunique() < 2: continue
        s0 = np.polyfit(d.lp, d[met], 1)[0]; vits = d.vision.unique(); bs = []
        for _ in range(B_BOOT):
            bd = pd.concat([d[d.vision == v] for v in rng.choice(vits, len(vits))]); bs.append(np.polyfit(bd.lp, bd[met], 1)[0])
        slopes.append(dict(family=f"{kind}/{rd}", metric=met, slope_per_doubling=s0,
                           ci_low=float(np.percentile(bs, 2.5)), ci_high=float(np.percentile(bs, 97.5))))
slopes = pd.DataFrame(slopes); slopes.to_csv(os.path.join(RES_DIR, "e6_scale_slopes.csv"), index=False)
print(slopes.round(4).to_string(index=False))

# Summary table: mean over vision models, per language set x vision family/readout (layers, huh)
summ = (sel("layers", "huh").groupby(["lang", "lang_kind", "lang_readout", "vfam", "vision_readout"])[ALLM].mean().reset_index())
summ.to_csv(os.path.join(RES_DIR, "e6_summary_table.csv"), index=False)

# Figure: the 2x2 (held-out unbiased CKA), DINOv2 vs MAE x decoder vs embedding, by size
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, vf in zip(axs, ("DINOv2", "MAE")):
    for kind, rd, mk in (("embedding", "last", "o-"), ("decoder", "mean", "s--"), ("decoder", "prompt", "^:")):
        d = sel("layers", "huh", kind, rd, vf, "cls").groupby("lang").agg(p=("lang_params", "first"), y=("ucka_heldout", "mean")).sort_values("p")
        if len(d): ax.plot(d.p, d.y, mk, label=f"Qwen3 {kind} ({rd})")
    ax.set_xscale("log"); ax.set_title(f"vision: {vf} (cls)"); ax.set_xlabel("language model parameters"); ax.grid(alpha=.3)
axs[0].set_ylabel("held-out unbiased CKA (layer search, huh)"); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_e6_2x2.png"), dpi=140); plt.show()

# %%
# ============================== PACKAGE FOR CLAUDE ==============================
import zipfile
zp = os.path.join(RUN, "e6_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in [LOG, os.path.join(RUN, "checker.csv")] + glob.glob(os.path.join(RES_DIR, "*.csv")) + glob.glob(os.path.join(FIG_DIR, "*.png")):
        if os.path.exists(f): z.write(f, os.path.relpath(f, RUN))
log("packaged", zp, f"{os.path.getsize(zp) / 1e6:.1f} MB")
if IN_COLAB:
    from google.colab import files; files.download(zp)
