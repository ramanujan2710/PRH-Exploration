"""Figures for the PRH results report. Reads saved results only (no model compute)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

S = "/tmp/claude-0/-home-claude/28df22f1-4718-53f5-8e99-902542254ccb/scratchpad"
OUT = os.path.join(S, "report", "figs"); os.makedirs(OUT, exist_ok=True)

# reference palette (light), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "legend.fontsize": 7.5, "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 200,
    "figure.facecolor": "white", "axes.facecolor": "white"})

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig)

# ---------------- E1: calibrated CKA vs width, truth fixed at 0.30 ----------------
d = pd.read_csv(f"{S}/e1/results/E1b_fixed.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, sp, title in zip(axs, ["isotropic", "powerlaw"], ["High effective rank (isotropic noise)", "Low effective rank (power-law spectrum)"]):
    g = d[d.spectrum == sp].groupby("dy")[["raw", "deb", "cal"]].mean()
    ax.axhline(0.30, color=MUTED, lw=1, ls="--"); ax.text(g.index.min(), 0.307, "true CKA = 0.30", color=INK2, fontsize=7)
    for col, lab, c in (("raw", "raw CKA", C[0]), ("cal", "calibrated CKA (Aristotelian)", C[1]), ("deb", "unbiased CKA", C[2])):
        ax.plot(g.index, g[col], "o-", color=c, label=lab)
    ax.set_xscale("log", base=2); ax.set_xlabel("feature width d"); ax.set_title(title)
axs[0].set_ylabel("score (n = 1,024)"); axs[0].set_ylim(0.1, 0.45); axs[1].legend(loc="lower left")
save(fig, "e1_width.png")

# ---------------- E2 data ----------------
r = pd.read_csv(f"{S}/res7/results/alignment_all.csv")
h = r[(r.variant == "huh") & (r.precision == "bf16")]
fam_order = ["BLOOMZ", "OpenLLaMA", "LLaMA", "Llama-3", "Mistral", "OLMo", "Gemma"]
fam_col = dict(zip(fam_order, C))

# E2a: threshold predicted vs observed
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.scatter(r.tau_pred, r.cka_tau, s=6, color=C[0], alpha=0.45, linewidths=0)
lim = [min(r.tau_pred.min(), r.cka_tau.min()) - 0.01, max(r.tau_pred.max(), r.cka_tau.max()) + 0.01]
ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
rr = np.corrcoef(r.tau_pred, r.cka_tau)[0, 1]
ax.set_xlabel("predicted threshold  √(r_max,x · r_max,y) / n"); ax.set_ylabel("observed calibration threshold τ")
ax.set_title(f"Threshold = chance level of the\nhighest-rank layer pair (r = {rr:.5f})")
ax.text(0.04, 0.9, f"{len(r)} model pairs × preprocessings", transform=ax.transAxes, fontsize=7, color=INK2)
save(fig, "e2_tau.png")

# E2b: per-family lines vs params, 4 small multiples
mm = h.groupby(["llm", "llm_family", "llm_params"])[["cka_tau", "cka_cal", "ucka_heldout", "mknn_max", "cka_max"]].mean().reset_index()
panels = [("cka_max", "Raw CKA (max over layers)"), ("cka_tau", "Calibration threshold τ"),
          ("cka_cal", "Calibrated CKA (Aristotelian)"), ("ucka_heldout", "Unbiased CKA, held-out layer")]
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.7))
for ax, (m, t) in zip(axs, panels):
    for fam in fam_order:
        g = mm[mm.llm_family == fam].sort_values("llm_params")
        ax.plot(g.llm_params / 1e9, g[m], "o-", color=fam_col[fam], label=fam, ms=4, lw=1.6)
    ax.set_xscale("log"); ax.set_title(t); ax.set_xlabel("LLM parameters (B)")
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.06))
save(fig, "e2_families.png")

# E2c: vision families
vf = h.groupby("vision_family")[["ucka_heldout", "mknn_max", "cka_cal"]].mean().reindex(["mae", "in21k", "dinov2", "clip_ft", "clip"])
lab = {"mae": "MAE", "in21k": "ImageNet-21k", "dinov2": "DINOv2", "clip_ft": "CLIP (fine-tuned)", "clip": "CLIP"}
fig, ax = plt.subplots(figsize=(4.2, 2.4))
y = np.arange(len(vf)); w = 0.38
ax.barh(y + w / 2, vf.ucka_heldout, height=w, color=C[0], label="unbiased CKA, held-out")
ax.barh(y - w / 2, vf.mknn_max, height=w, color=C[1], label="mutual k-NN")
ax.set_yticks(y, [lab[i] for i in vf.index]); ax.set_xlabel("mean alignment with the 20 LLMs"); ax.set_xlim(0, 0.44)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
ax.grid(axis="y", visible=False)
for i, (a, b) in enumerate(zip(vf.ucka_heldout, vf.mknn_max)):
    ax.text(a + 0.004, i + w / 2, f"{a:.2f}", va="center", fontsize=7, color=INK2)
    ax.text(b + 0.004, i - w / 2, f"{b:.2f}", va="center", fontsize=7, color=INK2)
save(fig, "e2_vision.png")

# ---------------- E3 ----------------
t = pd.read_csv(f"{S}/e3final/e3_interim_table.csv"); y3 = t.foscttm.values
groups = [np.flatnonzero(t.pair.values == p) for p in t.pair.unique()]
rng = np.random.default_rng(0)
draws = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(2000)]
preds = [("ros_cka", "Rosetta CKA (unbiased, n=10k)"), ("ucka_fp64_rosettaprep", "Unbiased CKA, Rosetta prep (n=1,024)"),
         ("cka_raw", "Raw CKA, Aristotelian prep"), ("cka_cal", "Calibrated CKA (Aristotelian)"),
         ("ucka_fp64_aristoprep", "Unbiased CKA, Aristotelian prep"), ("mknn_raw", "Mutual k-NN (raw = calibrated)"),
         ("ros_mutual_knn", "Rosetta mutual k-NN"), ("ros_tsi", "Rosetta TSI"), ("ros_qsi", "Rosetta QSI")]
rows = []
for c, l in preds:
    x = t[c].values; r0 = abs(stats.spearmanr(x, y3).statistic)
    bs = [abs(stats.spearmanr(x[dd], y3[dd]).statistic) for dd in draws]
    rows.append((l, r0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
rows = rows[::-1]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for i, (l, r0, lo, hi) in enumerate(rows):
    col = C[0] if ("Rosetta prep" in l or "n=10k" in l) else (C[1] if "Aristotelian" in l else MUTED)
    ax.plot([lo, hi], [i, i], color=col, lw=1.6); ax.plot(r0, i, "o", color=col, ms=6)
    ax.text(hi + 0.01, i, f"{r0:.2f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(rows)), [x[0] for x in rows]); ax.set_xlim(0.3, 1.0)
ax.set_xlabel("|Spearman ρ| with unpaired alignment error (FOSCTTM), 95% CI over 21 model pairs")
ax.grid(axis="y", visible=False)
save(fig, "e3_predictors.png")

t["lm"] = t.language.str.replace("_69445574b3021b8a2733e87d8603d769", "")
lm_lab = {"mpnet": "mpnet (encoder)", "qwen3-embedding-8b": "Qwen3-Embedding-8B", "qwen3-8b-gen": "Qwen3-8B generative (decoder)"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, (c, ttl) in zip(axs, [("ucka_fp64_rosettaprep", "Rosetta preprocessing"),
                              ("ucka_fp64_aristoprep", "Aristotelian clipping")]):
    for k, (lm, l) in enumerate(lm_lab.items()):
        g = t[t.lm == lm]
        ax.scatter(g[c], g.foscttm, s=18, color=C[k], label=l, edgecolors="white", linewidths=0.6)
    rho = stats.spearmanr(t[c], t.foscttm).statistic
    ax.set_title(f"{ttl}: Spearman ρ = {rho:.2f}"); ax.set_xlabel("unbiased CKA at n = 1,024")
axs[0].set_ylabel("FOSCTTM (lower = better alignment)")
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
save(fig, "e3_decoder.png")

# ---------------- E6 ----------------
e = pd.read_csv(f"{S}/e6v2/results/e6_all.csv")
def series(scope, variant, met, kind, rd, vfam="DINOv2", vr="cls"):
    g = e[(e.scope == scope) & (e.variant == variant) & (e.vfam == vfam) & (e.vision_readout == vr) & (e.lang_kind == kind) & (e.lang_readout == rd)]
    return g.groupby("lang").agg(p=("lang_params", "first"), y=(met, "mean")).sort_values("p")
lines = [("decoder", "prompt", "Qwen3 decoder, PromptEOL readout", C[0], "o-"),
         ("embedding", "last", "Qwen3-Embedding", C[1], "s-"),
         ("decoder", "mean", "Qwen3 decoder, mean pooling (E2 / Huh)", C[2], "^-"),
         ("decoder", "last", "Qwen3 decoder, last token", C[3], "v-")]
fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.9))
for ax, (scope, variant, met, ttl) in zip(axs, [("layers", "huh", "cka_cal", "Calibrated CKA\n(Aristotelian pipeline: clip + layer search)"),
                                                 ("final", "huh", "cka_cal", "Calibrated CKA, final layer\n(the model's own output)"),
                                                 ("final", "noclip", "ucka_max", "Unbiased CKA, final layer,\nno clipping (raw geometry)")]):
    for kind, rd, l, c, mk in lines:
        s = series(scope, variant, met, kind, rd); ax.plot(s.p / 1e9, s.y, mk, color=c, label=l)
    mp = series(scope, variant, met, "encoder", "mean")
    ax.plot(mp.p / 1e9, mp.y, "D", color=C[6], label="mpnet (encoder)", ms=6)
    ax.set_xscale("log"); ax.set_xlabel("language model parameters (B)"); ax.set_title(ttl)
axs[0].set_ylabel("alignment with DINOv2 (mean of 3 sizes)")
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), fontsize=7)
save(fig, "e6_readouts.png")
print("figures:", sorted(os.listdir(OUT)))
