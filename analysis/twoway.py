import sys, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
res = pd.read_csv(sys.argv[1]); meta = pd.read_csv(sys.argv[2])
h = res[(res.variant == "huh") & (res.precision == "bf16")].copy()
print("LLMs:", h.llm.nunique(), "ViTs:", h.vision.nunique(), "rows:", len(h))
h["cap"] = -h.bpb; h["log2p"] = np.log2(h.llm_params.astype(float))
m = meta[meta.precision == "bf16"]
cap_hinge = float(np.median(-m.bpb)); p_hinge = float(np.median(np.log2(m.llm_params.astype(float))))
print(f"hinges: bpb {-cap_hinge:.3f}, log2p {p_hinge:.2f} ({2**p_hinge/1e9:.1f}B)")
MET = ["ucka_heldout", "mknn_max", "cka_max", "cka_cal"]

# per-LLM means (Huh preprocessing)
mm = h.groupby(["llm", "llm_family", "bpb", "llm_params"])[MET + ["cka_tau"]].mean().reset_index().sort_values(["llm_family", "llm_params"])
print(mm.round(4).to_string(index=False))

def design(df, x, hinge, fe):
    cols = [df[x].values, np.maximum(df[x].values - hinge, 0.0), pd.get_dummies(df.vision).values.astype(float)]
    if fe: cols.append(pd.get_dummies(df.llm_family).values.astype(float)[:, 1:])   # absent families have no column
    return np.column_stack(cols)
def fit(df, y, x, hinge, fe):
    X = design(df, x, hinge, fe)
    if np.linalg.matrix_rank(X) < X.shape[1] or (df[x] > hinge).sum() == 0 or (df[x] <= hinge).sum() == 0: return np.nan, np.nan
    b = np.linalg.lstsq(X, df[y].values, rcond=None)[0]; return b[0], b[0] + b[1]

llms, vits = h.llm.unique(), h.vision.unique()
idx = {(l, v): i for i, (l, v) in enumerate(zip(h.llm, h.vision))}
def boot_frame(rng, mode):
    L = rng.choice(llms, len(llms)) if mode in ("llm", "two") else llms
    V = rng.choice(vits, len(vits)) if mode in ("vit", "two") else vits
    rows = [idx[(l, v)] for l in L for v in V]
    bd = h.iloc[rows].copy()
    bd["vision"] = np.tile([f"{v}#{k}" for k, v in enumerate(V)], len(L))
    bd["llm"] = np.repeat([f"{l}#{k}" for k, l in enumerate(L)], len(V))
    return bd

B = 2000
out = []
for x, hinge in (("log2p", p_hinge), ("cap", cap_hinge)):
    for fe in (False, True):
        for y in MET:
            lo, hi = fit(h, y, x, hinge, fe); r = dict(axis=x, family_fe=fe, metric=y, below=lo, above=hi)
            for mode in ("vit", "two"):
                rng = np.random.default_rng(0); bl, bh = [], []
                for _ in range(B):
                    a, b = fit(boot_frame(rng, mode), y, x, hinge, fe); bl.append(a); bh.append(b)
                bl, bh = np.array(bl), np.array(bh); ok = ~np.isnan(bh)
                r[f"{mode}_below_ci"] = f"[{np.nanpercentile(bl,2.5):+.4f}, {np.nanpercentile(bl,97.5):+.4f}]"
                r[f"{mode}_above_ci"] = f"[{np.nanpercentile(bh,2.5):+.4f}, {np.nanpercentile(bh,97.5):+.4f}]"
                d = bh[ok] - bl[ok]
                r[f"{mode}_diff_ci"] = f"[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]"
                r[f"{mode}_valid"] = ok.mean()
            out.append(r)
out = pd.DataFrame(out)
out.to_csv(sys.argv[3] + "/saturation_twoway.csv", index=False)
for c in ("below", "above"): out[c] = out[c].round(4)
print(out[["axis","family_fe","metric","below","above","vit_below_ci","two_below_ci","vit_above_ci","two_above_ci","two_diff_ci","two_valid"]].to_string(index=False))

# trend correlation with capability (LLM means), two-way CI; and difference ucka_heldout - cka_cal
tr = []
for y in MET + ["mknn_cal", "cka_heldout_cal"]:
    r0 = np.corrcoef(h.groupby("llm")[y].mean(), h.groupby("llm").cap.first())[0, 1]
    rng = np.random.default_rng(1); bs = []
    for _ in range(B):
        bd = boot_frame(rng, "two"); g = bd.groupby("llm"); bs.append(np.corrcoef(g[y].mean(), g.cap.first())[0, 1])
    tr.append(dict(metric=y, pearson_cap=round(r0, 3), ci=f"[{np.nanpercentile(bs,2.5):+.3f}, {np.nanpercentile(bs,97.5):+.3f}]"))
print(pd.DataFrame(tr).to_string(index=False))
rng = np.random.default_rng(2); ds = []
for _ in range(B):
    g = boot_frame(rng, "two").groupby("llm")
    ds.append(np.corrcoef(g.ucka_heldout.mean(), g.cap.first())[0,1] - np.corrcoef(g.cka_cal.mean(), g.cap.first())[0,1])
print("r(ucka_heldout) - r(cka_cal) vs capability:", np.round(np.nanpercentile(ds, [2.5, 50, 97.5]), 3))
