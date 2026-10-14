import json, glob, os, sys, numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
D = sys.argv[1]
al = pd.DataFrame([dict(**(j := json.load(open(f)))["task"], foscttm=j["metrics"]["foscttm"], secs=j["seconds"]) for f in glob.glob(f"{D}/align/*.json")])
ge = pd.DataFrame([dict(**(j := json.load(open(f)))["task"], **{f"ros_{k}": v for k, v in j["metrics"].items()}) for f in glob.glob(f"{D}/geometry/*.json")])
me = []
for f in glob.glob(f"{D}/metrics_gpu/*.json"):
    j = json.load(open(f)); s = pd.DataFrame(j["subsets"])
    me.append(dict(dataset=j["dataset"], vision=j["vision"], language=j["language"], **s.drop(columns="subset").mean().to_dict(),
                   cka_cal_sd=s.cka_cal.std(), cka_p_max=s.cka_p.max(), mknn_p_max=s.mknn_p.max()))
me = pd.DataFrame(me)
k = ["dataset", "vision", "language"]
print("seeds per combo:", al.groupby(k).size().value_counts().to_dict(), "| align by dataset:", al.dataset.value_counts().to_dict())
print("secs per alignment by dataset (min):", (al.groupby("dataset").secs.median() / 60).round(1).to_dict())
t = al.groupby(k).foscttm.mean().reset_index().merge(ge.drop(columns="kind"), on=k).merge(me, on=k)
t["pair"] = t.vision + "|" + t.language
print("combos in interim table:", len(t), "| model pairs:", t.pair.nunique())
print("sanity: cal<=raw", bool((t.cka_cal <= t.cka_raw + 1e-9).all()), "| fp32 vs fp64 ucka max diff", float((t.ucka_aristo_fp32 - t.ucka_fp64_aristoprep).abs().max()),
      "| max p (cka, mknn):", t.cka_p_max.max(), t.mknn_p_max.max(), "| FOSCTTM range", t.foscttm.min().round(3), t.foscttm.max().round(3))

P = {"ros_cka": "Rosetta CKA (unbiased, n=10k)", "ros_mutual_knn": "Rosetta mKNN (n=10k)", "ros_tsi": "Rosetta TSI", "ros_qsi": "Rosetta QSI",
     "cka_raw": "Aristo raw CKA (n=1024)", "cka_cal": "Aristo calibrated CKA", "ucka_aristo_fp32": "Aristo unbiased CKA fp32",
     "ucka_fp64_aristoprep": "unbiased CKA fp64 (their prep)", "ucka_fp64_rosettaprep": "unbiased CKA fp64 (Rosetta prep, n=1024)",
     "mknn_raw": "Aristo raw mKNN", "mknn_cal": "Aristo calibrated mKNN"}
groups = [np.flatnonzero(t.pair.values == p) for p in t.pair.unique()]
B = 4000; rng = np.random.default_rng(0)
draws = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]
y = t.foscttm.values
def rho(x, idx): return stats.spearmanr(x[idx], y[idx]).statistic
rows = []
for c, name in P.items():
    x = t[c].values; sl, ic = np.polyfit(x, y, 1); r2 = 1 - (y - (sl * x + ic)).var() / y.var()
    bs = np.array([rho(x, d) for d in draws])
    rows.append(dict(predictor=name, spearman=round(stats.spearmanr(x, y).statistic, 3), ci=f"[{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]",
                     pearson=round(stats.pearsonr(x, y).statistic, 3), R2=round(r2, 3)))
print(pd.DataFrame(rows).to_string(index=False))
def diff(a, b):
    xa, xb = t[a].values, t[b].values
    d0 = abs(rho(xa, np.arange(len(y)))) - abs(rho(xb, np.arange(len(y))))
    bs = np.array([abs(rho(xa, d)) - abs(rho(xb, d)) for d in draws])
    return f"{d0:+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]"
print("H1  |rho| unbiased fp64 (their prep) - calibrated CKA:", diff("ucka_fp64_aristoprep", "cka_cal"))
print("H1b |rho| raw CKA - calibrated CKA:            ", diff("cka_raw", "cka_cal"))
print("H2  |rho| calibrated mKNN - raw mKNN:          ", diff("mknn_cal", "mknn_raw"))
print("S1  |rho| Rosetta CKA n=10k - unbiased n=1024 (Rosetta prep):", diff("ros_cka", "ucka_fp64_rosettaprep"))
print("within-dataset Spearman vs FOSCTTM:")
print(t.groupby("dataset").apply(lambda g: pd.Series({c: round(stats.spearmanr(g[c], g.foscttm).statistic, 3) for c in ["ros_cka", "cka_raw", "cka_cal", "ucka_fp64_aristoprep", "mknn_raw", "mknn_cal"]} | {"n": len(g)}), include_groups=False).to_string())
t.to_csv(f"{D}/e3_interim_table.csv", index=False)
