import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = 'results/'; F = 'figs/'
import os; os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 150})
C = dict(truth='#222222', cal='#C8553D', deb='#2E5C9A', raw='#8A8A8A', mknn='#3B8E5A')
def ci(g, col):
    m = g[col].mean(); s = g[col].std(ddof=1) / np.sqrt(len(g)); return m, 1.96 * s
def line(ax, df, x, col, label, color, ls='-'):
    agg = df.groupby(x).apply(lambda g: pd.Series(ci(g, col), index=['m', 'e']), include_groups=False)
    ax.errorbar(agg.index, agg.m, yerr=agg.e, label=label, color=color, ls=ls, marker='o', ms=3.5, capsize=2, lw=1.6)
summary = {}

# ---- Fig 1: width penalty with fixed truth
b = pd.read_csv(R + 'E1b_fixed.csv')
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, sp, title in zip(axs, ['isotropic', 'powerlaw'], ['Uniform noise (high effective rank)', 'Power-law spectrum (low effective rank)']):
    d = b[b.spectrum == sp]
    ax.axhline(0.30, color=C['truth'], lw=1.2, ls=':', label='True CKA (fixed)')
    line(ax, d, 'dy', 'raw', 'Raw CKA', C['raw'], '--')
    line(ax, d, 'dy', 'cal', 'Calibrated (Aristotelian)', C['cal'])
    line(ax, d, 'dy', 'deb', 'Debiased CKA', C['deb'])
    ax.set_xscale('log', base=2); ax.set_title(title, fontsize=10); ax.set_xlabel('Language-side width d (n = 1,024)')
    g = d.groupby('dy')[['cal', 'deb', 'tau']].mean()
    summary[f'fixed_{sp}'] = g.round(3).to_dict()
axs[0].set_ylabel('Alignment score'); axs[0].set_ylim(0, 0.45); axs[0].legend(fontsize=8, loc='lower left')
fig.tight_layout(); fig.savefig(F + 'fig1_width_fixed_truth.png'); plt.close()

# ---- Fig 2: rising truth
c = pd.read_csv(R + 'E1c_rising.csv')
rows = []
for (sp, cond), d in c.groupby(['spectrum', 'cond']):
    if d.dy.nunique() < 6: continue
    x = np.log2(d.dy)
    rows.append(dict(spectrum=sp, true_rise=d.groupby('dy').truth.mean().iloc[-1] - d.groupby('dy').truth.mean().iloc[0],
                     cal_slope=np.polyfit(x, d.cal, 1)[0] * 5, deb_slope=np.polyfit(x, d.deb, 1)[0] * 5,
                     raw_slope=np.polyfit(x, d.raw, 1)[0] * 5))
rise = pd.DataFrame(rows); summary['rising'] = rise.round(3).to_dict('records')
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, sp, title in zip(axs, ['isotropic', 'powerlaw'], ['Uniform noise', 'Power-law spectrum']):
    d = rise[rise.spectrum == sp].sort_values('true_rise')
    ax.plot(d.true_rise, d.true_rise, color=C['truth'], ls=':', label='Truth (ideal)')
    ax.plot(d.true_rise, d.cal_slope, color=C['cal'], marker='o', label='Calibrated (Aristotelian)')
    ax.plot(d.true_rise, d.deb_slope, color=C['deb'], marker='o', label='Debiased CKA')
    ax.axhline(0, color='k', lw=0.6); ax.set_title(title, fontsize=10)
    ax.set_xlabel('True rise in CKA from d=256 to d=8192')
axs[0].set_ylabel('Measured rise (fitted)'); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(F + 'fig2_rising_truth.png'); plt.close()

# ---- Fig 3: effective rank governs the penalty, with the closed-form prediction
k = pd.read_csv(R + 'E1k_effrank.csv'); n = 1024
k['pred_raw'] = (k.truth + np.sqrt(k.reff_x * k.reff_y) / n) / np.sqrt((1 + k.reff_x / n) * (1 + k.reff_y / n))
k['shrink'] = 1 / np.sqrt((1 + k.reff_x / n) * (1 + k.reff_y / n))
k['penalty'] = k.truth - k.cal
summary['effrank'] = k.groupby(['alpha', 'dy'])[['reff_y', 'raw', 'pred_raw', 'cal', 'deb']].mean().round(3).reset_index().to_dict('records')
summary['pred_raw_corr'] = float(np.corrcoef(k.raw, k.pred_raw)[0, 1]); summary['pred_raw_mae'] = float(np.mean(np.abs(k.raw - k.pred_raw)))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
sc = axs[0].scatter(k.reff_y / n, k.cal, c=k.alpha, cmap='viridis', s=14)
axs[0].scatter(k.reff_y / n, k.deb, c='none', edgecolors=C['deb'], s=14, label='Debiased CKA')
axs[0].axhline(0.30, color=C['truth'], ls=':', lw=1.2, label='Truth')
axs[0].set_xscale('log'); axs[0].set_xlabel('Effective rank of language side / n'); axs[0].set_ylabel('Score')
axs[0].set_title('Calibrated score (filled, by spectrum exponent)', fontsize=10); axs[0].legend(fontsize=8, loc='lower left')
plt.colorbar(sc, ax=axs[0], label='spectrum exponent α')
axs[1].scatter(k.pred_raw, k.raw, s=12, color=C['raw']); lim = [k.raw.min() * 0.95, k.raw.max() * 1.05]
axs[1].plot(lim, lim, 'k:', lw=1); axs[1].set_xlabel('Predicted raw CKA (closed form)'); axs[1].set_ylabel('Simulated raw CKA')
axs[1].set_title(f'Shrinkage formula check (r = {summary["pred_raw_corr"]:.3f})', fontsize=10)
fig.tight_layout(); fig.savefig(F + 'fig3_effective_rank.png'); plt.close()

# ---- Fig 4: outliers hide global but not local alignment
g = pd.read_csv(R + 'E1g_outlier_local.csv')
g['setting'] = g.apply(lambda r: 'none' if r.q == 0 else f'{int(r.q)} dims ×{int(r.scale)}', axis=1)
order = ['none', '2 dims ×10', '2 dims ×30', '8 dims ×30']
summary['outlier_local'] = g.groupby(['spectrum', 'setting'])[['truth_with_outliers', 'cal', 'deb', 'mknn', 'mknn_cal']].mean().round(3).reset_index().to_dict('records')
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, sp, title in zip(axs, ['isotropic', 'powerlaw'], ['Uniform noise', 'Power-law spectrum']):
    d = g[g.spectrum == sp].groupby('setting').mean(numeric_only=True).reindex(order)
    base = d.iloc[0]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, d.deb / base.deb, w, color=C['deb'], label='Global: debiased CKA')
    ax.bar(x + w / 2, d.mknn / base.mknn, w, color=C['mknn'], label='Local: mutual k-NN')
    ax.set_xticks(x, order, fontsize=8); ax.set_title(title, fontsize=10); ax.axhline(1, color='k', lw=0.6)
axs[0].set_ylabel('Relative to no outliers'); axs[0].legend(fontsize=8)
fig.suptitle('Outlier dimensions with no shared signal', fontsize=10)
fig.tight_layout(); fig.savefig(F + 'fig4_outliers_global_vs_local.png'); plt.close()

# ---- Fig 5: sample size
s = pd.read_csv(R + 'E1n_samples.csv')
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, sp, title in zip(axs, ['isotropic', 'powerlaw'], ['Uniform noise, d = 4096', 'Power-law spectrum, d = 4096']):
    d = s[s.spectrum == sp]
    ax.axhline(0.30, color=C['truth'], ls=':', lw=1.2, label='Truth')
    line(ax, d, 'n', 'cal', 'Calibrated', C['cal']); line(ax, d, 'n', 'deb', 'Debiased', C['deb'])
    ax.set_xscale('log', base=2); ax.set_xlabel('Number of image–caption pairs n'); ax.set_title(title, fontsize=10)
axs[0].set_ylabel('Score'); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(F + 'fig5_sample_size.png'); plt.close()
summary['samples'] = s.groupby(['spectrum', 'n'])[['cal', 'deb', 'tau']].mean().round(3).reset_index().to_dict('records')

# ---- tables: null, depth, power, rbf
a = pd.read_csv(R + 'E1a_null.csv')
na = a.groupby('spectrum')[['raw', 'cal', 'deb', 'mknn_cal', 'rbf_cal']].agg(['mean', 'max']).round(4); na.columns = ['_'.join(c) for c in na.columns]; summary['null'] = na.reset_index().to_dict('records')
summary['null_false_pos'] = a.groupby('spectrum')[['p', 'mknn_p']].apply(lambda d: (d < 0.05).mean()).round(3).to_dict()
dd = pd.read_csv(R + 'E1d_depth.csv')
summary['depth'] = dd.groupby(['spectrum', 'L'])[['tau_max', 'cal_max', 'heldout_deb']].mean().round(3).reset_index().to_dict('records')
e = pd.read_csv(R + 'E1e_power.csv')
pw = e.groupby(['spectrum', 'cond', 'dy']).apply(lambda d: pd.Series({'cka_power': (d.p < 0.05).mean(), 'mknn_power': (d.mknn_p < 0.05).mean(),
     'cal': d.cal.mean(), 'deb': d.deb.mean(), 'mknn_cal': d.mknn_cal.mean()}), include_groups=False).round(3).reset_index()
summary['power'] = pw.to_dict('records')
r = pd.read_csv(R + 'E1r_rbf.csv')
summary['rbf'] = r.groupby(['target', 'dy', 'frac'])[['rbf', 'rbf_cal']].mean().round(3).reset_index().to_dict('records')
import json; json.dump(summary, open('summary.json', 'w'), indent=1, default=str)
print('ok')
