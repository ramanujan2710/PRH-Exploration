# ===================== Alignment engine (all layer pairs at once, GPU) =====================
# Every metric is computed for ALL (layer_x, layer_y) pairs with one matmul on flattened
# n x n matrices. Permutation nulls permute one side's matrices with the SAME permutation
# for every layer, so the max-over-layer-pairs aggregate is calibrated as in the Aristotelian paper.
import math, numpy as np, torch

VARIANTS = ("huh", "noclip", "std")
# huh    : clamp |x| at the per-sample 95th percentile (averaged), then L2-normalise each sample
#          (exactly Huh et al.'s prepare_features + compute_score(normalize=True))
# noclip : L2-normalise only
# std    : per-layer, per-dimension z-score over samples, then L2-normalise

def huh_clip(feats, q=0.95):
    """Huh et al. remove_outliers(exact=False): per-sample q-quantile of |x| over all layers & dims
    (linear interpolation, as torch.quantile), averaged over samples, then clamp. Uses kthvalue so it
    also works on tensors larger than torch.quantile's size limit."""
    a = feats.abs().flatten(start_dim=1); m = a.shape[1]
    pos = q * (m - 1); lo = int(math.floor(pos)); hi = int(math.ceil(pos)); w = pos - lo
    qs = []
    for i in range(a.shape[0]):
        v_lo = a[i].kthvalue(lo + 1).values
        v_hi = a[i].kthvalue(hi + 1).values if hi != lo else v_lo
        qs.append(v_lo + (v_hi - v_lo) * w)
    t = torch.stack(qs).mean()
    return feats.clamp(-t, t)

def preprocess(feats, variant):
    """feats: N x L x D float32 tensor -> list of L tensors (N x D), ready for metrics."""
    if variant == "huh":
        f = huh_clip(feats)
    elif variant == "noclip":
        f = feats
    elif variant == "std":
        f = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-6)
    else:
        raise ValueError(variant)
    return [torch.nn.functional.normalize(f[:, l, :], dim=-1) for l in range(f.shape[1])]

def _center(K):
    m = K.mean(0)
    return K - m[None, :] - m[:, None] + m.mean()

class SideStats:
    """Per-model, per-variant matrices needed for every metric, flattened over layers."""
    def __init__(self, layers, k=10, half_idx=None):
        n = layers[0].shape[0]; self.n = n; self.L = len(layers); self.k = k
        Kc, Kt, adj, reff = [], [], [], []
        Kt_a, Kt_b, Kc_a, Kc_b = [], [], [], []
        ia, ib = half_idx if half_idx is not None else (None, None)
        for X in layers:
            K = X @ X.T
            c = _center(K); Kc.append(c.flatten())
            Kd = X.double() @ X.double().T                                     # fp64: unbiased HSIC cancels large terms
            t = Kd.clone(); t.fill_diagonal_(0); Kt.append(t.flatten())
            reff.append((torch.trace(c) ** 2 / (c * c).sum()).item())
            S = K.clone(); S.fill_diagonal_(-1e8)
            idx = S.topk(k, dim=1).indices
            A = torch.zeros(n, n, device=X.device); A.scatter_(1, idx, 1.0); adj.append(A.flatten())
            if ia is not None:
                for sub, Kt_s, Kc_s in ((ia, Kt_a, Kc_a), (ib, Kt_b, Kc_b)):
                    Ks = K[sub][:, sub]; Kc_s.append(_center(Ks).flatten())
                    ts = Kd[sub][:, sub].clone(); ts.fill_diagonal_(0); Kt_s.append(ts.flatten())
        self.Kc = torch.stack(Kc); self.Kc_norm = self.Kc.norm(dim=1)
        self.Kt = torch.stack(Kt); self.adj = torch.stack(adj)
        self.reff = np.array(reff)
        self.hu = self._hsic_parts(self.Kt, n)
        if ia is not None:
            self.Kc_a = torch.stack(Kc_a); self.Kc_b = torch.stack(Kc_b)
            self.Kc_a_norm = self.Kc_a.norm(dim=1); self.Kc_b_norm = self.Kc_b.norm(dim=1)
            self.hu_a = self._hsic_parts(torch.stack(Kt_a), len(ia))
            self.hu_b = self._hsic_parts(torch.stack(Kt_b), len(ib))

    @staticmethod
    def _hsic_parts(Kt, n):
        M = Kt.view(Kt.shape[0], n, n)
        return dict(flat=Kt, tot=Kt.sum(1), rows=M.sum(2), n=n)

def _hsic_u_pairs(A, B):
    """Unbiased HSIC (Song et al. 2012) for all layer pairs. A, B: dicts from _hsic_parts."""
    n = A["n"]
    t1 = A["flat"] @ B["flat"].T
    t2 = A["tot"][:, None] * B["tot"][None, :] / ((n - 1) * (n - 2))
    t3 = 2.0 * (A["rows"] @ B["rows"].T) / (n - 2)
    return (t1 + t2 - t3) / (n * (n - 3))

def _hsic_u_self(A):
    n = A["n"]
    return ((A["flat"] ** 2).sum(1) + A["tot"] ** 2 / ((n - 1) * (n - 2))
            - 2.0 * (A["rows"] ** 2).sum(1) / (n - 2)) / (n * (n - 3))

def unbiased_cka_pairs(A, B, rel_tol=1e-6):
    """fp64. Layers whose self-HSIC is ~0 or negative (near-constant representations, e.g. a ViT's
    first-block CLS token) are undefined for CKA and returned as NaN instead of dividing by ~0."""
    sa, sb = _hsic_u_self(A), _hsic_u_self(B)
    va = sa > rel_tol * sa.max().clamp_min(1e-300); vb = sb > rel_tol * sb.max().clamp_min(1e-300)
    u = _hsic_u_pairs(A, B) / torch.sqrt(sa.clamp_min(1e-300)[:, None] * sb.clamp_min(1e-300)[None, :])
    u[~va, :] = float("nan"); u[:, ~vb] = float("nan")
    return u.float()

def _nanargmax2(M):
    i = int(torch.nan_to_num(M, nan=-float("inf")).argmax()); return divmod(i, M.shape[1])

def linear_cka_pairs(x, y):
    return (x.Kc @ y.Kc.T) / (x.Kc_norm[:, None] * y.Kc_norm[None, :])

def mknn_pairs(x, y):
    return (x.adj @ y.adj.T) / (x.n * x.k)

def _perm_flat_index(p, n):
    return (p[:, None] * n + p[None, :]).flatten()

def calibrate(obs, null, alpha=0.05, smax=1.0):
    K = len(null)
    allv = np.sort(np.append(null, obs))
    tau = allv[int(math.ceil((1 - alpha) * (K + 1))) - 1]
    cal = max((obs - tau) / (smax - tau), 0.0) if smax > tau else 0.0
    p = (1 + np.sum(null >= obs)) / (K + 1)
    return cal, tau, p

def align_pair(x, y, perms, alpha=0.05, perms_half=None):
    """x: language SideStats, y: vision SideStats. Returns dict of scalar results + matrices."""
    n = x.n
    cka = linear_cka_pairs(x, y); ucka = unbiased_cka_pairs(x.hu, y.hu); mk = mknn_pairs(x, y)
    i_c, j_c = divmod(int(cka.argmax()), cka.shape[1])
    i_u, j_u = _nanargmax2(ucka)
    i_m, j_m = divmod(int(mk.argmax()), mk.shape[1])
    null_cka_max, null_mk_max, null_cka_fixed, null_arg = [], [], [], []
    for p in perms:
        fi = _perm_flat_index(p, n)
        Kc_p = y.Kc.index_select(1, fi); A_p = y.adj.index_select(1, fi)
        s = (x.Kc @ Kc_p.T) / (x.Kc_norm[:, None] * y.Kc_norm[None, :])
        a_ = int(s.argmax()); null_arg.append(a_)
        null_cka_max.append(s.view(-1)[a_].item()); null_cka_fixed.append(s[i_c, j_c].item())
        null_mk_max.append(((x.adj @ A_p.T) / (n * x.k)).max().item())
    cka_max = cka.max().item(); mk_max = mk.max().item()
    cal_agg, tau_agg, p_agg = calibrate(cka_max, np.array(null_cka_max), alpha)
    cal_fix, tau_fix, _ = calibrate(cka_max, np.array(null_cka_fixed), alpha)
    mk_cal, mk_tau, mk_p = calibrate(mk_max, np.array(null_mk_max), alpha)
    out = dict(cka_max=cka_max, cka_cal=cal_agg, cka_tau=tau_agg, cka_p=p_agg,
               cka_cal_fixedpair=cal_fix, cka_tau_fixedpair=tau_fix,
               cka_null_mean=float(np.mean(null_cka_max)),
               ucka_max=float(np.nanmax(ucka.cpu().numpy())), mknn_max=mk_max, mknn_cal=mk_cal, mknn_tau=mk_tau, mknn_p=mk_p,
               best_cka=(i_c, j_c), best_ucka=(i_u, j_u), best_mknn=(i_m, j_m),
               reff_x=float(x.reff[i_c]), reff_y=float(y.reff[j_c]))
    out["pred_shrink"] = 1.0 / math.sqrt((1 + out["reff_x"] / n) * (1 + out["reff_y"] / n))
    # which layer pair sets the aggregate null threshold most often, and how low-rank is it?
    mode = max(set(null_arg), key=null_arg.count); im, jm = divmod(mode, cka.shape[1])
    out.update(nullmax_pair=(im, jm), nullmax_pair_freq=null_arg.count(mode) / len(null_arg),
               nullmax_reff_x=float(x.reff[im]), nullmax_reff_y=float(y.reff[jm]))
    if hasattr(x, "hu_a") and hasattr(y, "hu_a"):
        ua = unbiased_cka_pairs(x.hu_a, y.hu_a)
        ia, ja = _nanargmax2(ua)
        ub = unbiased_cka_pairs(x.hu_b, y.hu_b)
        out["ucka_heldout"] = ub[ia, ja].item(); out["best_heldout"] = (ia, ja)
        # held-out CALIBRATED CKA: pick the layer pair on half A by UNBIASED CKA (raw CKA is inflated by the
        # chance offset and would favour high-rank layers), then calibrate that single pair's raw CKA on half B
        ic, jc = ia, ja
        xb, yb = x.Kc_b[ic], y.Kc_b[jc]; nb = int(round(math.sqrt(xb.numel())))
        obs = (xb @ yb / (x.Kc_b_norm[ic] * y.Kc_b_norm[jc])).item()
        nulls = [(xb @ yb.index_select(0, _perm_flat_index(p, nb)) / (x.Kc_b_norm[ic] * y.Kc_b_norm[jc])).item()
                 for p in (perms_half or [])]
        if nulls:
            c_, t_, p_ = calibrate(obs, np.array(nulls), alpha)
            out.update(cka_heldout_raw=obs, cka_heldout_cal=c_, cka_heldout_tau=t_, best_heldout_cka=(ic, jc))
    mats = dict(cka=cka.cpu().numpy(), ucka=ucka.cpu().numpy(), mknn=mk.cpu().numpy())
    return out, mats

def layer_diagnostics(feats):
    """Outlier diagnostics on raw (un-normalised) features. feats: N x L x D."""
    rows = []
    for l in range(feats.shape[1]):
        f = feats[:, l, :]
        v = f.var(0); vs = torch.sort(v, descending=True).values; tot = v.sum()
        absf = f.abs()
        fc = f - f.mean(0, keepdim=True)
        G = fc @ fc.T
        rows.append(dict(layer=l, top1_var_share=(vs[0] / tot).item(), top10_var_share=(vs[:10].sum() / tot).item(),
                         max_over_median_abs=(absf.max() / absf.median().clamp_min(1e-12)).item(),
                         reff_raw=(torch.trace(G) ** 2 / (G * G).sum()).item()))
    return rows
