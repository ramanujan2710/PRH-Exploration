"""E1 library: planted-signal worlds with exact population CKA, and the estimators under test.

World:  X = Z A^T + U C^T + E_x * sqrt(Dx)     (vision,   width dx)
        Y = Z B^T + V P^T + E_y * sqrt(Dy)     (language, width dy)
Z (shared, r dims), U, V (private, rp dims), E (noise) are i.i.d. standard normal.
Population linear CKA = ||Sxy||_F^2 / (||Sxx||_F ||Syy||_F), computed exactly.
"""
import numpy as np

# ---------------------------------------------------------------- world
def powerlaw(m, alpha):
    w = np.arange(1, m + 1, dtype=float) ** (-alpha)
    return w / w.mean()

def make_side(rng, d, r, rp, alpha, signal_scale=1.0):
    """Loadings for shared (r) and private (rp) latents, and a noise-variance profile."""
    lat_w = np.sqrt(powerlaw(r, alpha))           # latent strengths decay
    prv_w = np.sqrt(powerlaw(rp, alpha))
    A = rng.standard_normal((d, r)) / np.sqrt(r) * lat_w * signal_scale
    C = rng.standard_normal((d, rp)) / np.sqrt(rp) * prv_w * signal_scale
    noise_profile = powerlaw(d, alpha) if alpha > 0 else np.ones(d)
    return A, C, noise_profile

def _fro2_cov(M, D):
    """||M M^T + diag(D)||_F^2 without forming the d x d matrix."""
    G = M.T @ M
    return np.sum(G * G) + 2 * np.sum(D * np.sum(M * M, 1)) + np.sum(D * D)

def pop_cka(A, C, Dx, B, P, Dy):
    hxy = np.trace((A.T @ A) @ (B.T @ B))
    hxx = _fro2_cov(np.hstack([A, C]), Dx)
    hyy = _fro2_cov(np.hstack([B, P]), Dy)
    return hxy / np.sqrt(hxx * hyy)

def tune_noise(target, A, C, Dx, B, P, profile):
    """Find s so that pop_cka(..., Dy = s^2 * profile) == target (monotone in s)."""
    lo, hi = 1e-5, 1e3
    if pop_cka(A, C, Dx, B, P, lo**2 * profile) < target:
        raise ValueError("target above achievable maximum")
    for _ in range(100):
        m = np.sqrt(lo * hi)
        if pop_cka(A, C, Dx, B, P, m**2 * profile) > target: lo = m
        else: hi = m
    return np.sqrt(lo * hi)

def sample_side(rng, n, Z, A, C, D):
    d = A.shape[0]
    U = rng.standard_normal((n, C.shape[1]))
    E = rng.standard_normal((n, d)) * np.sqrt(D)
    return Z @ A.T + U @ C.T + E

# ---------------------------------------------------------------- estimators
def center_gram(K):
    m = K.mean(0)
    return K - m[None, :] - m[:, None] + m.mean()

def cka_from_centered(Kc, Lc):
    return np.sum(Kc * Lc) / np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))

def hsic_unbiased(K, L):
    """Song et al. (2012) unbiased HSIC estimator."""
    K = K.copy(); L = L.copy()
    np.fill_diagonal(K, 0.0); np.fill_diagonal(L, 0.0)
    n = K.shape[0]
    t1 = np.sum(K * L)
    t2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    t3 = 2.0 * (K.sum(0) @ L.sum(0)) / (n - 2)
    return (t1 + t2 - t3) / (n * (n - 3))

def cka_debiased(K, L):
    return hsic_unbiased(K, L) / np.sqrt(hsic_unbiased(K, K) * hsic_unbiased(L, L))

def knn_adj(X, k):
    """Mutual-kNN neighbour adjacency on L2-normalised features (as in Huh et al.)."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    M = np.zeros(S.shape, dtype=np.float32)
    np.put_along_axis(M, idx, 1.0, axis=1)
    return M

def rbf_centered(K, frac):
    sq = np.diag(K)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * K, 0)
    sigma2 = (frac * np.median(np.sqrt(D2[np.triu_indices_from(D2, 1)]))) ** 2
    return center_gram(np.exp(-D2 / (2 * sigma2)))

def threshold(obs, null, alpha=0.05):
    K = len(null)
    allv = np.sort(np.append(null, obs))
    return allv[int(np.ceil((1 - alpha) * (K + 1))) - 1]

def calibrate(obs, null, alpha=0.05, smax=1.0):
    tau = threshold(obs, null, alpha)
    cal = max((obs - tau) / (smax - tau), 0.0)
    p = (1 + np.sum(null >= obs)) / (len(null) + 1)
    return cal, tau, p

def measure(X, Y, rng, nperm=200, k=10, rbf_frac=0.25, do_local=True):
    """All estimators on one dataset, sharing the same permutations (as the paper does)."""
    Kx, Ky = X @ X.T, Y @ Y.T
    Kxc, Kyc = center_gram(Kx), center_gram(Ky)
    nx, ny = np.sqrt(np.sum(Kxc * Kxc)), np.sqrt(np.sum(Kyc * Kyc))
    raw = np.sum(Kxc * Kyc) / (nx * ny)
    out = dict(raw=raw, deb=cka_debiased(Kx, Ky))
    n = X.shape[0]
    perms = [rng.permutation(n) for _ in range(nperm)]
    null = np.array([np.sum(Kxc[np.ix_(p, p)] * Kyc) for p in perms]) / (nx * ny)
    out['cal'], out['tau'], out['p'] = calibrate(raw, null)
    out['null_mu'] = null.mean()
    if do_local:
        Mx, My = knn_adj(X, k), knn_adj(Y, k)
        mk = np.sum(Mx * My) / (n * k)
        mnull = np.array([np.sum(Mx[np.ix_(p, p)] * My) for p in perms]) / (n * k)
        out['mknn'] = mk
        out['mknn_cal'], out['mknn_tau'], out['mknn_p'] = calibrate(mk, mnull)
        Rx, Ry = rbf_centered(Kx, rbf_frac), rbf_centered(Ky, rbf_frac)
        rnx, rny = np.sqrt(np.sum(Rx * Rx)), np.sqrt(np.sum(Ry * Ry))
        rb = np.sum(Rx * Ry) / (rnx * rny)
        rnull = np.array([np.sum(Rx[np.ix_(p, p)] * Ry) for p in perms]) / (rnx * rny)
        out['rbf'] = rb
        out['rbf_cal'], _, out['rbf_p'] = calibrate(rb, rnull)
    return out
