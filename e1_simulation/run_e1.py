"""Runs E1a-E1f and the sample-size sweep. Usage: python run_e1.py <exp> ; results -> results/<exp>.csv"""
import sys, time, numpy as np, pandas as pd
from e1lib import *

DX, R, RP, N = 1024, 32, 32, 1024
WIDTHS = [256, 512, 1024, 2048, 4096, 8192]
SPECTRA = {'isotropic': 0.0, 'powerlaw': 1.0}
OUT = 'results'

def world(rng, dy, alpha, target):
    A, C, px = make_side(rng, DX, R, RP, alpha)
    B, P, py = make_side(rng, dy, R, RP, alpha)
    Dx = 0.5 * px
    if target is None:          # null world: language side has no shared latent
        B = np.zeros_like(B); s = 1.0
    else:
        s = tune_noise(target, A, C, Dx, B, P, py)
    Dy = s**2 * py
    truth = 0.0 if target is None else pop_cka(A, C, Dx, B, P, Dy)
    return (A, C, Dx), (B, P, Dy), truth

def draw(rng, n, vis, lan):
    Z = rng.standard_normal((n, R))
    return sample_side(rng, n, Z, *vis), sample_side(rng, n, Z, *lan)

def sweep(name, targets_fn, reps, do_local, seed):
    rng = np.random.default_rng(seed); rows = []
    for sname, alpha in SPECTRA.items():
        for label, tfn in targets_fn.items():
            for dy in WIDTHS:
                for rep in range(reps):
                    try: vis, lan, truth = world(rng, dy, alpha, tfn(dy))
                    except ValueError: continue          # target not reachable in this spectrum
                    X, Y = draw(rng, N, vis, lan)
                    m = measure(X, Y, rng, do_local=do_local)
                    rows.append(dict(spectrum=sname, cond=label, dy=dy, ratio=dy / N, rep=rep, truth=truth, **m))
            print(name, sname, label, 'done', time.strftime('%X'), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/{name}.csv', index=False)

def rising(rise):
    return lambda dy: 0.25 + rise * np.log2(dy / 256) / 5.0   # +rise over the 256->8192 range

exp = sys.argv[1]
if exp == 'a':   # null: no shared signal
    sweep('E1a_null', {'null': lambda dy: None}, reps=10, do_local=True, seed=10)
elif exp == 'b': # fixed truth 0.30
    sweep('E1b_fixed', {'fixed0.30': lambda dy: 0.30}, reps=10, do_local=True, seed=11)
elif exp == 'c': # rising truth, several slopes
    sweep('E1c_rising', {f'rise{r:.2f}': rising(r) for r in [0.0, 0.05, 0.10, 0.15, 0.20]},
          reps=5, do_local=False, seed=12)
elif exp == 'e': # weak signals: detection power, global vs local
    sweep('E1e_power', {f'weak{t:.2f}': (lambda t: (lambda dy: t))(t) for t in [0.02, 0.05]},
          reps=15, do_local=True, seed=13)
elif exp == 'k': # effective-rank sweep: spectrum steepness x width, truth fixed at 0.30
    def reff(M, D):
        tr = np.sum(M * M) + np.sum(D)
        return tr**2 / (np.sum((M.T @ M)**2) + 2 * np.sum(D * np.sum(M * M, 1)) + np.sum(D * D))
    rng = np.random.default_rng(18); rows = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for dy in [256, 1024, 4096, 8192]:
            for rep in range(4):
                (A, C, Dx), (B, P, Dy), truth = world(rng, dy, alpha, 0.30)
                X, Y = draw(rng, N, (A, C, Dx), (B, P, Dy))
                m = measure(X, Y, rng, do_local=False)
                rows.append(dict(alpha=alpha, dy=dy, rep=rep, truth=truth,
                                 reff_x=reff(np.hstack([A, C]), Dx), reff_y=reff(np.hstack([B, P]), Dy), **m))
            print('k', alpha, dy, time.strftime('%X'), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/E1k_effrank.csv', index=False)
elif exp == 'n': # sample-size sweep at d=4096, truth 0.30
    rng = np.random.default_rng(14); rows = []
    for sname, alpha in SPECTRA.items():
        for n in [512, 1024, 2048, 4096]:
            for rep in range(4):
                vis, lan, truth = world(rng, 4096, alpha, 0.30)
                X, Y = draw(rng, n, vis, lan)
                m = measure(X, Y, rng, do_local=False)
                rows.append(dict(spectrum=sname, n=n, ratio=4096 / n, rep=rep, truth=truth, **m))
            print('n', sname, n, time.strftime('%X'), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/E1n_samples.csv', index=False)
elif exp == 'f': # outlier ("massive") dimensions on the language side, d=4096
    rng = np.random.default_rng(15); rows = []
    for sname, alpha in SPECTRA.items():
        for q in [0, 2, 8]:
            for scale in [30.0, 100.0]:
                if q == 0 and scale == 100.0: continue
                for rep in range(8):
                    vis, (B, P, Dy), truth_clean = world(rng, 4096, alpha, 0.30)
                    Dy2 = Dy.copy(); tot = Dy.mean()
                    oi = rng.choice(len(Dy2), q, replace=False)
                    Dy2[oi] = scale**2 * tot          # huge-variance dims carrying no shared signal
                    truth_out = pop_cka(*vis, B, P, Dy2)
                    X, Y = draw(rng, N, vis, (B, P, Dy2))
                    m = measure(X, Y, rng, do_local=False)
                    Ys = (Y - Y.mean(0)) / Y.std(0)    # fix: per-dimension standardisation
                    ms = measure(X, Ys, rng, do_local=False)
                    rows.append(dict(spectrum=sname, q=q, scale=scale, rep=rep, truth_clean=truth_clean,
                                     truth_with_outliers=truth_out, **m,
                                     **{f'std_{k}': v for k, v in ms.items()}))
            print('f', sname, q, time.strftime('%X'), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/E1f_outliers.csv', index=False)
elif exp == 'r': # RBF bandwidth sweep: is small-bandwidth RBF-CKA informative at all?
    rng = np.random.default_rng(16); rows = []
    for target in [None, 0.30]:
        for dy in [1024, 4096]:
            for rep in range(5):
                vis, lan, truth = world(rng, dy, 1.0, target)
                X, Y = draw(rng, N, vis, lan)
                for frac in [0.1, 0.25, 0.5, 1.0, 2.0]:
                    m = measure(X, Y, rng, nperm=100, rbf_frac=frac)
                    rows.append(dict(target=str(target), dy=dy, rep=rep, frac=frac, truth=truth,
                                     rbf=m['rbf'], rbf_cal=m['rbf_cal'], rbf_p=m['rbf_p']))
    pd.DataFrame(rows).to_csv(f'{OUT}/E1r_rbf.csv', index=False)
    print('r done', flush=True)
elif exp == 'd': # depth: max over layers, calibrated with the aggregate null, vs held-out selection
    rng = np.random.default_rng(17); rows = []; dy = 2048; rho = 0.9
    for sname, alpha in SPECTRA.items():
        for L in [8, 16, 32, 48, 80]:
            for rep in range(5):
                A, C, px = make_side(rng, DX, R, RP, alpha); B, P, py = make_side(rng, dy, R, RP, alpha)
                Dx = 0.5 * px
                s = tune_noise(0.30, A, C, Dx, B, P, py); Dy = s**2 * py
                pos = (np.arange(L) + 0.5) / L
                w = np.exp(-(pos - 0.6)**2 / (2 * 0.15**2)); w = w / w.max()   # best layer has w = 1
                truths = np.array([pop_cka(A, C, Dx, B * wl, P, Dy) for wl in w])
                Z = rng.standard_normal((N, R)); X = sample_side(rng, N, Z, A, C, Dx)
                V = rng.standard_normal((N, RP)); E = rng.standard_normal((N, dy))
                Kx = X @ X.T; Kxc = center_gram(Kx); nx = np.sqrt(np.sum(Kxc**2))
                half = rng.permutation(N); ia, ib = half[:N // 2], half[N // 2:]
                grams = []
                for l in range(L):
                    if l > 0:
                        E = np.sqrt(rho) * E + np.sqrt(1 - rho) * rng.standard_normal((N, dy))
                    Y = Z @ (B * w[l]).T + V @ P.T + E * np.sqrt(Dy)
                    grams.append(Y @ Y.T)
                cent = [center_gram(G) for G in grams]
                norms = np.array([np.sqrt(np.sum(G**2)) for G in cent])
                raws = np.array([np.sum(Kxc * G) for G in cent]) / (nx * norms)
                perms = [rng.permutation(N) for _ in range(200)]
                nullmax = []
                for p in perms:
                    Kp = Kxc[np.ix_(p, p)]
                    nullmax.append(np.max(np.array([np.sum(Kp * G) for G in cent]) / (nx * norms)))
                cal_max, tau_max, _ = calibrate(raws.max(), np.array(nullmax))
                # held-out: choose layer on half A by debiased CKA, evaluate on half B
                deb_a = [cka_debiased(Kx[np.ix_(ia, ia)], G[np.ix_(ia, ia)]) for G in grams]
                lsel = int(np.argmax(deb_a))
                Kb, Gb = Kx[np.ix_(ib, ib)], grams[lsel][np.ix_(ib, ib)]
                deb_b = cka_debiased(Kb, Gb)
                Kbc, Gbc = center_gram(Kb), center_gram(Gb)
                rb = cka_from_centered(Kbc, Gbc)
                nb = np.array([cka_from_centered(Kbc[np.ix_(p, p)], Gbc)
                               for p in (rng.permutation(len(ib)) for _ in range(200))])
                cal_b, _, _ = calibrate(rb, nb)
                rows.append(dict(spectrum=sname, L=L, rep=rep, truth_best=truths.max(),
                                 truth_selected=truths[lsel], raw_max=raws.max(), tau_max=tau_max,
                                 cal_max=cal_max, heldout_deb=deb_b, heldout_cal=cal_b))
            print('d', sname, L, time.strftime('%X'), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/E1d_depth.csv', index=False)
if exp == 'g':   # outliers + local metrics: does mKNN survive when global CKA collapses?
    rng = np.random.default_rng(19); rows = []
    for sname, alpha in SPECTRA.items():
        for q, scale in [(0, 1.0), (2, 10.0), (2, 30.0), (8, 30.0)]:
            for rep in range(6):
                vis, (B, P, Dy), truth_clean = world(rng, 4096, alpha, 0.30)
                Dy2 = Dy.copy(); oi = rng.choice(len(Dy2), q, replace=False); Dy2[oi] = scale**2 * Dy.mean()
                X, Y = draw(rng, N, vis, (B, P, Dy2))
                m = measure(X, Y - Y.mean(0), rng, do_local=True)   # centred, as alignment code does
                rows.append(dict(spectrum=sname, q=q, scale=scale, rep=rep,
                                 truth_with_outliers=pop_cka(*vis, B, P, Dy2), **m))
            print('g', sname, q, scale, time.strftime('%X'), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/E1g_outlier_local.csv', index=False)
