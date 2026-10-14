import sys, json, time
import numpy as np
sys.path.insert(0, '.')
from e1lib import *

N, R, DX = 1024, 32, 1024
WIDTHS = [256, 512, 1024, 2048, 4096, 8192]


def f(x):
    return {k: float(v) for k, v in x.items()}


def save(name, rows):
    json.dump(rows, open(f'res_{name}.json', 'w'), indent=1)


def e1a():  # null: no shared signal
    rng = np.random.default_rng(10); rows = []
    for beta in [0.0, 0.5]:
        X = Side(DX, R, rng, beta=beta)
        for d in [256, 1024, 4096, 8192]:
            Y = Side(d, R, rng, beta=beta); Y.sigma = 1.0
            for rep in range(20):
                Z, Z2 = rng.standard_normal((N, R)), rng.standard_normal((N, R))
                m = measure(X.sample(Z, rng), Y.sample(Z2, rng), rng)
                rows.append(dict(beta=beta, d=d, rep=rep, **f(m)))
            save('e1a', rows); print('e1a', beta, d, flush=True)


def e1b():  # width sweep, fixed truth, 3 spectra
    rng = np.random.default_rng(11); rows = []
    for beta in [0.0, 0.5, 1.0]:
        X = Side(DX, R, rng, beta=beta)
        for d in WIDTHS:
            Y = Side(d, R, rng, beta=beta); tune_sigma(X, Y, 0.30)
            prx, pry = part_ratio(X), part_ratio(Y)
            for rep in range(5):
                Z = rng.standard_normal((N, R))
                m = measure(X.sample(Z, rng), Y.sample(Z, rng), rng, rbf=True)
                rows.append(dict(beta=beta, d=d, rep=rep, truth=pop_cka(X, Y), prx=prx, pry=pry,
                                 pred_gap=predicted_gap(pop_cka(X, Y), prx, pry, N), **f(m)))
            save('e1b', rows); print('e1b', beta, d, flush=True)


def e1c():  # rising truth: can calibration hide/flip it?
    rng = np.random.default_rng(12); rows = []
    ds = [256, 1024, 4096, 8192]
    for beta in [0.0, 0.5]:
        X = Side(DX, R, rng, beta=beta)
        for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for d in ds:
                t = 0.25 * (1 + g * np.log2(d / 256) / 5)
                Y = Side(d, R, rng, beta=beta); tune_sigma(X, Y, t)
                for rep in range(3):
                    Z = rng.standard_normal((N, R))
                    m = measure(X.sample(Z, rng), Y.sample(Z, rng), rng)
                    rows.append(dict(beta=beta, g=g, d=d, rep=rep, truth=t, pry=part_ratio(Y), **f(m)))
            save('e1c', rows); print('e1c', beta, g, flush=True)


def e1e():  # power: weak signal, local vs global detection
    rng = np.random.default_rng(14); rows = []
    X = Side(DX, R, rng, beta=0.0)
    for d in [256, 1024, 4096, 8192]:
        Y = Side(d, R, rng, beta=0.0); tune_sigma(X, Y, 0.04)
        for rep in range(25):
            Z = rng.standard_normal((N, R))
            m = measure(X.sample(Z, rng), Y.sample(Z, rng), rng)
            rows.append(dict(d=d, rep=rep, truth=0.04, pry=part_ratio(Y), **f(m)))
        save('e1e', rows); print('e1e', d, flush=True)


def e1f():  # outlier dimensions on the language side
    rng = np.random.default_rng(15); rows = []
    d, beta = 4096, 0.5
    X = Side(DX, R, rng, beta=beta)
    for q in [0, 1, 4, 16, 64]:
        Y = Side(d, R, rng, beta=beta, n_out=4, out_var=0.0); tune_sigma(X, Y, 0.30)
        clean_tr = Y.fro2_tr()[1]
        Y.out_var = q * clean_tr / 4  # 4 outlier dims carry q x the total clean variance
        truth_clean = 0.30; truth_dirty = pop_cka(X, Y)
        for rep in range(5):
            Z = rng.standard_normal((N, R)); Fx, Fy = X.sample(Z, rng), Y.sample(Z, rng)
            variants = {'raw': Fy,
                        'zscore': (Fy - Fy.mean(0)) / (Fy.std(0) + 1e-8),
                        'drop_top4': np.delete(Fy, np.argsort(Fy.var(0))[-4:], axis=1)}
            for vname, Fv in variants.items():
                m = measure(Fx, Fv.astype(np.float32), rng)
                rows.append(dict(q=q, variant=vname, rep=rep, truth_clean=truth_clean,
                                 truth_dirty=truth_dirty, **f(m)))
        save('e1f', rows); print('e1f', q, flush=True)


def e1g():  # sample size sweep
    rng = np.random.default_rng(16); rows = []
    d = 4096
    for beta in [0.0, 0.5]:
        X = Side(DX, R, rng, beta=beta); Y = Side(d, R, rng, beta=beta); tune_sigma(X, Y, 0.30)
        for n in [512, 1024, 2048, 4096]:
            for rep in range(3 if n < 4096 else 2):
                Z = rng.standard_normal((n, R))
                m = measure(X.sample(Z, rng), Y.sample(Z, rng), rng, kperm=200)
                rows.append(dict(beta=beta, n=n, rep=rep, truth=0.30, prx=part_ratio(X), pry=part_ratio(Y),
                                 pred_gap=predicted_gap(0.30, part_ratio(X), part_ratio(Y), n), **f(m)))
            save('e1g', rows); print('e1g', beta, n, flush=True)


def e1h():  # RBF bandwidth sweep: does 'no local-distance alignment' follow from small bandwidth?
    rng = np.random.default_rng(17); rows = []
    X = Side(DX, R, rng, beta=0.5)
    for d in [1024, 8192]:
        Y = Side(d, R, rng, beta=0.5); tune_sigma(X, Y, 0.30)
        for mult in [0.25, 0.5, 1.0, 2.0, 4.0]:
            for rep in range(3):
                Z = rng.standard_normal((N, R))
                m = measure(X.sample(Z, rng), Y.sample(Z, rng), rng, rbf=True, rbf_mult=mult, kperm=100)
                rows.append(dict(d=d, mult=mult, rep=rep, truth=0.30, **f(m)))
            save('e1h', rows); print('e1h', d, mult, flush=True)


def e1d():  # depth: max over layer pairs vs held-out layer selection
    rng = np.random.default_rng(13); rows = []
    d, beta, rho, LV = 2048, 0.5, 0.9, 6
    X = Side(DX, R, rng, beta=beta); Y = Side(d, R, rng, beta=beta); tune_sigma(X, Y, 0.30)
    ax = np.exp(-((np.linspace(0, 1, LV) - 0.6) ** 2) / (2 * 0.2 ** 2)); ax /= ax.max()

    def layers(S, Z, a, n):
        U = rng.standard_normal((n, S.C.shape[1])); priv = U @ S.C.T
        sl = np.sqrt(S.lam()); Nn = rng.standard_normal((n, S.d)); out = []
        for i, ai in enumerate(a):
            if i: Nn = rho * Nn + np.sqrt(1 - rho ** 2) * rng.standard_normal((n, S.d))
            out.append((ai * (Z @ S.B.T) + priv + Nn * sl).astype(np.float32))
        return out

    for L in [12, 24, 40, 60, 80]:
        t = np.linspace(0, 1, L); ay = np.exp(-((t - 0.6) ** 2) / (2 * 0.2 ** 2)); ay[np.argmax(ay)] = 1.0; ay /= ay.max()
        T = np.array([[pop_cka(X, Y, ax=a, ay=b) for b in ay] for a in ax])
        for rep in range(3):
            Z = rng.standard_normal((N, R))
            Fx, Fy = layers(X, Z, ax, N), layers(Y, Z, ay, N)
            Kx = [cgram(F) for F in Fx]; Ky = [cgram(F) for F in Fy]
            gx = np.stack([nrm(K) for K in Kx]); gy = np.stack([nrm(K) for K in Ky]).reshape(L, -1)
            raw = gx.reshape(LV, -1) @ gy.T
            nullmax = []
            for _ in range(K_PERM):
                p = rng.permutation(N)
                nullmax.append((gx[:, p][:, :, p].reshape(LV, -1) @ gy.T).max())
            tau, scal_max, _ = calib(raw.max(), np.array(nullmax))
            # oracle: calibrate the truly-best pair alone (no search)
            i, j = np.unravel_index(T.argmax(), T.shape)
            mo = measure(Fx[i], Fy[j], rng)
            # held-out: select on half A, evaluate on half B
            idx = rng.permutation(N); A, B = idx[:N // 2], idx[N // 2:]
            rawA = np.stack([nrm(K[np.ix_(A, A)]) for K in Kx]).reshape(LV, -1) @ \
                np.stack([nrm(K[np.ix_(A, A)]) for K in Ky]).reshape(L, -1).T
            si, sj = np.unravel_index(rawA.argmax(), rawA.shape)
            mh = measure(Fx[si][B], Fy[sj][B], rng)
            rows.append(dict(L=L, rep=rep, truth_best=float(T.max()), raw_max=float(raw.max()),
                             tau_max=float(tau), scal_max=float(scal_max),
                             oracle_scal=mo['scal'], oracle_deb=mo['deb'],
                             heldout_truth=float(T[si, sj]), heldout_scal=mh['scal'], heldout_deb=mh['deb']))
        save('e1d', rows); print('e1d', L, flush=True)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t0 = time.time(); globals()[name](); print(name, 'done', round(time.time() - t0), 's', flush=True)
