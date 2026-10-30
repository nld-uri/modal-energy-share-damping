"""Averaged (first-order multiple-scales) combination-resonance theory for the
sideband instability of a periodic carrier in the cubic lattice.

Model (modal coordinates, M = I):
    a_i'' + c_i a_i' + w_i^2 a_i + psi_i k3 (psi.a)^3 = psi_i A cos(Omega t)

Carrier: w = psi.a ~ W cos(Omega t), kappa := k3 W^2 (W = peak station
displacement). Linearised about the carrier, the tangent stiffness is
    diag(w^2) + 3 k3 w(t)^2 psi psi^T
  = diag(w^2) + 1.5 kappa psi psi^T  +  1.5 kappa cos(2 Omega t) psi psi^T.
The time-mean part is diagonalised exactly ("mean shift"): frequencies wt_i,
projections pt_i = (Phit^T psi)_i, damping mu_i = (Phit^T C Phit)_ii / 2.
The 2 Omega modulation drives sum-type combination resonance of pair (j,k)
(j = k: principal parametric) with
    Gamma_jk = (3/8) kappa |pt_j pt_k| / sqrt(wt_j wt_k),
    sigma_jk = 2 Omega - wt_j - wt_k.
Slow-flow exponent: largest real part of the eigenvalues of
    [[-mu_j - i sigma/2,  i Gamma], [-i Gamma, -mu_k + i sigma/2]],
i.e. for mu_j = mu_k = c/2: lambda = -c/2 + sqrt(Gamma^2 - sigma^2/4).
Unstable iff Gamma^2 > mu_j mu_k (1 + sigma^2/(mu_j+mu_k)^2)  (equal damping:
Gamma^2 > sigma^2/4 + c^2/4). Predicted Floquet multiplier |lam| =
exp(lambda * 2 pi / Omega) (other directions sit at exp(-mu T)).

Pairs that contain the carrier's own (mean-shifted) mode are excluded from
the sideband set by default: (m,m) is the stability of the forced response
curve itself, not a sideband instability.

Mode labels returned are 1-based (matching onset_mechanism 'content').
"""
import numpy as np
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# mean-shifted linear problem
# --------------------------------------------------------------------------
def mean_shift(L, kappa):
    """Mean-shifted frequencies, shapes, psi-projections and modal damping.

    Tilde modes are labelled by the original mode with which they overlap most
    (falls back to ascending order if that labelling is not one-to-one).
    """
    K = np.diag(L.wn ** 2) + 1.5 * kappa * np.outer(L.psi, L.psi)
    w2, Pt = np.linalg.eigh(K)
    lab = np.argmax(np.abs(Pt), axis=0)
    if len(set(lab.tolist())) == len(lab):
        order = np.argsort(lab)
        w2, Pt = w2[order], Pt[:, order]
    wt = np.sqrt(w2)
    pt = Pt.T @ L.psi
    mu = 0.5 * np.einsum('ij,i,ij->j', Pt, L.ci, Pt)
    return dict(wt=wt, Pt=Pt, pt=pt, mu=mu)


def pair_exponent(G, sig, mj, mk):
    """Slow-flow growth exponent (largest real eigenvalue part)."""
    J = np.array([[-mj - 0.5j * sig, 1j * G], [-1j * G, -mk + 0.5j * sig]])
    return float(np.max(np.linalg.eigvals(J).real))


def pair_margin(G, sig, mj, mk):
    """Gamma^2 - threshold; > 0 means unstable (closed form, smooth in kappa)."""
    s = mj + mk
    return G ** 2 - mj * mk * (1.0 + (sig / s) ** 2) if s > 0 else G ** 2 - sig ** 2 / 4


def pair_table(L, kappa, Omega, carrier, include_carrier=False, raw=False):
    """Every pair (j <= k): Gamma, sigma, damping, margin, exponent, multiplier.

    raw=True evaluates frequencies/projections WITHOUT the mean shift (the
    paper's raw-detuning / raw-coupling argument), for comparison.
    """
    if raw:
        wt, pt, mu = L.wn.copy(), L.psi.copy(), 0.5 * L.ci
    else:
        ms = mean_shift(L, kappa)
        wt, pt, mu = ms['wt'], ms['pt'], ms['mu']
    T = 2 * np.pi / Omega
    rows = []
    for j in range(L.n):
        for k in range(j, L.n):
            if not include_carrier and carrier in (j, k):
                continue
            G = 0.375 * kappa * abs(pt[j] * pt[k]) / np.sqrt(wt[j] * wt[k])
            sig = 2 * Omega - wt[j] - wt[k]
            lam = pair_exponent(G, sig, mu[j], mu[k])
            rows.append(dict(pair=(j + 1, k + 1), Gamma=float(G), sigma=float(sig),
                             mu_j=float(mu[j]), mu_k=float(mu[k]),
                             margin=float(pair_margin(G, sig, mu[j], mu[k])),
                             exponent=lam, multiplier=float(np.exp(lam * T))))
    return rows


def unstable_set(L, kappa, Omega, carrier, **kw):
    return [r for r in pair_table(L, kappa, Omega, carrier, **kw) if r['margin'] > 0]


def predicted_max_multiplier(L, kappa, Omega, carrier):
    """max over sideband pairs and over uncoupled directions exp(-mu T)."""
    tab = pair_table(L, kappa, Omega, carrier)
    T = 2 * np.pi / Omega
    ms = mean_shift(L, kappa)
    passive = float(np.exp(-np.min(np.delete(ms['mu'], carrier)) * T))
    best = max(tab, key=lambda r: r['multiplier'])
    return max(best['multiplier'], passive), best


# --------------------------------------------------------------------------
# excitation-frequency laws
# --------------------------------------------------------------------------
def omega_backbone_hb(L, kappa, carrier):
    """Single-mode harmonic-balance backbone: w^2 = w_m^2 + (3/4) kappa psi_m^2."""
    return float(np.sqrt(L.wn[carrier] ** 2 + 0.75 * kappa * L.psi[carrier] ** 2))


def omega_backbone_table(nnm_records):
    """Interpolant omega_NNM(kappa) from nnm.json records (conservative NNM)."""
    k = np.array([r['kappa'] for r in nnm_records])
    w = np.array([r['omega'] for r in nnm_records])
    o = np.argsort(k)
    return lambda kap: float(np.interp(kap, k[o], w[o]))


def omega_law(L, carrier, law='linear', table=None):
    if law == 'linear':
        w0 = float(L.wn[carrier]); return lambda kap: w0
    if law == 'backbone_hb':
        return lambda kap: omega_backbone_hb(L, kap, carrier)
    if law == 'backbone_nnm':
        return table
    if callable(law):
        return law
    raise ValueError(law)


# --------------------------------------------------------------------------
# thresholds by root finding
# --------------------------------------------------------------------------
def pair_windows(L, carrier, pair, Omega_of_kappa, kmax=0.3, nscan=600, kmin=1e-6):
    """Unstable kappa-intervals of one pair, edges located by brentq.

    A geometric scan only BRACKETS sign changes of the smooth margin; every
    edge is then solved to xtol 1e-12 by Brent's method.
    """
    j, k = pair[0] - 1, pair[1] - 1

    def margin(kap):
        Om = Omega_of_kappa(kap)
        ms = mean_shift(L, kap)
        wt, pt, mu = ms['wt'], ms['pt'], ms['mu']
        G = 0.375 * kap * abs(pt[j] * pt[k]) / np.sqrt(wt[j] * wt[k])
        return pair_margin(G, 2 * Om - wt[j] - wt[k], mu[j], mu[k])

    ks = np.geomspace(kmin, kmax, nscan)
    m = np.array([margin(x) for x in ks])
    edges = [brentq(margin, ks[i], ks[i + 1], xtol=1e-12, rtol=1e-12)
             for i in range(len(ks) - 1) if m[i] * m[i + 1] < 0]
    wins, start = [], (ks[0] if m[0] > 0 else None)
    for e in edges:
        if start is None:
            start = e
        else:
            wins.append((float(start), float(e))); start = None
    if start is not None:
        wins.append((float(start), float('inf')))
    return wins


def all_windows(L, carrier, Omega_of_kappa, kmax=0.3, nscan=600, sigma_raw_max=None):
    """Windows of every sideband pair. sigma_raw_max (optional) skips pairs whose
    UNSHIFTED detuning at kappa=0 exceeds it (speed only; default: no skip)."""
    out = {}
    Om0 = Omega_of_kappa(0.0)
    for j in range(L.n):
        for k in range(j, L.n):
            if carrier in (j, k):
                continue
            if sigma_raw_max is not None and abs(2 * Om0 - L.wn[j] - L.wn[k]) > sigma_raw_max:
                continue
            w = pair_windows(L, carrier, (j + 1, k + 1), Omega_of_kappa, kmax, nscan)
            if w:
                out[(j + 1, k + 1)] = w
    return out


def first_threshold(L, carrier, Omega_of_kappa, kmax=0.3, nscan=600, sigma_raw_max=None):
    """kappa_c = lowest lower edge over all sideband pairs, with that pair."""
    W = all_windows(L, carrier, Omega_of_kappa, kmax, nscan, sigma_raw_max)
    if not W:
        return dict(kappa_c=None, pair=None, windows={})
    p, w = min(W.items(), key=lambda kv: kv[1][0][0])
    return dict(kappa_c=w[0][0], pair=p,
                windows={f"{a}-{b}": v for (a, b), v in W.items()})


# --------------------------------------------------------------------------
# kappa(mu) for forced response at fixed Omega (screening map only)
# --------------------------------------------------------------------------
def kappa_forced_hb(L, A, Omega):
    """Single-harmonic, all-mode harmonic balance of the forced response.

    W = g F,  F = A - (3/4) k3 |W|^2 W,  g = sum psi_i^2 / (w_i^2 - Om^2 + i c_i Om).
    With s = |F|^2:  |b|^2 s^3 + 2 Re(b) s^2 + s - A^2 = 0,  b = 0.75 k3 |g|^2 g.
    Returns (kappa = k3 |W|^2 on the smallest positive root, number of roots).
    """
    g = np.sum(L.psi ** 2 / (L.wn ** 2 - Omega ** 2 + 1j * L.ci * Omega))
    b = 0.75 * L.k3 * abs(g) ** 2 * g
    r = np.roots([abs(b) ** 2, 2 * b.real, 1.0, -A ** 2])
    s = np.sort([x.real for x in r if abs(x.imag) < 1e-9 * max(1, abs(x)) and x.real > 0])
    return float(L.k3 * abs(g) ** 2 * s[0]), len(s)


# --------------------------------------------------------------------------
# beyond isolated pairs: first-order multi-mode (cluster) slow flow
# --------------------------------------------------------------------------
def cluster_slow_flow(L, kappa, Omega, carrier, anchor, delta=0.06, mmax=3):
    """First-order averaged slow flow on a near-resonant LADDER cluster.

    Anchor mode j sets nu = wt_j. Every (mode i, integer m) with
    | |nu + 2 m Omega| - wt_i | < delta becomes a variable Y with frame
    frequency f = nu + 2 m Omega, a_i ~ Y e^{i f t} + c.c. Dropping Y'':
        2 i f Y' = -(wt_i^2 - f^2 + i c_i f) Y - (h/2) pt_i sum pt_k Y_k,
    the sum over variables whose f differs by exactly +-2 Omega; h = 1.5 kappa.
    Sum-type (j,k) and difference-type couplings, and chains of them, are all
    retained; for an isolated sum-type pair this reduces to pair_exponent.
    Returns (max real growth exponent, variable list [(mode_1based, m)]).
    """
    ms = mean_shift(L, kappa)
    wt, pt, mu = ms['wt'], ms['pt'], ms['mu']
    nu = wt[anchor]
    var = [(i, m) for i in range(L.n) if i != carrier for m in range(-mmax, mmax + 1)
           if abs(abs(nu + 2 * m * Omega) - wt[i]) < delta]
    nv = len(var)
    f = np.array([nu + 2 * m * Omega for _, m in var])
    J = np.zeros((nv, nv), complex)
    h = 1.5 * kappa
    for a, (i, m) in enumerate(var):
        J[a, a] = -(wt[i] ** 2 - f[a] ** 2 + 2j * mu[i] * f[a]) / (2j * f[a])
        for b, (k, mk) in enumerate(var):
            if abs(m - mk) == 1:
                J[a, b] += -(0.5 * h) * pt[i] * pt[k] / (2j * f[a])
    ev = np.linalg.eigvals(J)
    return float(np.max(ev.real)), [(i + 1, m) for i, m in var]


def cluster_growth(L, kappa, Omega, carrier, delta=0.06):
    """Max over anchors of the cluster slow-flow growth exponent."""
    best = (-np.inf, None)
    for j in range(L.n):
        if j == carrier:
            continue
        g, v = cluster_slow_flow(L, kappa, Omega, carrier, j, delta)
        if g > best[0]:
            best = (g, v)
    return best


def cluster_threshold(L, carrier, Omega_of_kappa, kmax=0.3, nscan=300, delta=0.06):
    """Lowest kappa with positive cluster growth (scan to bracket, brentq)."""
    g = lambda k: cluster_growth(L, k, Omega_of_kappa(k), carrier, delta)[0]
    ks = np.geomspace(1e-4, kmax, nscan)
    prev = g(ks[0])
    for a, b in zip(ks, ks[1:]):
        cur = g(b)
        if prev <= 0 < cur:
            kc = brentq(g, a, b, xtol=1e-10)
            return dict(kappa_c=float(kc),
                        cluster=cluster_growth(L, kc * 1.001, Omega_of_kappa(kc), carrier, delta)[1])
        prev = cur
    return dict(kappa_c=None, cluster=None)
