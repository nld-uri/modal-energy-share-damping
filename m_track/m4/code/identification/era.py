"""Eigensystem realization algorithm (Juang & Pappa 1985) for free-decay data.

The independent check on the decomposition in paper 3. It shares none of the
GCVD machinery: it uses displacement samples only (no velocity, acceleration or
mode shapes), stacks them in a block Hankel matrix, truncates its SVD and reads
poles from the balanced realization.

This file reproduces, line for line, the implementation that produced the
ERA numbers first quoted in the manuscript, which had been run interactively and
not saved with the code (found 2026-10-02).
"""
import numpy as np


def era(y, dt, n_modes=8, r=None, s=None):
    """ERA poles of a free-decay record.

    y        (N, p) displacement samples, one column per sensor
    dt       sample interval of y
    n_modes  realization order / 2 (number of conjugate pole pairs kept)
    r, s     block rows and columns of the Hankel matrix (default min(400, N/3))

    Returns (wn, zeta), sorted by wn, for poles with positive imaginary part.
    """
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    Nn, m = y.shape
    r = min(400, Nn // 3) if r is None else r
    s = min(400, Nn // 3) if s is None else s
    H0 = np.empty((r * m, s))
    H1 = np.empty((r * m, s))
    for i in range(r):
        H0[i * m:(i + 1) * m, :] = y[i:i + s].T
        H1[i * m:(i + 1) * m, :] = y[i + 1:i + 1 + s].T
    U, S, Vt = np.linalg.svd(H0, full_matrices=False)
    k = 2 * n_modes
    Uk, Sk, Vk = U[:, :k], S[:k], Vt[:k].T
    A = np.diag(Sk ** -0.5) @ Uk.T @ H1 @ Vk @ np.diag(Sk ** -0.5)
    sc = np.log(np.linalg.eigvals(A).astype(complex)) / dt
    keep = sc.imag > 1e-9
    wn = np.abs(sc[keep])
    zeta = -sc[keep].real / np.maximum(wn, 1e-30)
    o = np.argsort(wn)
    return wn[o], zeta[o]


def era_damping(y, dt, w_target, **kw):
    """(c, wn) of the ERA pole nearest w_target, c = 2 zeta wn."""
    wn, zeta = era(y, dt, **kw)
    i = int(np.argmin(np.abs(wn - w_target)))
    return float(2 * zeta[i] * wn[i]), float(wn[i])


def era_modes(y, dt, n_modes=8, r=None, s=None):
    """ERA poles with each pole's contribution to the record.

    Contribution of pole i is ||C psi_i|| * |(Psi^-1 x0)_i| summed over the
    initial-condition columns, the modal singular value of the realization:
    how much of the measured response that pole carries. It is computed from
    the realization alone (no mode shapes).

    Returns dict(wn, zeta, contrib) sorted by wn, positive-frequency poles only.
    """
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    Nn, m = y.shape
    r = min(400, Nn // 3) if r is None else r
    s = min(400, Nn // 3) if s is None else s
    H0 = np.empty((r * m, s))
    H1 = np.empty((r * m, s))
    for i in range(r):
        H0[i * m:(i + 1) * m, :] = y[i:i + s].T
        H1[i * m:(i + 1) * m, :] = y[i + 1:i + 1 + s].T
    U, S, Vt = np.linalg.svd(H0, full_matrices=False)
    k = 2 * n_modes
    Uk, Sk, Vk = U[:, :k], S[:k], Vt[:k].T
    Sh = np.diag(Sk ** 0.5)
    A = np.diag(Sk ** -0.5) @ Uk.T @ H1 @ Vk @ np.diag(Sk ** -0.5)
    C = (Uk @ Sh)[:m]                       # output matrix
    X0 = Sh @ Vk.T                          # realized states over the s columns
    lam, Psi = np.linalg.eig(A)
    Z = np.linalg.solve(Psi, X0)            # modal states
    contrib = np.linalg.norm(C @ Psi, axis=0) * np.linalg.norm(Z, axis=1)
    sc = np.log(lam.astype(complex)) / dt
    keep = sc.imag > 1e-9
    wn = np.abs(sc[keep])
    zeta = -sc[keep].real / np.maximum(wn, 1e-30)
    o = np.argsort(wn)
    return dict(wn=wn[o], zeta=zeta[o], contrib=contrib[keep][o])


def era_damping_dominant(y, dt, w_target, band=0.05, **kw):
    """(c, wn, share) of the largest-contribution ERA pole within +-band of w_target.

    share is that pole's contribution over the summed contribution of all poles
    in the band, i.e. how unambiguous the selection was.
    """
    md = era_modes(y, dt, **kw)
    sel = np.flatnonzero(np.abs(md["wn"] / w_target - 1) <= band)
    if sel.size == 0:
        return float("nan"), float("nan"), 0.0
    i = sel[int(np.argmax(md["contrib"][sel]))]
    share = float(md["contrib"][i] / md["contrib"][sel].sum())
    return float(2 * md["zeta"][i] * md["wn"][i]), float(md["wn"][i]), share
