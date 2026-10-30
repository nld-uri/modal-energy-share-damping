"""Protocols track: phase-resonance (force-appropriation) forced orbits of the lattice.

Phase resonance of mode m: the station force is A cos(Omega t) and the FIRST
HARMONIC of the tracked modal coordinate a_m is in quadrature with it,
    g = (2/T) int_0^T a_m(t) cos(Omega t) dt = 0 .
At fixed Omega the unknowns are the initial state y0 (2n) and the force
amplitude A; the equations are the period-map closure phi_T(y0;A) - y0 = 0 and
g = 0. Newton uses the exact variational sensitivities (monodromy M = dphi/dy0,
s = dphi/dA, and the quadrature-integral sensitivities), integrated together.
Sweeping Omega upward traces the damped backbone (hardening).

Scaling: a -> s b, k3 -> k3 s^2, A -> A/s is an exact symmetry, so k3 = 1 here
without loss; kappa = k3 * max|psi.a|^2 over the orbit is the invariant measure
and mu = k3 A^2 the invariant forcing measure.
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("models", "identification", "experiments"):
    sys.path.insert(0, os.path.join(HERE, "..", _sub))
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402
import onset_mechanism as om  # noqa: E402

RT = dict(method="DOP853", rtol=1e-12, atol=1e-14)


def _aug_rhs(L, A, Om, m):
    n, n2 = L.n, 2 * L.n
    W2, C, PSI, k3 = L.wn ** 2, L.ci, L.psi, L.k3

    def rhs(t, z):
        y = z[:n2]
        M = z[n2:n2 + n2 * n2].reshape(n2, n2)
        s = z[n2 + n2 * n2:n2 + n2 * n2 + n2]
        a, v = y[:n], y[n:]
        w = PSI @ a
        co = np.cos(Om * t)
        dK = 3 * k3 * w ** 2
        fy = np.concatenate([v, -C * v - W2 * a - PSI * (k3 * w ** 3) + PSI * (A * co)])

        def Jx(X):          # tangent map applied to the columns of X (2n x k)
            Xa, Xv = X[:n], X[n:]
            return np.vstack([Xv, -W2[:, None] * Xa - dK * np.outer(PSI, PSI @ Xa)
                              - C[:, None] * Xv])
        dM = Jx(M)
        ds = Jx(s[:, None])[:, 0] + np.concatenate([np.zeros(n), PSI * co])
        dg = a[m] * co
        dgM = M[m] * co            # row m of the displacement block
        dgs = s[m] * co
        return np.concatenate([fy, dM.ravel(), ds, [dg], dgM, [dgs]])
    return rhs


def period_map(L, A, Om, y0, m):
    n2 = 2 * L.n
    T = 2 * np.pi / Om
    z0 = np.concatenate([y0, np.eye(n2).ravel(), np.zeros(n2), [0.0], np.zeros(n2), [0.0]])
    sol = solve_ivp(_aug_rhs(L, A, Om, m), (0.0, T), z0, **RT)
    z = sol.y[:, -1]
    o = n2
    yT = z[:o]; M = z[o:o + n2 * n2].reshape(n2, n2); o += n2 * n2
    s = z[o:o + n2]; o += n2
    g = z[o] * 2 / T; o += 1
    gM = z[o:o + n2] * 2 / T; o += n2
    gs = z[o] * 2 / T
    return yT, M, s, g, gM, gs


def shoot_phase_resonance(L, Om, y0, A0, m=bm.MODE, iters=25, tol=1e-12):
    """Newton on (y0, A) for the phase-resonant orbit at forcing frequency Om."""
    n2 = 2 * L.n
    y, A = np.array(y0, float), float(A0)
    hist = []
    for it in range(iters):
        yT, M, s, g, gM, gs = period_map(L, A, Om, y, m)
        r = yT - y
        res = float(np.linalg.norm(r) / max(np.linalg.norm(y), 1e-300))
        qres = float(abs(g) / max(np.sqrt(np.sum(y[[m, m + L.n]] ** 2)), 1e-300))
        hist.append((res, qres))
        if res < tol and qres < tol:
            break
        Jac = np.zeros((n2 + 1, n2 + 1))
        Jac[:n2, :n2] = M - np.eye(n2)
        Jac[:n2, n2] = s
        Jac[n2, :n2] = gM
        Jac[n2, n2] = gs
        d = np.linalg.solve(Jac, -np.concatenate([r, [g]]))
        # damp very large steps (stay on the non-trivial branch)
        sc = 1.0
        nd = np.linalg.norm(d[:n2]) / max(np.linalg.norm(y), 1e-300)
        if nd > 0.5:
            sc = 0.5 / nd
        y = y + sc * d[:n2]
        A = A + sc * d[n2]
    yT, M, s, g, gM, gs = period_map(L, A, Om, y, m)
    res = float(np.linalg.norm(yT - y) / max(np.linalg.norm(y), 1e-300))
    qres = float(abs(g) / max(np.sqrt(np.sum(y[[m, m + L.n]] ** 2)), 1e-300))
    return dict(y0=y, A=A, residual=res, quad_residual=qres, M=M, newton_iters=len(hist))


def orbit_metrics(L, A, Om, y0, m=bm.MODE, npts=1601):
    """kappa, first-harmonic phase of a_m relative to the force, energy shares."""
    T = 2 * np.pi / Om
    tt = np.linspace(0.0, T, npts)
    sol = solve_ivp(L.rhs_forced(A, Om), (0.0, T), y0, t_eval=tt, **RT)
    a, v = sol.y[:L.n].T, sol.y[L.n:].T
    w = a @ L.psi
    E = np.mean(0.5 * (v ** 2 + (L.wn ** 2)[None, :] * a ** 2), axis=0)
    # first harmonic of a_m (trapezoid on closed periodic grid -> drop last pt)
    ph = np.exp(-1j * Om * tt[:-1])
    H = 2 * np.mean(a[:-1, m] * ph)
    lag = float(np.angle(H / (L.psi[m] * A)))        # -pi/2 at phase resonance
    return dict(kappa=float(L.k3 * np.max(np.abs(w)) ** 2), w_peak=float(np.max(np.abs(w))),
                phase_am_minus_force_deg=float(np.degrees(lag)),
                share_m=float(E[m] / E.sum()),
                harmonic1_am=float(abs(H)))


def floquet(M, wn):
    ev, evec = np.linalg.eig(M)
    c = om.unstable_mode_content(ev, evec, wn)
    return float(np.max(np.abs(ev))), c, ev


def guess_state(L, Om, kappa, m=bm.MODE):
    """Single-mode quadrature guess: a_m = X sin(Om t), A = c X Om / psi_m."""
    X = np.sqrt(kappa / L.k3) / abs(L.psi[m])
    y = np.zeros(2 * L.n)
    y[L.n + m] = X * Om
    A = L.ci[m] * X * Om / L.psi[m]
    return y, A


def averaged_pairs(L, Om, kappa, exclude_mode=None):
    """Averaged combination-resonance margin for every pair (j,k) at (Om, kappa).

    Same formulas as m13.pair_thresholds, but Omega is an argument (the
    protocol's forcing frequency) instead of omega_lin. Returns the dict
    {(j,k) 1-based: margin = Gamma^2 - sigma^2/4 - c^2/4} and c.
    """
    c = L.ci[0]
    w2, Ph = np.linalg.eigh(np.diag(L.wn ** 2) + 1.5 * kappa * np.outer(L.psi, L.psi))
    wt = np.sqrt(w2); pt = Ph.T @ L.psi
    out = {}
    for j in range(L.n):
        for k in range(j, L.n):
            if exclude_mode is not None and exclude_mode in (j, k):
                continue
            sig = 2 * Om - wt[j] - wt[k]
            G = 0.375 * kappa * abs(pt[j] * pt[k]) / np.sqrt(wt[j] * wt[k])
            out[(j + 1, k + 1)] = G ** 2 - sig ** 2 / 4 - c ** 2 / 4
    return out
