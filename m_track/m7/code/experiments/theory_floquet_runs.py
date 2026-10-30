"""Newton-shot Floquet evaluations used to validate the averaged theory.

forced_floquet      : forced orbit at a given Omega (default omega_lin), via
                      onset_mechanism.shoot_orbit (unchanged reference code).
phase_res_floquet   : forced orbit at PHASE RESONANCE. Unknowns (y0, Omega);
                      equations phi_T(y0) - y0 = 0 and a_m(0) = 0 (tracked-mode
                      displacement in quadrature with the force A cos(Omega t),
                      exact for the fundamental, approximate by the tracked
                      mode's own 3rd-harmonic content).
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
for _s in ("models", "identification", "experiments"):
    sys.path.insert(0, os.path.join(HERE, "..", _s))
import lattice_nl as ln  # noqa: E402
import onset_mechanism as om  # noqa: E402


def _orbit_summary(L, A, Om, y0, M):
    ev, evec = np.linalg.eig(M)
    T = 2 * np.pi / Om
    sp = solve_ivp(L.rhs_forced(A, Om), (0.0, T), y0, method="DOP853",
                   rtol=1e-11, atol=1e-14, t_eval=np.linspace(0.0, T, 801))
    wq = sp.y[:L.n].T @ L.psi
    wpk = float(np.abs(wq).max())
    c = om.unstable_mode_content(ev, evec, L.wn)
    return dict(Omega=float(Om), k3=float(L.k3), A=float(A), w_peak=wpk,
                kappa=float(L.k3 * wpk ** 2), max_multiplier=float(np.abs(ev).max()),
                unstable=bool(np.abs(ev).max() > 1.0),
                content=[(d['mode'], round(d['share'], 4)) for d in c['content'][:3]])


def forced_floquet(mu, zeta1, j0=1, mode=6, n=24, A=2e-2, Omega=None, t_seed=300.0):
    L = ln.Lattice(n=n, j0=j0, k3=mu / A ** 2, zeta1=zeta1)
    Om = L.wn[mode] if Omega is None else Omega
    s = solve_ivp(L.rhs_forced(A, Om), (0.0, t_seed), np.zeros(2 * L.n),
                  method="DOP853", rtol=1e-10, atol=1e-13)
    y0, res, M = om.shoot_orbit(L, A, Om, s.y[:, -1])
    out = _orbit_summary(L, A, Om, y0, M)
    out.update(mu=float(mu), zeta1=float(zeta1), j0=j0, mode_1based=mode + 1,
               return_residual=res)
    return out


def _flow_var(L, A, Om, y):
    n2 = 2 * L.n
    f = L.rhs_forced(A, Om)

    def rhs(t, z):
        yy, M = z[:n2], z[n2:].reshape(n2, n2)
        dK = 3 * L.k3 * (L.psi @ yy[:L.n]) ** 2
        J = np.zeros((n2, n2))
        J[:L.n, L.n:] = np.eye(L.n)
        J[L.n:, :L.n] = -(np.diag(L.wn ** 2) + dK * np.outer(L.psi, L.psi))
        J[L.n:, L.n:] = -np.diag(L.ci)
        return np.concatenate([f(t, yy), (J @ M).ravel()])
    s = solve_ivp(rhs, (0.0, 2 * np.pi / Om), np.concatenate([y, np.eye(n2).ravel()]),
                  method="DOP853", rtol=1e-12, atol=1e-14)
    return s.y[:n2, -1], s.y[n2:, -1].reshape(n2, n2)


def _flow(L, A, Om, y):
    s = solve_ivp(L.rhs_forced(A, Om), (0.0, 2 * np.pi / Om), y,
                  method="DOP853", rtol=1e-12, atol=1e-14)
    return s.y[:, -1]


def phase_res_floquet(A, zeta1, y_guess, Om_guess, j0=1, mode=6, n=24, k3=1.0,
                      iters=20, tol=1e-11):
    """Newton on (y0, Omega) with phase condition a_m(0) = 0."""
    L = ln.Lattice(n=n, j0=j0, k3=k3, zeta1=zeta1)
    n2 = 2 * L.n
    y, Om = y_guess.copy(), float(Om_guess)
    res = np.inf
    for _ in range(iters):
        yT, M = _flow_var(L, A, Om, y)
        r = np.concatenate([yT - y, [y[mode]]])
        res = float(np.linalg.norm(r[:n2]) / max(np.linalg.norm(y), 1e-300))
        if res < tol and abs(y[mode]) < tol:
            break
        h = 1e-7 * Om
        dO = (_flow(L, A, Om + h, y) - _flow(L, A, Om - h, y)) / (2 * h)
        J = np.zeros((n2 + 1, n2 + 1))
        J[:n2, :n2] = M - np.eye(n2)
        J[:n2, n2] = dO
        J[n2, mode] = 1.0
        dx = np.linalg.solve(J, -r)
        # damp very large steps (first iterations from a crude guess)
        sc = min(1.0, 0.2 * np.linalg.norm(y) / max(np.linalg.norm(dx[:n2]), 1e-300))
        y = y + sc * dx[:n2]
        Om = Om + sc * dx[n2]
    yT, M = _flow_var(L, A, Om, y)
    res = float(np.linalg.norm(yT - y) / np.linalg.norm(y))
    out = _orbit_summary(L, A, Om, y, M)
    out.update(zeta1=float(zeta1), j0=j0, mode_1based=mode + 1, return_residual=res,
               phase_residual=float(abs(y[mode])), v_m0=float(y[L.n + mode]))
    return out, y


def phase_res_sweep(zeta1, kappa_targets, j0=1, mode=6, n=24, k3=1.0):
    """Continuation in forcing amplitude along the phase-resonance curve."""
    L = ln.Lattice(n=n, j0=j0, k3=k3, zeta1=zeta1)
    pm, wm, c = L.psi[mode], L.wn[mode], L.ci[mode]
    rows, y_prev, Om_prev, a_prev = [], None, None, None
    for kt in kappa_targets:
        a = np.sqrt(kt / (k3 * pm ** 2))
        Om = np.sqrt(wm ** 2 + 0.75 * kt * pm ** 2)
        A = c * a * Om / pm                   # modal power balance
        if y_prev is None:
            y = np.zeros(2 * n); y[n + mode] = a * Om
        else:
            y = y_prev * (a / a_prev); Om = Om_prev
        r, y = phase_res_floquet(A, zeta1, y, Om, j0, mode, n, k3)
        r['kappa_target'] = float(kt)
        rows.append(r)
        if r['return_residual'] < 1e-8:
            y_prev, Om_prev, a_prev = y, r['Omega'], a
        print(f"  z={zeta1} kt={kt:.4f} kap={r['kappa']:.5f} Om={r['Omega']:.6f} "
              f"|lam|={r['max_multiplier']:.6f} res={r['return_residual']:.1e} "
              f"{r['content'][:2]}", flush=True)
    return rows
