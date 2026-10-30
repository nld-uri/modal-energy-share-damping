"""Where the estimand stops being recoverable: a dissipative nonlinearity.

Paper 3, boundary case.  Everything else in this paper concerns a CONSERVATIVE
nonlinearity (a grounded cubic spring).  There the modal damping coefficient c_m
is untouched by the nonlinear force, total energy decays at the energy-weighted
mean modal damping, and the single-mode estimate is biased only because energy
is redistributed between modes.

This script replaces the cubic by a hysteretic (Bouc-Wen) element at the same
station and shows that the framework's central premise fails -- not its algebra,
its interpretation.  A single additional system suffices because the boundary is
qualitative: the nonlinearity now does net work over a cycle.

WHAT CHANGES

  conservative (cubic)        gamma_tot / c_true = 0.999 at every amplitude,
                              so fitting total energy recovers the estimand;
                              single-mode estimates are biased by share drift.

  dissipative (Bouc-Wen)      gamma_tot / c_true rises to about 6 and is
                              strongly amplitude-dependent; total-energy decay
                              is NOT the linear modal damping.  Moreover the
                              single-mode, subset and total estimates now agree
                              with one another, because the dominant term is a
                              real dissipation common to all modes rather than a
                              redistribution between them.

So for a dissipative nonlinearity the identified damping is not biased -- it is
measuring a different and physically meaningful quantity, the amplitude-dependent
effective damping.  The estimand of Section 2 (a velocity coefficient the
nonlinear force leaves alone) no longer exists.

MODELLING NOTE.  At small amplitude a Bouc-Wen element is a linear spring of
stiffness kh, which changes the mode shapes.  The reference structure must
therefore be the chain PLUS the stuck element stiffness, and the residual force
driving the modal equations is the element's departure from that spring.  Tracked
on that basis the elastic limit recovers gamma_tot/c -> 1 and a clean mode
(energy share 0.997), which is the control this comparison needs.

Usage:  python friction_boundary.py     (writes friction_boundary_results.json)
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("models", "identification", "experiments"):
    sys.path.insert(0, os.path.join(HERE, "..", _sub))
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402

KH, ALPHA, UY = 1.0, 0.1, 0.002
AMPS = (1e-6, 1e-5, 3e-5, 1e-4, 1e-3, 5e-3, 2e-2, 6e-2)
T_SETTLE, SPAN, DT, RTOL, EDGE = 4000.0, 400.0, 0.05, 1e-9, 20


def system(kh=KH, alpha=ALPHA, uy=UY, n=bm.N, j0=bm.J0, zeta1=bm.ZETA1):
    L = ln.Lattice(n=n, j0=j0, k3=0.0, zeta1=zeta1)
    Kmod = np.diag(L.wn ** 2) + kh * np.outer(L.psi, L.psi)
    lam, V = np.linalg.eigh(Kmod)
    V = V * np.sign(V[0])       # canonicalise signs; see chain_modes
    return dict(n=n, wn=np.sqrt(lam), psit=V.T @ L.psi, c=float(L.ci[0]),
                kh=kh, alpha=alpha, uy=uy, beta=0.5 / uy, gamma=0.5 / uy)


def rhs(S, A=0.0, Om=0.0):
    n, W2, c, pt = S["n"], S["wn"] ** 2, S["c"], S["psit"]
    kh, al, be, ga = S["kh"], S["alpha"], S["beta"], S["gamma"]

    def f(t, y):
        p, v, z = y[:n], y[n:2 * n], y[2 * n]
        w, wd = pt @ p, pt @ v
        resid = (al * kh * w + (1 - al) * kh * z) - kh * w
        zd = wd - be * abs(wd) * z - ga * wd * abs(z)
        acc = -c * v - W2 * p - pt * resid + (pt * (A * np.cos(Om * t)) if A else 0.0)
        return np.concatenate([v, acc, [zd]])
    return f


def one_amplitude(A, mode=bm.MODE):
    S = system()
    s = solve_ivp(rhs(S, A, S["wn"][mode]), (0.0, T_SETTLE), np.zeros(2 * S["n"] + 1),
                  method="DOP853", rtol=RTOL, atol=RTOL * 1e-2)
    tt = np.arange(0.0, SPAN, DT)
    d = solve_ivp(rhs(S), (0.0, SPAN), s.y[:, -1], method="DOP853",
                  rtol=RTOL, atol=RTOL * 1e-2, t_eval=tt)
    pp, vv, z = d.y[:S["n"]].T, d.y[S["n"]:2 * S["n"]].T, d.y[2 * S["n"]]
    Ei = 0.5 * (vv ** 2 + (S["wn"] ** 2)[None, :] * pp ** 2)
    Etot = Ei.sum(axis=1)
    k = len(tt) // EDGE
    sl = slice(k, len(tt) - k)
    sub = [i for i in range(mode - 2, mode + 3) if 0 <= i < S["n"]]
    env = np.abs(hilbert(pp[:, mode]))
    w = pp @ S["psit"]
    return dict(
        A=float(A), slip_ratio=float(np.abs(z).max() / S["uy"]),
        w_rms=float(np.sqrt(np.mean(w ** 2))),
        e_frac=float(Ei[:, mode].mean() / Etot.mean()),
        gamma_tot_over_c=float(-np.polyfit(tt[sl], np.log(Etot[sl]), 1)[0] / S["c"]),
        gamma_sub_over_c=float(-np.polyfit(
            tt[sl], np.log(Ei[:, sub].sum(axis=1)[sl]), 1)[0] / S["c"]),
        r_env_over_c=float(-2 * np.polyfit(tt[sl], np.log(env[sl]), 1)[0] / S["c"]))


def main():
    recs = [one_amplitude(A) for A in AMPS]
    g = np.array([r["gamma_tot_over_c"] for r in recs])
    e = np.array([r["r_env_over_c"] for r in recs])
    res = dict(
        config=dict(N=bm.N, j0=bm.J0, mode_1based=bm.MODE + 1, zeta1=bm.ZETA1,
                    kh=KH, alpha=ALPHA, uy=UY, t_settle=T_SETTLE, span=SPAN,
                    dt=DT, rtol=RTOL, edge_trim=EDGE,
                    basis="chain + stuck element stiffness"),
        records=recs,
        summary=dict(
            gamma_tot_min=float(g.min()), gamma_tot_max=float(g.max()),
            single_vs_total_mean_abs_diff=float(np.mean(np.abs(e - g))),
            single_vs_total_worst=float(np.max(np.abs(e - g))),
            note=("for the conservative cubic gamma_tot/c_true is 0.999 at every "
                  "amplitude and the single-mode estimate departs from it by up "
                  "to 0.54; here gamma_tot/c_true is amplitude-dependent and the "
                  "single-mode estimate tracks it")))
    out = os.path.join(HERE, "..", "..", "checkpoints",
                       "friction_boundary_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}")
    for r in recs:
        print(f"  A={r['A']:.0e}  slip {r['slip_ratio']:.3f}  "
              f"E_m {r['e_frac']:.4f}  gamma_tot/c {r['gamma_tot_over_c']:.4f}  "
              f"envelope {r['r_env_over_c']:.4f}")


if __name__ == "__main__":
    main()
