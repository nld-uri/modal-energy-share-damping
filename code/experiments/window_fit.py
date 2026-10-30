"""Does a longer record really hurt? No -- the estimator did.

Section 9 originally reported that the identification error is non-monotone in
analysis-window length and that the best window shortens as the signal-to-noise
ratio falls. Both statements are properties of the ESTIMATOR, not of the
measurement problem. Additive noise puts a constant pedestal under a measured
energy (energy is quadratic, so the noise contributes in the mean rather than
averaging out), and an unweighted straight-line fit to log E is dragged toward
zero slope as the signal decays under that pedestal.

Fitting the pedestal as a third parameter,

    E(t) = E0 exp(-gamma t) + b ,

removes the effect: at 20 dB the corrected estimate is good to 0.5 percent out
to 8000 time units, where the naive fit has lost more than half the value.

This script stores the comparison AND the underlying log-energy traces, so the
fit can be plotted against the data it was fitted to. That is the check that
would have caught the original error without any argument.

Usage:  python window_fit.py        (writes window_fit_results.json)
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("models", "identification", "experiments"):
    sys.path.insert(0, os.path.join(HERE, "..", _sub))
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402

MU, A_FIX = 1.0e-3, 2e-2
SPANS = (400.0, 1200.0, 2400.0, 4000.0, 6000.0, 8000.0)
SNRS_DB = (np.inf, 40.0, 30.0, 20.0)
N_REAL = 3
EDGE = 20
SEED0 = 20260928


def _noisy(xx, xd, snr, rng):
    if not np.isfinite(snr):
        return xx, xd
    sx = np.sqrt(np.mean(xx ** 2)) * 10 ** (-snr / 20)
    sv = np.sqrt(np.mean(xd ** 2)) * 10 ** (-snr / 20)
    return xx + rng.normal(0, sx, xx.shape), xd + rng.normal(0, sv, xd.shape)


def _naive(t, E):
    """Unweighted straight line through log E -- the original estimator."""
    return -np.polyfit(t, np.log(np.maximum(E, 1e-300)), 1)[0]


def _floor_aware(t, E, g0):
    """Three-parameter fit E0 exp(-g t) + b, b the noise pedestal."""
    def model(tt, lE0, g, lb):
        return np.log(np.exp(lE0 - g * (tt - tt[0])) + np.exp(lb))
    p, _ = curve_fit(model, t, np.log(np.maximum(E, 1e-300)),
                     p0=[np.log(E[0]), g0, np.log(max(E[-1] * 0.5, 1e-300))],
                     maxfev=20000)
    return p[1], float(np.exp(p[2]))


def main():
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=MU / A_FIX ** 2, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A_FIX, bm.MODE)
    sub = [i for i in range(bm.MODE - 2, bm.MODE + 3) if 0 <= i < L.n]
    tt = np.arange(0.0, max(SPANS), bm.DT)
    qq, qd, _ = L.decay(y0, tt, rtol=bm.RTOL)
    xx_f, xd_f = qq @ L.Phi.T, qd @ L.Phi.T

    rows, traces = [], []
    for span in SPANS:
        k = int(span / bm.DT)
        for snr in SNRS_DB:
            for rep in range(1 if not np.isfinite(snr) else N_REAL):
                rng = np.random.default_rng(
                    SEED0 + 31 * rep + int(0 if not np.isfinite(snr) else snr))
                xn, vn = _noisy(xx_f[:k], xd_f[:k], snr, rng)
                qh, qdh = xn @ L.Phi, vn @ L.Phi
                Ei = 0.5 * (qdh ** 2 + (L.wn ** 2)[None, :] * qh ** 2)
                Es = Ei[:, sub].sum(axis=1)
                env = np.abs(hilbert(qh[:, bm.MODE]))
                j = len(tt[:k]) // EDGE
                sl = slice(j, k - j)
                t, E = tt[:k][sl], Es[sl]
                g_naive = _naive(t, E)
                g_floor, b = _floor_aware(t, E, L.ci[bm.MODE])
                # the envelope reference has the same pedestal problem, in
                # amplitude rather than energy, so it is measured both ways too
                e_naive = -2 * _naive(t, env[sl] ** 2) / 2
                e_floor = _floor_aware(t, env[sl] ** 2, L.ci[bm.MODE])[0]
                rows.append(dict(
                    span=float(span), rep=rep,
                    snr_db=(None if not np.isfinite(snr) else float(snr)),
                    gamma_naive=float(g_naive / L.ci[bm.MODE]),
                    gamma_floor=float(g_floor / L.ci[bm.MODE]),
                    env_naive=float(e_naive / L.ci[bm.MODE]),
                    env_floor=float(e_floor / L.ci[bm.MODE]),
                    pedestal_over_E0=float(b / E[0])))
                if rep == 0 and span == max(SPANS):
                    d = max(1, len(t) // 600)          # decimate for storage
                    traces.append(dict(
                        snr_db=(None if not np.isfinite(snr) else float(snr)),
                        t=[float(v) for v in t[::d]],
                        logE=[float(v) for v in np.log(E[::d])],
                        fit_naive=[float(np.log(E[0]) - g_naive * (v - t[0]))
                                   for v in t[::d]],
                        fit_floor=[float(np.log(np.exp(np.log(E[0] - b)
                                                       - g_floor * (v - t[0])) + b))
                                   for v in t[::d]],
                        pedestal=float(b)))

    res = dict(config=dict(mu=MU, A=A_FIX, N=bm.N, j0=bm.J0,
                           mode_1based=bm.MODE + 1, zeta1=bm.ZETA1,
                           spans=list(SPANS), n_realisations=N_REAL,
                           edge_trim=EDGE, subset="carrier +/- 2"),
               records=rows, traces=traces)
    agg = {}
    for r in rows:
        key = f"{r['snr_db']}|{r['span']:.0f}"
        agg.setdefault(key, {"naive": [], "floor": []})
        agg[key]["naive"].append(abs(r["gamma_naive"] - 1))
        agg[key]["floor"].append(abs(r["gamma_floor"] - 1))
    res["summary"] = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()}
                      for k, v in agg.items()}
    res["worst"] = dict(
        naive=float(max(v["naive"] for v in res["summary"].values())),
        floor=float(max(v["floor"] for v in res["summary"].values())))
    out = os.path.join(HERE, "..", "..", "checkpoints", "window_fit_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}")
    print(f"worst mean |error|: naive {res['worst']['naive']:.4f}, "
          f"floor-aware {res['worst']['floor']:.4f}")


if __name__ == "__main__":
    main()
