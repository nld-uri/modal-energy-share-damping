"""Does the recipe survive measurement noise and partial instrumentation?

Paper 3.  `recovery_recipe.py` establishes that the estimand is recovered by
fitting the decay of energy summed over the carrier and its m+/-2 sideband pair,
to 0.4 percent, where the single-mode estimate is wrong by up to 54 percent.
That result is computed in exact modal coordinates from a noise-free, fully
observed simulation.  Both idealisations are load-bearing, because the recipe
needs MODAL ENERGIES, which a test does not measure directly:

  - modal coordinates must be projected out of physical sensor signals, so the
    mode shapes and enough sensors are needed;
  - energy needs displacement AND velocity, so both are exposed to noise;
  - energy is quadratic, so additive noise biases it UPWARD rather than
    averaging out, and the bias grows as the record decays into the noise floor.

The last point is the one to watch, and it is a property of the ESTIMATOR
before it is a property of the measurement.  A log-linear fit to a measured
energy reads the noise pedestal in the tail as a slower decay, so the error
grows with window length and the estimator appears to have an optimum.
Estimating the pedestal as a third parameter, E(t) = E0 exp(-gamma t) + b,
removes that bias: what is left is a shallow, weakly SNR-dependent optimum in
window length rather than a collapse.  Both estimates are stored per record --
`gamma_sub_over_c` (floor-aware) and `gamma_sub_naive_over_c` (log-linear) --
so the size of the artifact can be read off the file.

TWO SWEEPS

  noise     independent white noise on physical displacement and velocity at
            every mass, SNR from 60 dB down, several realisations.
  sensing   only m of the 24 masses instrumented, modal coordinates recovered by
            least squares on the retained mode set.  Spatial aliasing is the
            failure mode: with few sensors a high mode is indistinguishable from
            a low one and leaks into the projection.

Usage:  python noise_sensing.py       (writes noise_sensing_results.json)
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
SNRS_DB = (np.inf, 60.0, 50.0, 40.0, 30.0, 20.0)
N_REAL = 5
SENSOR_COUNTS = (24, 16, 12, 8, 6, 5)
TRIM = 20.0     # edge trim in TIME units, not a fraction of the record
SEED0 = 20260928


def base_record(mu=MU, A=A_FIX, mode=bm.MODE):
    """Clean modal and physical histories for one transition-band record."""
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A, mode)
    tt = np.arange(0.0, bm.SPAN, bm.DT)
    qq, qd, _ = L.decay(y0, tt, rtol=bm.RTOL)
    return L, tt, qq, qd, qq @ L.Phi.T, qd @ L.Phi.T     # physical disp, vel


def _floor_fit(t, y):
    """Decay rate from y(t) = y0 exp(-gamma t) + b, the floor-aware fit.

    Additive measurement noise puts a positive pedestal under a measured
    ENERGY: the tail of the record flattens at the noise floor instead of
    continuing to decay.  A log-linear fit reads that flattening as a slower
    decay and biases the rate low, increasingly so as the window runs further
    past the point where the signal meets the floor.  Estimating the pedestal
    as a third parameter removes the bias, and -- unlike truncating the record
    at the floor -- it keeps the points that identify the pedestal.  Returns
    (gamma, b, y0); gamma is nan if the fit does not converge.
    """
    y0 = max(float(np.mean(y[:max(3, len(y) // 100)])), 1e-300)
    b0 = max(float(np.min(y)), y0 * 1e-12)
    span = t[-1] - t[0]
    g0 = max(float(-np.polyfit(t, np.log(np.maximum(y, 1e-300)), 1)[0]), 1e-12)
    try:
        p, _ = curve_fit(lambda tv, g, a, b: a * np.exp(-g * (tv - t[0])) + b,
                         t, y, p0=(g0, y0, b0), maxfev=40000,
                         bounds=([1e-12, 0.0, 0.0],
                                 [1e3 / max(span, 1e-12), np.inf, np.inf]))
        return float(p[0]), float(p[2]), float(p[1])
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _rates(L, tt, qh, qdh, mode, sub):
    """Decay-rate estimates for one record, naive and floor-aware.

    The edge trim is a fixed LENGTH in time units, not a fixed fraction of the
    record.  A proportional trim (the earlier `len(tt)//20`) discards the first
    twentieth of every record, which on an 8000-unit record throws away the first
    400 units -- the highest signal-to-noise part of the decay -- and makes the
    estimator look window-length dependent when the record is not.
    """
    k = int(round(TRIM / bm.DT))
    sl = slice(k, len(tt) - k)
    Ei = 0.5 * (qdh ** 2 + (L.wn ** 2)[None, :] * qh ** 2)
    Es = Ei[:, sub].sum(axis=1)
    env = np.abs(hilbert(qh[:, mode]))
    c = L.ci[mode]

    good = Es[sl] > 0
    g_naive = float(-np.polyfit(tt[sl][good], np.log(Es[sl][good]), 1)[0] / c)
    r_env_naive = float(-2 * np.polyfit(tt[sl], np.log(np.maximum(env[sl], 1e-300)),
                                        1)[0] / c)
    g_sub, b_sub, e0 = _floor_fit(tt[sl], Es[sl])
    g_env, _, _ = _floor_fit(tt[sl], env[sl] ** 2)
    # time at which the clean signal meets the fitted floor: the point beyond
    # which a log-linear fit is reading noise
    t_star = (float(np.log(e0 / b_sub) / g_sub)
              if np.isfinite(g_sub) and b_sub > 0 and e0 > b_sub else float("inf"))
    return dict(gamma_sub_over_c=g_sub / c, gamma_sub_naive_over_c=g_naive,
                r_env_over_c=g_env / c, r_env_naive_over_c=r_env_naive,
                floor_rel=(b_sub / e0 if e0 > 0 else float("nan")),
                t_star=t_star)


def noise_sweep(mode=bm.MODE):
    L, tt, qq, qd, xx, xd = base_record(mode=mode)
    sub = [i for i in range(mode - 2, mode + 3) if 0 <= i < L.n]
    Phi = L.Phi
    out = []
    for snr in SNRS_DB:
        for rep in range(1 if not np.isfinite(snr) else N_REAL):
            rng = np.random.default_rng(SEED0 + 1000 * rep + int(0 if not
                                        np.isfinite(snr) else snr))
            if np.isfinite(snr):
                sx = np.sqrt(np.mean(xx ** 2)) * 10 ** (-snr / 20)
                sv = np.sqrt(np.mean(xd ** 2)) * 10 ** (-snr / 20)
                xn = xx + rng.normal(0, sx, xx.shape)
                vn = xd + rng.normal(0, sv, xd.shape)
            else:
                xn, vn = xx, xd
            qh, qdh = xn @ Phi, vn @ Phi          # full sensing: exact projection
            out.append(dict(snr_db=(None if not np.isfinite(snr) else float(snr)),
                            rep=rep, **_rates(L, tt, qh, qdh, mode, sub)))
    return out


def sensing_sweep(mode=bm.MODE, snr_db=60.0):
    L, tt, qq, qd, xx, xd = base_record(mode=mode)
    sub = [i for i in range(mode - 2, mode + 3) if 0 <= i < L.n]
    out = []
    for m in SENSOR_COUNTS:
        idx = np.unique(np.linspace(0, L.n - 1, m).round().astype(int))
        Ps = L.Phi[idx][:, sub]                   # sensed rows, retained modes
        cond = float(np.linalg.cond(Ps))
        pinv = np.linalg.pinv(Ps)
        for rep in range(N_REAL):
            rng = np.random.default_rng(SEED0 + 7919 * rep + m)
            sx = np.sqrt(np.mean(xx ** 2)) * 10 ** (-snr_db / 20)
            sv = np.sqrt(np.mean(xd ** 2)) * 10 ** (-snr_db / 20)
            xn = xx[:, idx] + rng.normal(0, sx, (len(tt), len(idx)))
            vn = xd[:, idx] + rng.normal(0, sv, (len(tt), len(idx)))
            qs, qds = xn @ pinv.T, vn @ pinv.T    # least squares onto retained modes
            qh = np.zeros((len(tt), L.n)); qdh = np.zeros((len(tt), L.n))
            qh[:, sub], qdh[:, sub] = qs, qds
            out.append(dict(n_sensors=int(len(idx)), rep=rep, cond=cond,
                            **_rates(L, tt, qh, qdh, mode, sub)))
    return out


def window_sweep(spans=(400.0, 1200.0, 2400.0, 4000.0), mode=bm.MODE,
                 snrs=(np.inf, 60.0, 50.0, 40.0, 30.0, 20.0), nrep=3):
    """The noise limit is an interaction between SNR and window length.

    Over a short window the record never approaches the noise floor and the
    recipe looks immune.  Lengthening it has two opposing effects: the
    decay-rate fit gains lever arm and improves, while the tail sinks into the
    noise floor.  Under a log-linear fit the second effect dominates and the
    error collapses; under the floor-aware fit it does not.  Both are stored,
    so the window dependence of each can be read from the file rather than
    asserted here.  `window_fit.py` maps the same question on a finer grid of
    spans out to 8000 units.
    """
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=MU / A_FIX ** 2, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A_FIX, mode)
    sub = [i for i in range(mode - 2, mode + 3) if 0 <= i < L.n]
    rows = []
    for span in spans:
        tt = np.arange(0.0, span, bm.DT)
        qq, qd, _ = L.decay(y0, tt, rtol=bm.RTOL)
        xx, xd = qq @ L.Phi.T, qd @ L.Phi.T
        for snr in snrs:
            for rep in range(1 if not np.isfinite(snr) else nrep):
                rng = np.random.default_rng(SEED0 + 31 * rep
                                            + int(0 if not np.isfinite(snr) else snr))
                if np.isfinite(snr):
                    sx = np.sqrt(np.mean(xx ** 2)) * 10 ** (-snr / 20)
                    sv = np.sqrt(np.mean(xd ** 2)) * 10 ** (-snr / 20)
                    xn = xx + rng.normal(0, sx, xx.shape)
                    vn = xd + rng.normal(0, sv, xd.shape)
                else:
                    xn, vn = xx, xd
                rows.append(dict(span=float(span), rep=rep,
                                 snr_db=(None if not np.isfinite(snr) else float(snr)),
                                 energy_decay=float(np.exp(-L.ci[mode] * span)),
                                 **_rates(L, tt, xn @ L.Phi, vn @ L.Phi, mode, sub)))
    return rows


def main():
    ns, ss, ws = noise_sweep(), sensing_sweep(), window_sweep()

    def agg(rows, key):
        out = {}
        for k in sorted({r[key] for r in rows}, key=lambda v: (v is None, v)):
            g = np.array([r["gamma_sub_over_c"] for r in rows if r[key] == k])
            out[str(k)] = dict(n=int(g.size), mean=float(g.mean()),
                               std=float(g.std()),
                               mean_abs_err=float(np.mean(np.abs(g - 1.0))))
        return out

    res = dict(config=dict(mu=MU, A=A_FIX, N=bm.N, j0=bm.J0,
                           mode_1based=bm.MODE + 1, zeta1=bm.ZETA1,
                           span=bm.SPAN, dt=bm.DT, edge_trim_time=TRIM,
                           n_realisations=N_REAL, sensing_snr_db=60.0),
               noise=ns, sensing=ss, window=ws,
               noise_summary=agg(ns, "snr_db"),
               sensing_summary=agg(ss, "n_sensors"))
    out = os.path.join(HERE, "..", "..", "checkpoints", "noise_sensing_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}")
    print("SNR sweep (full sensing), gamma_sub/c_true:")
    for k, v in res["noise_summary"].items():
        print(f"  {k:>6} dB : mean {v['mean']:.4f} +/- {v['std']:.4f}  "
              f"|err| {v['mean_abs_err']:.4f}")
    print("window x SNR, gamma_sub/c_true:")
    for sp in sorted({r["span"] for r in ws}):
        row = [f"{np.mean([r['gamma_sub_over_c'] for r in ws if r['span']==sp and r['snr_db']==s]):.4f}"
               for s in (None, 40.0, 30.0, 20.0)]
        dec = [r["energy_decay"] for r in ws if r["span"]==sp][0]
        print(f"  span {sp:>6.0f} (decay {dec:.1e}): " + "  ".join(row))
    print("sensor sweep at 60 dB, gamma_sub/c_true:")
    for k, v in res["sensing_summary"].items():
        print(f"  {k:>3} sensors: mean {v['mean']:.4f} +/- {v['std']:.4f}  "
              f"|err| {v['mean_abs_err']:.4f}")


if __name__ == "__main__":
    main()
