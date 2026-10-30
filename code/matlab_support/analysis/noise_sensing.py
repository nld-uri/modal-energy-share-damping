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

The last point is the one to watch, but it does NOT make short windows safe.
Two effects compete as the window lengthens: the decay-rate fit gains lever arm
and improves, while the tail sinks into the noise floor and biases the rate
downward.  The error is therefore non-monotone in window length, and the optimum
SHORTENS as SNR falls -- longest window when clean, 1200 units at 20 dB here.

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
EDGE = 20
SEED0 = 20260928


def base_record(mu=MU, A=A_FIX, mode=bm.MODE):
    """Clean modal and physical histories for one transition-band record."""
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A, mode)
    tt = np.arange(0.0, bm.SPAN, bm.DT)
    qq, qd, _ = L.decay(y0, tt, rtol=bm.RTOL)
    return L, tt, qq, qd, qq @ L.Phi.T, qd @ L.Phi.T     # physical disp, vel


def _rates(L, tt, qh, qdh, mode, sub):
    k = len(tt) // EDGE
    sl = slice(k, len(tt) - k)
    Ei = 0.5 * (qdh ** 2 + (L.wn ** 2)[None, :] * qh ** 2)
    Es = Ei[:, sub].sum(axis=1)
    env = np.abs(hilbert(qh[:, mode]))
    good = Es[sl] > 0
    g_sub = float(-np.polyfit(tt[sl][good], np.log(Es[sl][good]), 1)[0] / L.ci[mode])
    r_env = float(-2 * np.polyfit(tt[sl], np.log(np.maximum(env[sl], 1e-300)),
                                  1)[0] / L.ci[mode])
    return g_sub, r_env


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
            g_sub, r_env = _rates(L, tt, qh, qdh, mode, sub)
            out.append(dict(snr_db=(None if not np.isfinite(snr) else float(snr)),
                            rep=rep, gamma_sub_over_c=g_sub, r_env_over_c=r_env))
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
            g_sub, r_env = _rates(L, tt, qh, qdh, mode, sub)
            out.append(dict(n_sensors=int(len(idx)), rep=rep, cond=cond,
                            gamma_sub_over_c=g_sub, r_env_over_c=r_env))
    return out


def window_sweep(spans=(400.0, 1200.0, 2400.0, 4000.0), mode=bm.MODE,
                 snrs=(np.inf, 60.0, 50.0, 40.0, 30.0, 20.0), nrep=3):
    """The noise limit is set by window length, not by SNR alone.

    Over a short window the record never approaches the noise floor and the
    recipe looks immune; lengthen it and the tail flattens, so the fitted rate
    is too SMALL.  The bias is therefore one-sided and predictable.
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
                g_sub, _ = _rates(L, tt, xn @ L.Phi, vn @ L.Phi, mode, sub)
                rows.append(dict(span=float(span), rep=rep,
                                 snr_db=(None if not np.isfinite(snr) else float(snr)),
                                 energy_decay=float(np.exp(-L.ci[mode] * span)),
                                 gamma_sub_over_c=g_sub))
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
                           span=bm.SPAN, dt=bm.DT, edge_trim=EDGE,
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
