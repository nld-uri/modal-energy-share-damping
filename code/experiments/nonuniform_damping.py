"""Non-uniform (Rayleigh) damping: the test that gives the drift law content.

Every other experiment in this paper uses UNIFORM modal damping,
c_i = 2 zeta_1 omega_1.  That makes the share-weighted mean damping

    cbar(t) = sum_i E_i(t) c_i / sum_i E_i(t)

identically equal to c_true whatever the energy distribution does, so the
law

    c_ident  =  cbar  -  d(ln f)/dt ,        f = E_m / E_tot          (*)

collapses to c_ident = c_true - d(ln f)/dt and its first term carries no
information.  Agreement with the measured envelope estimate is then close to
algebra: the envelope measures -d(ln E_m)/dt, E_m = f E_tot, and
d(ln E_tot)/dt = -cbar, so (*) cannot fail.  Reporting that agreement as a
validation of the law would be an identity check dressed as a test.

With Rayleigh damping, C = alpha M + beta K, the modal coefficients

    c_i = alpha + beta omega_i^2

differ from mode to mode -- here by a factor of about sixteen across the
band -- so cbar moves as energy redistributes, and (*) becomes falsifiable in
the part that matters: the WEIGHTING.  The share weighting is correct only if
each mode is separately in equipartition over a cycle, which is a physical
assumption, not bookkeeping.  Three alternative weightings that are equally
plausible a priori are carried through the same records:

    c_m         the tracked mode's own coefficient
    c_flat      the unweighted mean over all modes
    c_amp       weighted by modal amplitude rather than energy

alpha and beta are fixed by two requirements: a spread of SPREAD across the
band, and c at the tracked mode equal to the uniform value used everywhere
else in the paper, so that the tracked mode's decay time and hence the
analysis window are unchanged.

Usage:  python nonuniform_damping.py   (writes nonuniform_damping_results.json)
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
import bandwidth_axis as ba  # noqa: E402

SPREAD = 16.0          # c_max / c_min across the 24 modes
TRIM = 20.0            # fixed-length edge trim, in time units, not samples
BETAS = (0.0, 0.10, 0.20, 0.30, 0.40)
TARGETS = (0.45, 0.75)
RELEASES = (0.0, 137.0, 291.0)
SEED = 1
SINGLE_TONE_MU = (1.0e-3, 1.6e-3, 2.0e-3)
K3_TONE = 1.0


def rayleigh_ci(wn, mode, c_track, spread=SPREAD):
    """c_i = alpha + beta omega_i^2, with c[mode] = c_track and the ratio
    c_max/c_min = spread over the modes present."""
    w2 = wn ** 2
    # c_max/c_min = (alpha + beta w2max)/(alpha + beta w2min) = spread
    # -> alpha (spread - 1) = beta (w2max - spread w2min)
    r = (w2.max() - spread * w2.min()) / (spread - 1.0)   # alpha = beta * r
    beta = c_track / (r + w2[mode])
    alpha = beta * r
    return alpha + beta * w2, float(alpha), float(beta)


def _weightings(L, aa, vv, mode):
    """cbar under four weightings, each averaged over the trimmed window."""
    Ei = 0.5 * (vv ** 2 + (L.wn ** 2)[None, :] * aa ** 2)
    Etot = Ei.sum(axis=1)
    amp = np.abs(aa)
    return dict(
        c_share=(Ei * L.ci[None, :]).sum(axis=1) / Etot,
        c_mode=np.full(len(aa), L.ci[mode]),
        c_flat=np.full(len(aa), L.ci.mean()),
        c_amp=(amp * L.ci[None, :]).sum(axis=1) / amp.sum(axis=1),
    ), Ei[:, mode] / Etot


def _record(L, aa, vv, mode, dt, **meta):
    tt = np.arange(len(aa)) * dt
    k = int(round(TRIM / dt))
    sl = slice(k, len(tt) - k)
    cb, share = _weightings(L, aa, vv, mode)

    dlnf = float(np.polyfit(tt[sl], np.log(share[sl]), 1)[0])
    env = np.abs(hilbert(aa[:, mode]))
    lg = np.log(env[sl])
    p, res = np.polyfit(tt[sl], lg, 1, full=True)[:2]
    c_env = float(-2 * p[0])
    r2 = float(1 - res[0] / ((lg - lg.mean()) ** 2).sum()) if len(res) else float("nan")
    h = (sl.stop - sl.start) // 2
    s1 = np.polyfit(tt[sl][:h], lg[:h], 1)[0]
    s2 = np.polyfit(tt[sl][h:], lg[h:], 1)[0]

    out = dict(meta, c_true_mode=float(L.ci[mode]), c_env=c_env, dlnf_dt=dlnf,
               share_mean=float(share[sl].mean()), share_start=float(share[0]),
               share_end=float(share[-1]), env_r2=r2,
               env_split=float(abs(s1 - s2) / abs(0.5 * (s1 + s2))))
    for name, series in cb.items():
        bar = float(series[sl].mean())
        out[f"{name}_bar"] = bar
        out[f"{name}_pred"] = bar - dlnf          # the law under this weighting
        out[f"{name}_err"] = (bar - dlnf - c_env) / c_env
    return out


def band_cells(L, mode):
    out = []
    for beta in BETAS:
        for target in TARGETS:
            wk, ph, a = ba.band_force(L, beta, mode, target, seed=SEED)
            n, W2, C, PSI = L.n, L.wn ** 2, L.ci, L.psi

            def f_rhs(t, z):
                u = a * np.cos(wk * t + ph).sum()
                return np.concatenate([z[n:], -C * z[n:] - W2 * z[:n]
                                       - PSI * (L.k3 * (PSI @ z[:n]) ** 3)
                                       + PSI * u])

            from scipy.integrate import solve_ivp
            s = solve_ivp(f_rhs, (0.0, ba.T_SETTLE + max(RELEASES)),
                          np.zeros(2 * n), method="DOP853", rtol=1e-9,
                          atol=1e-11,
                          t_eval=ba.T_SETTLE + np.array(RELEASES))
            tt = np.arange(0.0, bm.SPAN, bm.DT)
            for j in range(s.y.shape[1]):
                aa, vv, _ = L.decay(s.y[:, j], tt, rtol=bm.RTOL)
                out.append(_record(L, aa, vv, mode, bm.DT, protocol="band",
                                   beta=float(beta), target_am=float(target),
                                   release=float(RELEASES[j])))
                print(f"    band beta={beta:.2f} target={target:.2f} "
                      f"release={RELEASES[j]:.0f}  share_err="
                      f"{out[-1]['c_share_err']:+.4f}", flush=True)
    return out


def tone_cells(L, mode):
    out = []
    tt = np.arange(0.0, bm.SPAN, bm.DT)
    for mu in SINGLE_TONE_MU:
        A = float(np.sqrt(mu / K3_TONE))
        y0 = bm._settle(L, A, mode)
        if y0 is None:
            continue
        aa, vv, _ = L.decay(y0, tt, rtol=bm.RTOL)
        out.append(_record(L, aa, vv, mode, bm.DT, protocol="tone",
                           mu=float(mu), A=A))
        print(f"    tone mu={mu:.1e}  share_err={out[-1]['c_share_err']:+.4f}",
              flush=True)
    return out


def main():
    mode = bm.MODE
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=K3_TONE, zeta1=bm.ZETA1)
    c_track = 2 * bm.ZETA1 * L.wn[0]
    L.ci, alpha, beta = rayleigh_ci(L.wn, mode, c_track)
    print(f"Rayleigh damping: alpha={alpha:.6e} beta={beta:.6e}; "
          f"c in [{L.ci.min():.5f}, {L.ci.max():.5f}], "
          f"c[{mode}]={L.ci[mode]:.6f} (uniform value {c_track:.6f})",
          flush=True)
    recs = band_cells(L, mode) + tone_cells(L, mode)

    summary = {}
    for w in ("c_share", "c_mode", "c_flat", "c_amp"):
        e = np.abs([r[f"{w}_err"] for r in recs])
        summary[w] = dict(median=float(np.median(e)), worst=float(e.max()),
                          n_above_10pct=int((e > 0.10).sum()))
    cb = [r["c_share_bar"] / r["c_true_mode"] for r in recs]
    summary["cbar_over_cm"] = dict(min=float(min(cb)), max=float(max(cb)))
    out = dict(config=dict(n=bm.N, j0=bm.J0, mode=mode, k3=K3_TONE,
                           span=bm.SPAN, dt=bm.DT, trim=TRIM, spread=SPREAD,
                           alpha=alpha, beta=beta, seed=SEED,
                           c_i=[float(c) for c in L.ci],
                           c_uniform_elsewhere=float(c_track)),
               records=recs, summary=summary)
    dst = os.path.join(HERE, "..", "..", "checkpoints",
                       "nonuniform_damping_results.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n{len(recs)} records -> {os.path.normpath(dst)}")
    for w, s in summary.items():
        if "median" not in s:          # cbar_over_cm is a range, not an error
            print(f"  {w:>8}: min {s['min']:.4f}  max {s['max']:.4f}")
            continue
        print(f"  {w:>8}: median |err| {s['median']:.4f}  worst {s['worst']:.4f}"
              f"  above 10%: {s['n_above_10pct']}")


if __name__ == "__main__":
    main()
