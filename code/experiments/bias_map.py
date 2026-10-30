"""Damping-bias map for the 24-mass lattice with a grounded cubic element.

Paper 3 (damping identifiability).  Each record carries THREE estimates of the
same modal damping coefficient c_m, so that a physical bias can be separated
from an artefact of the decomposition:

  c_gcvd  -- from the windowed GCVD state relation (the method under test)
  c_env   -- from a Hilbert log-envelope fit to the modal coordinate
             (needs no decomposition; the reference measurement)
  c_flux  -- the prediction c_m - <P_m>/<qdot_m^2> from the exact modal
             energy balance, with P_m = -psi_m k3 (psi.q)^3 qdot_m

Protocol.  A free decay CANNOT be used as the amplitude axis: two windows at
the same station rms but reached from different launch amplitudes give
different bias (0.979 vs 0.450 at k3=1), because station rms does not fix the
modal state.  Instead each cell is prepared independently: resonant station
forcing at Omega = omega_m builds up mode m from rest, the force is released,
and the first window of the ensuing free decay is analysed.  The state is then
pinned by the protocol rather than inherited from a violent history.

Usage:  python bias_map.py            (writes bias_map_results.json)
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "identification"))
import lattice_nl as ln  # noqa: E402
from gcvd import windowed_gcvd  # noqa: E402

N = 24
J0 = 1          # 0-based station index (retuned configuration)
MODE = 6        # 0-based; mode 7
ZETA1 = 0.010
SPAN = 400.0
DT = 0.05
TRIM = 20.0     # edge trim in TIME units (not samples): see _three_estimates
T_SETTLE = 6000.0
RTOL = 1e-9
K3S = (0.0, 0.5, 1.0, 2.0, 4.0)
AMPS = (1e-3, 3e-3, 5e-3, 1e-2, 2e-2)
SPANS = (50.0, 100.0, 200.0, 400.0, 800.0)
A_THRESHOLD = 2e-2          # fixed forcing amplitude for the onset sweep
THRESHOLD_MUS = (3.0e-4, 4.0e-4, 5.0e-4, 6.0e-4, 7.0e-4, 8.0e-4,
                 1.0e-3, 1.3e-3, 1.6e-3, 2.0e-3)


def _settle(L, A, mode):
    """Integrate the resonantly forced lattice from rest to steady state."""
    s = solve_ivp(L.rhs_forced(A, L.wn[mode]), (0.0, T_SETTLE), np.zeros(2 * L.n),
                  method="DOP853", rtol=RTOL, atol=RTOL * 1e-2)
    return None if s.status != 0 else s.y[:, -1]


def _three_estimates(L, aa, vv, ac, mode, dt=None):
    """c_gcvd / c_env / c_flux for one free-decay window, each over c_true.

    `dt` defaults to the module-level DT; pass it explicitly for a record
    sampled at any other rate, or the decay rates come back wrong silently.
    """
    psi, wn2 = L.psi, L.wn ** 2
    w = aa @ psi
    tt = np.arange(len(aa)) * (DT if dt is None else dt)

    # n_track must reach the tracked mode. windowed_gcvd keeps the n_track
    # LOWEST-frequency poles, so a fixed 8 silently excludes any mode above
    # the eighth: tracking mode 11 that way returns the mode-8 pole, which
    # looks like a 9 per cent damping error and is really a missing candidate.
    n_tr = max(8, mode + 3)
    gc = windowed_gcvd(aa, vv, ac, n_windows=1, n_track=n_tr, M=np.eye(L.n))
    m_ = int(np.argmax([abs(gc["shape"][0, q][mode]) for q in range(n_tr)]))
    r_gcvd = 2 * gc["zeta"][0, m_] * gc["wn"][0, m_] / L.ci[mode]

    env = np.abs(hilbert(aa[:, mode]))
    # Fixed-LENGTH edge trim. A proportional trim (len//20) discards a twentieth of
    # every record, which on a long record throws away the highest-signal
    # segment and makes the fit look window-length dependent when it is not.
    k = int(round(TRIM / (DT if dt is None else dt)))
    sl = slice(k, len(env) - k)
    lg = np.log(env[sl])
    p, res = np.polyfit(tt[sl], lg, 1, full=True)[:2]
    r_env = -2 * p[0] / L.ci[mode]

    # Straightness of the log-decay: the data-only criterion for whether an
    # exponential rate exists at all. R^2 of the log-linear fit, and the
    # first-half/second-half slope split. A rate read off a decay that fails
    # these is a window average of a varying rate, not a damping coefficient.
    env_r2 = float(1 - res[0] / ((lg - lg.mean()) ** 2).sum()) if len(res) else float("nan")
    h = (sl.stop - sl.start) // 2
    s1 = np.polyfit(tt[sl][:h], lg[:h], 1)[0]
    s2 = np.polyfit(tt[sl][h:], lg[h:], 1)[0]
    env_split = float(abs(s1 - s2) / abs(0.5 * (s1 + s2)))

    # Share and its drift: the two terms of the law of Section 8, stored per
    # record so the law can be read off the file rather than recomputed.
    Ei_t = 0.5 * (vv ** 2 + wn2[None, :] * aa ** 2)
    share_t = Ei_t[:, mode] / Ei_t.sum(axis=1)
    dlnf_dt = float(np.polyfit(tt[sl], np.log(share_t[sl]), 1)[0])

    P = -(L.k3 * w ** 3)[:, None] * vv * psi[None, :]
    r_flux = (L.ci[mode] - np.mean(P[:, mode]) / np.mean(vv[:, mode] ** 2)) / L.ci[mode]

    E = np.mean(0.5 * (vv ** 2 + wn2[None, :] * aa ** 2), axis=0)
    sv = np.linalg.svd(np.hstack([aa, vv]), compute_uv=False)
    return dict(r_gcvd=float(r_gcvd), r_env=float(r_env), r_flux=float(r_flux),
                stiff_pct=float(100 * (gc["wn"][0, m_] / L.wn[mode] - 1)),
                a_m=float(np.sqrt(np.mean(aa[:, mode] ** 2))),
                w_rms=float(np.sqrt(np.mean(w ** 2))),
                e_frac=float(E[mode] / E.sum()), cond=float(sv[0] / sv[-1]),
                env_r2=env_r2, env_split=env_split, dlnf_dt=dlnf_dt,
                share_start=float(share_t[0]), share_end=float(share_t[-1]))


def controlled_release(k3, A, span=SPAN, mode=MODE, j0=J0):
    """One map cell: settle under resonant forcing, release, analyse."""
    L = ln.Lattice(n=N, j0=j0, k3=k3, zeta1=ZETA1)
    y0 = _settle(L, A, mode)
    if y0 is None:
        return None
    aa, vv, ac = L.decay(y0, np.arange(0.0, span, DT), rtol=RTOL)
    out = _three_estimates(L, aa, vv, ac, mode)
    out.update(k3=float(k3), A=float(A), span=float(span))
    return out


def span_sweep(k3, A, spans=SPANS, mode=MODE, j0=J0):
    """Nested windows of one release record: is the bias window-length driven?"""
    L = ln.Lattice(n=N, j0=j0, k3=k3, zeta1=ZETA1)
    y0 = _settle(L, A, mode)
    aa, vv, ac = L.decay(y0, np.arange(0.0, max(spans), DT), rtol=RTOL)
    rows = []
    for sp in spans:
        k = int(sp / DT)
        r = _three_estimates(L, aa[:k], vv[:k], ac[:k], mode)
        r.update(k3=float(k3), A=float(A), span=float(sp),
                 cycles=float(sp / (2 * np.pi / L.wn[mode])))
        rows.append(r)
    return rows


def history_test(k3, target, mode=MODE, j0=J0):
    """Same station rms reached two ways: deep in a violent decay, or at once."""
    L = ln.Lattice(n=N, j0=j0, k3=k3, zeta1=ZETA1)
    nsp, step = int(SPAN / DT), int(SPAN / DT) // 4

    def window(u0, long_run):
        tt = np.arange(0.0, 6000.0 if long_run else SPAN + DT, DT)
        aa, vv, ac = L.decay(np.concatenate([u0 * L.psi, np.zeros(L.n)]), tt, rtol=1e-10)
        w = aa @ L.psi
        if long_run:
            st = np.arange(0, len(tt) - nsp, step)
            rms = np.array([np.sqrt(np.mean(w[i:i + nsp] ** 2)) for i in st])
            j = int(st[np.argmin(np.abs(rms - target))])
        else:
            j = 0
        sl = slice(j, j + nsp)
        return aa[sl], vv[sl], ac[sl], float(tt[j])

    lo, hi = 1e-3, 8.0
    for _ in range(40):                      # launch amplitude hitting target at once
        mid = 0.5 * (lo + hi)
        a_, _, _, _ = window(mid, False)
        if np.sqrt(np.mean((a_ @ L.psi) ** 2)) < target:
            lo = mid
        else:
            hi = mid
    rows = []
    for u0, tag, long_run in ((3.0, "violent", True), (0.5 * (lo + hi), "gentle", False)):
        aa, vv, ac, t0 = window(u0, long_run)
        r = _three_estimates(L, aa, vv, ac, mode)
        r.update(k3=float(k3), target=float(target), u0=float(u0),
                 history=tag, t_window=t0)
        rows.append(r)
    return rows


def threshold_sweep(mus=THRESHOLD_MUS, A=A_THRESHOLD):
    """Fine sweep in mu across the onset, at fixed forcing amplitude.

    The grid above is coarse near the onset because it is laid out in (k3, A);
    this walks mu directly, which is the only parameter that matters (Sec 5),
    by setting k3 = mu / A^2 at fixed A.  Sections 5 and 6 of the paper use the
    union of this sweep and the grid.
    """
    return [c for mu in mus if (c := controlled_release(mu / A ** 2, A)) is not None]


def similarity_check(grid):
    """Spread within each group of cells sharing one mu = k3 A^2.

    Stored rather than quoted from a run log: several quantities hold the
    invariance, each to its own precision, and the figure captions quote them.
    The damping ratios are near unity below the onset, so their ABSOLUTE
    spread is the informative number; the frequency shift ranges over three
    decades, so its RELATIVE spread is.
    """
    groups = {}
    for c in grid:
        groups.setdefault(round(c["k3"] * c["A"] ** 2, 12), []).append(c)
    inv = {m: v for m, v in groups.items() if len(v) > 1 and m > 0.0}
    out = dict(n_groups=len(inv), cells_per_group=sorted(len(v) for v in inv.values()))
    for key in ("r_env", "r_gcvd", "stiff_pct"):
        out[f"max_spread_{key}"] = max(
            max(c[key] for c in v) - min(c[key] for c in v) for v in inv.values())
        out[f"max_rel_spread_{key}"] = max(
            (max(c[key] for c in v) - min(c[key] for c in v))
            / abs(np.mean([c[key] for c in v])) for v in inv.values())
    out["groups"] = [dict(mu=m, k3=[c["k3"] for c in v], A=[c["A"] for c in v],
                          r_env=[c["r_env"] for c in v],
                          r_gcvd=[c["r_gcvd"] for c in v],
                          stiff_pct=[c["stiff_pct"] for c in v])
                     for m, v in sorted(inv.items())]
    return out


def main():
    res = {"config": dict(N=N, j0=J0, mode_0based=MODE, mode_1based=MODE + 1,
                          zeta1=ZETA1, span=SPAN, dt=DT, t_settle=T_SETTLE,
                          rtol=RTOL, forcing="station, Omega = omega_m, released",
                          threshold_A=A_THRESHOLD)}
    res["grid"] = [c for k3 in K3S for A in AMPS
                   if (c := controlled_release(k3, A)) is not None]
    res["threshold_sweep"] = threshold_sweep()
    res["span_sweep"] = [r for k3, A in ((1.0, 2e-2), (4.0, 5e-3), (4.0, 2e-2))
                         for r in span_sweep(k3, A)]
    res["similarity_check"] = similarity_check(res["grid"])
    res["history_test"] = [r for k3, t in ((1.0, 0.160), (2.0, 0.100), (4.0, 0.060))
                           for r in history_test(k3, t)]
    out = os.path.join(HERE, "..", "..", "checkpoints", "bias_map_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}  ({len(res['grid'])} grid cells)")


if __name__ == "__main__":
    main()
