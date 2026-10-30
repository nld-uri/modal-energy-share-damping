"""Excitation bandwidth: the second axis, and the test that separates law from proxy.

Paper 3.  Section 5 shows that for a cubic element excitation amplitude and
nonlinearity strength collapse into one similarity parameter mu = k3 A^2, so the
only independent second axis must break that homogeneity.  Excitation bandwidth
does, because a scaling of the state leaves the forcing SPECTRUM untouched.

Why this experiment matters beyond adding an axis.  A first pass found the
identified damping ratio tracking the tracked mode's energy share across the mu
sweep (correlation 0.996) and it was tempting to call the share the governing
variable.  The bandwidth records refute that: across them the share and the bias
are uncorrelated (-0.13), and at band halfwidth 0.25 the mode can hold 0.65 of
the record energy and still return the correct damping.

The derived law of `share_law.py` predicts exactly this, and that is the point
of running it here:

    c_ident  =  cbar  -  d(ln f)/dt

The bias is the DRIFT of the share, not the share.  Under resonant single-tone
forcing past the instability onset the share drifts strongly (the sidebands stop
being pumped during decay and energy returns to the carrier), so a low share and
a large bias coincide and the proxy looks right.  Under broadband forcing the
share is low because neighbouring modes are driven DIRECTLY -- the band reaches
omega_6 and omega_8 once the halfwidth exceeds about 0.14 -- but it is roughly
steady, so the drift and hence the bias are small.  Two protocols, similar
shares, different drift, different bias: if the law holds on both, it is the law.

Usage:  python bandwidth_axis.py      (writes bandwidth_axis_results.json)
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

K3 = 1.0
BETAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
TARGETS = (0.45, 0.75)
SEEDS = (1, 2)
RELEASES = (0.0, 137.0, 291.0)
N_HARM = 25
T_SETTLE = 6000.0
EDGE = 20


def band_force(L, beta, mode, target_am, n_h=N_HARM, seed=0):
    """Equal-amplitude random-phase harmonics on [w_m(1-beta), w_m(1+beta)],
    scaled so the LINEAR modal rms of the tracked mode equals target_am."""
    wm = L.wn[mode]
    wk = np.array([wm]) if beta == 0.0 else np.linspace(wm * (1 - beta),
                                                        wm * (1 + beta), n_h)
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, wk.size)
    H2 = 1.0 / np.abs(wm ** 2 - wk ** 2 + 1j * L.ci[mode] * wk) ** 2
    a = np.sqrt(2 * target_am ** 2 / (L.psi[mode] ** 2 * H2.sum()))
    return wk, ph, float(a)


def neighbours_in_band(L, beta, mode):
    """How many other modal frequencies the excitation band reaches directly."""
    wm = L.wn[mode]
    lo, hi = wm * (1 - beta), wm * (1 + beta)
    return int(sum(1 for i, w in enumerate(L.wn) if i != mode and lo <= w <= hi))


def one_cell(beta, target_am, seed, k3=K3, mode=bm.MODE):
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=k3, zeta1=bm.ZETA1)
    wk, ph, a = band_force(L, beta, mode, target_am, seed=seed)
    n = L.n
    W2, C, PSI = L.wn ** 2, L.ci, L.psi

    def f_rhs(t, z):
        u = a * np.cos(wk * t + ph).sum()
        return np.concatenate([z[n:],
                               -C * z[n:] - W2 * z[:n]
                               - PSI * (k3 * (PSI @ z[:n]) ** 3) + PSI * u])

    s = solve_ivp(f_rhs, (0.0, T_SETTLE + max(RELEASES)), np.zeros(2 * n),
                  method="DOP853", rtol=1e-9, atol=1e-11,
                  t_eval=T_SETTLE + np.array(RELEASES))
    out = []
    tt = np.arange(0.0, bm.SPAN, bm.DT)
    k = len(tt) // EDGE
    sl = slice(k, len(tt) - k)
    for j in range(s.y.shape[1]):
        aa, vv, ac = L.decay(s.y[:, j], tt, rtol=bm.RTOL)
        r = bm._three_estimates(L, aa, vv, ac, mode)

        Ei = 0.5 * (vv ** 2 + W2[None, :] * aa ** 2)
        Etot = Ei.sum(axis=1)
        fsh = Ei[:, mode] / Etot
        cbar = (Ei * C[None, :]).sum(axis=1) / Etot
        dlnf = float(np.polyfit(tt[sl], np.log(fsh[sl]), 1)[0])
        env = np.abs(hilbert(aa[:, mode]))
        r_env = float(-2 * np.polyfit(tt[sl], np.log(env[sl]), 1)[0] / C[mode])

        r.update(beta=float(beta), target_am=float(target_am), seed=int(seed),
                 release=float(RELEASES[j]), forcing_amp=a,
                 n_neighbours_in_band=neighbours_in_band(L, beta, mode),
                 share_mean=float(fsh.mean()), share_start=float(fsh[0]),
                 share_end=float(fsh[-1]), dlnf_dt=dlnf,
                 cbar_over_ctrue=float(cbar[sl].mean() / C[mode]),
                 r_env_trimmed=r_env,
                 r_derived=float((cbar[sl].mean() - dlnf) / C[mode]))
        out.append(r)
    return out


def main():
    recs = [r for b in BETAS for t in TARGETS for sd in SEEDS
            for r in one_cell(b, t, sd)]
    e = np.array([r["r_env_trimmed"] for r in recs])
    d = np.array([r["r_derived"] for r in recs])
    s = np.array([r["share_mean"] for r in recs])
    ok = np.abs(e) < 5.0            # exclude records where the fit itself diverged
    res = dict(
        config=dict(N=bm.N, j0=bm.J0, mode_1based=bm.MODE + 1, k3=K3,
                    zeta1=bm.ZETA1, betas=list(BETAS), targets=list(TARGETS),
                    seeds=list(SEEDS), releases=list(RELEASES),
                    n_harmonics=N_HARM, t_settle=T_SETTLE, edge_trim=EDGE),
        records=recs,
        summary=dict(
            n=len(recs), n_used=int(ok.sum()),
            derived_mean_abs_err=float(np.mean(np.abs(e[ok] - d[ok]))),
            derived_worst=float(np.max(np.abs(e[ok] - d[ok]))),
            derived_corr=float(np.corrcoef(e[ok], d[ok])[0, 1]),
            share_proxy_mean_abs_err=float(np.mean(np.abs(e[ok] - s[ok]))),
            share_proxy_corr=float(np.corrcoef(e[ok], s[ok])[0, 1])))
    out = os.path.join(HERE, "..", "..", "checkpoints",
                       "bandwidth_axis_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}  ({len(recs)} records)")
    print(f"derived law: mean |err| {res['summary']['derived_mean_abs_err']:.4f}, "
          f"corr {res['summary']['derived_corr']:.4f}")
    print(f"share proxy: mean |err| {res['summary']['share_proxy_mean_abs_err']:.4f}, "
          f"corr {res['summary']['share_proxy_corr']:.4f}")


if __name__ == "__main__":
    main()
