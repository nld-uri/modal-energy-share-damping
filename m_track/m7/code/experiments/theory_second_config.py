import os as _os
CK = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "checkpoints"))
"""Second configuration(s): averaged-theory prediction, Newton-shot Floquet
confirmation, and a release sweep across the boundary.

Protocol identical to bias_map (zeta1 = 0.010, station forcing at
Omega = omega_lin(m), A = 2e-2, mu = k3 A^2, settle T_SETTLE = 6000, release,
400-unit decay at dt = 0.05, bias_map._three_estimates with n_track =
max(8, m+3)).

CONFIGS (0-based j0, 0-based mode) chosen from theory_second_config_screen.json:
  primary   j0=4, mode 5 (tracked mode 6): predicted pair (4,8)
  secondary j0=9, mode 6 (tracked mode 7): predicted pair (1,14)
Writes theory_second_config.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _s in ("models", "identification", "experiments"):
    sys.path.insert(0, os.path.join(HERE, "..", _s))
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402
import theory_averaged_threshold as ta  # noqa: E402
import theory_floquet_runs as fr  # noqa: E402

A, Z = 2e-2, 0.010
CONFIGS = {"primary": (4, 5), "secondary": (9, 6)}
MU_COARSE = (2e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1.3e-3, 1.6e-3, 2e-3)
MU_RELEASE = {"primary": (3e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3, 1.3e-3, 2e-3),
              "secondary": (4e-4, 6e-4, 7e-4, 8.5e-4, 1e-3, 1.3e-3, 2e-3)}


def floq(args):
    tag, mu = args
    j0, m = CONFIGS[tag]
    r = fr.forced_floquet(mu, Z, j0=j0, mode=m)
    r["tag"] = tag
    return r


def release(args):
    tag, mu = args
    j0, m = CONFIGS[tag]
    L = ln.Lattice(n=bm.N, j0=j0, k3=mu / A ** 2, zeta1=Z)
    y0 = bm._settle(L, A, m)
    aa, vv, ac = L.decay(y0, np.arange(0.0, bm.SPAN, bm.DT), rtol=bm.RTOL)
    est = bm._three_estimates(L, aa, vv, ac, m)
    E = np.mean(0.5 * (vv ** 2 + (L.wn ** 2)[None, :] * aa ** 2), axis=0)
    E = E / E.sum()
    top = np.argsort(E)[::-1][:4]
    est.update(tag=tag, mu=float(mu), k3=float(L.k3), j0=j0, mode_1based=m + 1,
               energy_top=[(int(t) + 1, float(E[t])) for t in top],
               n_track=max(8, m + 3))
    return est


def bisect(tag, lo, hi, nstep=5):
    rows = []
    for _ in range(nstep):
        mid = np.sqrt(lo * hi)
        r = floq((tag, mid)); rows.append(r)
        if r["max_multiplier"] > 1:
            hi = mid
        else:
            lo = mid
    return rows


def main():
    res = {"protocol": dict(A=A, zeta1=Z, Omega="omega_lin(m)", t_settle=bm.T_SETTLE,
                            span=bm.SPAN, dt=bm.DT)}
    with Pool(4) as pool:   # m_track: 10 -> 4 workers (verifier running alongside); no other change
        rel = pool.map_async(release, [(t, mu) for t in CONFIGS for mu in MU_RELEASE[t]])
        coarse = pool.map(floq, [(t, mu) for t in CONFIGS for mu in MU_COARSE])
        bis_jobs = []
        for t in CONFIGS:
            rr = sorted([r for r in coarse if r["tag"] == t], key=lambda r: r["mu"])
            br = next(((a["mu"], b["mu"]) for a, b in zip(rr, rr[1:])
                       if a["max_multiplier"] <= 1 < b["max_multiplier"]), None)
            bis_jobs.append((t, br))
        bis = pool.starmap(bisect, [(t, *br) for t, br in bis_jobs if br is not None])
        rel = rel.get()
    bis_flat = [r for b in bis for r in b]

    for t, (j0, m) in CONFIGS.items():
        L = ln.Lattice(24, j0, 1.0, Z)
        th = ta.first_threshold(L, m, ta.omega_law(L, m, "linear"), kmax=0.5)
        kap = lambda mu: ta.kappa_forced_hb(ln.Lattice(24, j0, mu / A ** 2, Z), A, L.wn[m])[0]
        mu_c_hb = float(brentq(lambda mu: kap(mu) - th["kappa_c"], 1e-8, 2e-3, xtol=1e-12))
        fl = sorted([r for r in coarse + bis_flat if r["tag"] == t], key=lambda r: r["mu"])
        br = None
        for a, b in zip(fl, fl[1:]):
            if a["max_multiplier"] <= 1 < b["max_multiplier"]:
                f = (1 - a["max_multiplier"]) / (b["max_multiplier"] - a["max_multiplier"])
                br = dict(mu_stable=a["mu"], mu_unstable=b["mu"],
                          kappa_stable=a["kappa"], kappa_unstable=b["kappa"],
                          kappa_interp=a["kappa"] + f * (b["kappa"] - a["kappa"]),
                          mu_interp=float(np.exp(np.log(a["mu"]) + f * np.log(b["mu"] / a["mu"]))),
                          unstable_content=b["content"])
                break
        # any later restabilisation in the sweep?
        restab = [(a["mu"], b["mu"]) for a, b in zip(fl, fl[1:])
                  if a["max_multiplier"] > 1 >= b["max_multiplier"]]
        mc = []
        for r in fl:
            p, best = ta.predicted_max_multiplier(L, r["kappa"], L.wn[m], m)
            mc.append(dict(mu=r["mu"], kappa=r["kappa"], floquet=r["max_multiplier"],
                           theory=p, theory_pair=best["pair"], floquet_content=r["content"],
                           kappa_hb=kap(r["mu"])))
        res[t] = dict(j0_0based=j0, station_1based=j0 + 1, mode_1based=m + 1,
                      omega_lin=float(L.wn[m]), psi_m=float(L.psi[m]),
                      theory=dict(kappa_c=th["kappa_c"], pair=th["pair"],
                                  windows=th["windows"], mu_c_via_hb=mu_c_hb),
                      floquet_bracket=br, floquet_restabilisations=restab,
                      floquet_records=fl, multiplier_compare=mc,
                      release=sorted([r for r in rel if r["tag"] == t], key=lambda r: r["mu"]))
    json.dump(res, open(os.path.join(CK, "theory_second_config.json"), "w"), indent=1, default=float)
    print("wrote theory_second_config.json")


if __name__ == "__main__":
    main()
