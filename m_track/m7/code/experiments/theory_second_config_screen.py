import os as _os
CK = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "checkpoints"))
"""Screen (station j0, tracked mode m) of the 24-chain for a second
configuration predicted to cross its sideband boundary within mu <= 2e-3.

Protocol held fixed: zeta1 = 0.010, station forcing at Omega = omega_lin(m),
A = 2e-2, mu = k3 A^2. kappa(mu) from single-harmonic all-mode harmonic
balance (theory_averaged_threshold.kappa_forced_hb; within ~1.5% of Newton
kappa on the main configuration). kappa_c from the averaged theory.
Only m with 3 omega_m above the band edge (0-based m >= 5) are screened, so
no third-harmonic resonance is available. j0 and 23 - j0 are mirror images.
Writes theory_second_config_screen.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "models"))
import lattice_nl as ln  # noqa: E402
import theory_averaged_threshold as ta  # noqa: E402

A, Z, MU_MAX = 2e-2, 0.010, 2e-3


def kappa_of_mu(j0, m, mu):
    L = ln.Lattice(24, j0, mu / A ** 2, Z)
    return ta.kappa_forced_hb(L, A, L.wn[m])


def screen(args):
    j0, m = args
    L = ln.Lattice(24, j0, 1.0, Z)
    th = ta.first_threshold(L, m, ta.omega_law(L, m, "linear"), kmax=0.5, nscan=500,
                            sigma_raw_max=0.25)
    k_max, nroots = kappa_of_mu(j0, m, MU_MAX)
    out = dict(j0=j0, mode_1based=m + 1, omega=float(L.wn[m]), psi_m=float(L.psi[m]),
               kappa_c=th["kappa_c"], pair=th["pair"], windows=th["windows"],
               kappa_at_mu_max=k_max, hb_roots_at_mu_max=nroots, mu_c=None)
    if th["kappa_c"] is not None and th["kappa_c"] < k_max:
        out["mu_c"] = float(brentq(lambda mu: kappa_of_mu(j0, m, mu)[0] - th["kappa_c"],
                                   1e-8, MU_MAX, xtol=1e-12))
        p, best = ta.predicted_max_multiplier(L, k_max, L.wn[m], m)
        out["pred_mult_at_mu_max"] = p
        out["pred_pair_at_mu_max"] = best["pair"]
    return out


if __name__ == "__main__":
    jobs = [(j0, m) for j0 in range(12) for m in range(5, 24)]
    with Pool(6) as p:
        rows = p.map(screen, jobs)
    json.dump(rows, open(os.path.join(CK, "theory_second_config_screen.json"), "w"), indent=1, default=float)
    ok = sorted([r for r in rows if r["mu_c"]], key=lambda r: r["mu_c"])
    print(len(rows), "screened;", len(ok), "predicted to cross by mu=2e-3")
    for r in ok:
        print(r["j0"], r["mode_1based"], round(r["omega"], 4), round(r["psi_m"], 4),
              r["pair"], round(r["kappa_c"], 4), f"{r['mu_c']:.2e}",
              round(r["pred_mult_at_mu_max"], 5), r["pred_pair_at_mu_max"])
    ref = [r for r in rows if r["j0"] == 1 and r["mode_1based"] in (7, 11)]
    print("reference:", [(r["mode_1based"], r["kappa_c"], r["pair"], r["mu_c"], r["kappa_at_mu_max"]) for r in ref])
