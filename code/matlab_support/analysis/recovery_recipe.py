"""What to measure to recover the true damping, and why the naive choice fails.

Paper 3.  A correction of the form c_true = c_ident + d(ln f)/dt was briefly
entertained and is WRONG -- or rather, vacuous.  With E_m = f E_tot the product
rule gives, for ANY modal subset S used to define the share,

    gamma_S - d(ln f_S)/dt  ==  -d(ln E_m)/dt

identically, which is the single-mode estimate itself.  Evaluating the "correction"
returns the quantity it was supposed to correct.  Checked numerically: five modal
subsets from {mode 7} to all 24 return the same number to every printed digit.

What survives is simpler and is genuinely useful.  For a CONSERVATIVE nonlinearity
with linear modal damping the total energy decays at the energy-weighted mean modal
damping cbar, so the estimand is recovered by fitting the decay of TOTAL energy
rather than of one modal coordinate.  The practical question is then how much of the
structure has to be observed for that sum to be adequate, and the answer is sharper
than "all of it":

    mode 7 alone          mean |gamma/c - 1| = 0.304, worst 0.541
    modes 7 +/- 1         0.287, 0.501     <- the NAIVE choice, and it fails
    modes 7 +/- 2         0.0039, 0.0048   <- the sideband pair; sufficient
    modes 7 +/- 4         0.0012, 0.0016
    all 24 modes          0.0012, 0.0014

Including the nearest neighbours buys almost nothing because the energy does not go
there; it goes to the m-2 / m+2 sideband pair selected by the instability of
`onset_mechanism.py`.  The Floquet analysis is therefore operationally necessary and
not decorative: it names, before any test is run, the small set of modes whose energy
must be captured for the damping estimate to be meaningful.

Usage:  python recovery_recipe.py     (writes recovery_recipe_results.json)
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

A_FIX = 2e-2
MUS = (6.0e-4, 8.0e-4, 1.0e-3, 1.3e-3, 1.6e-3)
EDGE = 20


def _subsets(mode, n):
    return {"carrier only": (mode,),
            "carrier +/- 1": tuple(i for i in range(mode - 1, mode + 2) if 0 <= i < n),
            "carrier +/- 2": tuple(i for i in range(mode - 2, mode + 3) if 0 <= i < n),
            "carrier +/- 4": tuple(i for i in range(mode - 4, mode + 5) if 0 <= i < n),
            "all modes": tuple(range(n))}


def one_mu(mu, A=A_FIX, mode=bm.MODE):
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A, mode)
    tt = np.arange(0.0, bm.SPAN, bm.DT)
    aa, vv, _ = L.decay(y0, tt, rtol=bm.RTOL)
    Ei = 0.5 * (vv ** 2 + (L.wn ** 2)[None, :] * aa ** 2)
    k = len(tt) // EDGE
    sl = slice(k, len(tt) - k)

    env = np.abs(hilbert(aa[:, mode]))
    r_env = float(-2 * np.polyfit(tt[sl], np.log(env[sl]), 1)[0] / L.ci[mode])

    gamma, vacuous = {}, {}
    for nm, ss in _subsets(mode, L.n).items():
        Es = Ei[:, list(ss)].sum(axis=1)
        g = float(-np.polyfit(tt[sl], np.log(Es[sl]), 1)[0])
        gamma[nm] = g / L.ci[mode]
        f = Ei[:, mode] / Es
        dl = float(np.polyfit(tt[sl], np.log(f[sl]), 1)[0])
        vacuous[nm] = (g - dl) / L.ci[mode]      # identically the single-mode estimate
    return dict(mu=float(mu), r_env_single_mode=r_env,
                gamma_over_ctrue=gamma, vacuous_correction=vacuous)


def main():
    recs = [one_mu(m) for m in MUS]
    names = list(_subsets(bm.MODE, bm.N))
    summary = {nm: dict(
        mean_abs_err=float(np.mean([abs(r["gamma_over_ctrue"][nm] - 1) for r in recs])),
        worst_abs_err=float(np.max([abs(r["gamma_over_ctrue"][nm] - 1) for r in recs])))
        for nm in names}
    spread = float(np.max([max(r["vacuous_correction"].values())
                           - min(r["vacuous_correction"].values()) for r in recs]))
    res = dict(config=dict(N=bm.N, j0=bm.J0, mode_1based=bm.MODE + 1,
                           zeta1=bm.ZETA1, A_fixed=A_FIX, span=bm.SPAN,
                           dt=bm.DT, edge_trim=EDGE),
               records=recs, gamma_summary=summary,
               vacuity_check=dict(
                   max_spread_across_subsets=spread,
                   note=("gamma_S - dln f_S/dt is independent of the subset S to "
                         "numerical precision, confirming it is identically the "
                         "single-mode estimate and carries no information")))
    out = os.path.join(HERE, "..", "..", "checkpoints", "recovery_recipe_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}")
    for nm in names:
        print(f"  {nm:>16}: mean {summary[nm]['mean_abs_err']:.4f}, "
              f"worst {summary[nm]['worst_abs_err']:.4f}")
    print(f"  vacuity check, max spread across subsets: {spread:.2e}")


if __name__ == "__main__":
    main()
