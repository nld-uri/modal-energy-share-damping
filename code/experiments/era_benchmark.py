"""ERA on the same controlled-release records as bias_map.py, below AND above onset.

Paper 3, Sections 5-6. Every record of bias_map.py's grid and onset sweep is
regenerated (settle under resonant forcing, release, 400 time units of free
decay) and given to the eigensystem realization algorithm (identification/era.py)
alongside the three estimates of bias_map._three_estimates.

ERA input: PHYSICAL displacements x = Phi a of all 24 masses, decimated by
DEC, with no mode shapes; the pole nearest the tracked mode's linear frequency
is taken. The run that first produced the manuscript's ERA numbers fed it the
modal coordinates a instead; since Phi is orthogonal (unit masses) the two are
the same realization, and the difference is stored per record as a check.

Above the onset the record is no longer single-mode, and the question is
whether a realization algorithm, which fits one linear model to the whole
record, returns a stable rate at all. Every record is therefore realized at
orders 6-16 and two Hankel sizes (a stabilization table); the pole taken at
each setting is the largest-contribution pole within 5 % of the tracked mode's
linear frequency (era.era_damping_dominant). The reported setting is order 8,
200 block rows, the one first used; r_era_nearest repeats it with the original
nearest-frequency selection, and the two agree wherever a single pole exists.

    cd code/experiments && python era_benchmark.py   (writes era_benchmark_results.json)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "identification"))
sys.path.insert(0, HERE)
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402
from era import era_damping, era_damping_dominant, era_modes  # noqa: E402

DEC = 10                       # decimation: dt_era = 0.5
BASE = dict(n_modes=8, r=200, s=200)
ORDERS = (6, 7, 8, 9, 10, 12, 14, 16)
HANKEL = (200, 300)
BAND = 0.05


def one(args):
    group, k3, A = args
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=k3, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A, bm.MODE)
    if y0 is None:
        return None
    aa, vv, ac = L.decay(y0, np.arange(0.0, bm.SPAN, bm.DT), rtol=bm.RTOL)
    out = bm._three_estimates(L, aa, vv, ac, bm.MODE)
    ct, wt = L.ci[bm.MODE], L.wn[bm.MODE]
    x = aa @ L.Phi.T                                  # physical displacements
    dte = bm.DT * DEC
    c, w, sh = era_damping_dominant(x[::DEC], dte, wt, band=BAND, **BASE)
    c_near, _ = era_damping(x[::DEC], dte, wt, **BASE)
    c_modal, _ = era_damping(aa[::DEC], dte, wt, **BASE)
    stab = {}
    for h in HANKEL:
        for o in ORDERS:
            cc, ww, ss = era_damping_dominant(x[::DEC], dte, wt, band=BAND, n_modes=o, r=h, s=h)
            stab[f"{o}_{h}"] = dict(r=cc / ct, stiff_pct=100 * (ww / wt - 1), share=ss)
    # poles the order-12 realization places within 1 % of the tracked mode, and
    # how many carry more than a tenth of the largest one's contribution
    md = era_modes(x[::DEC], dte, n_modes=12, r=200, s=200)
    near = np.flatnonzero(np.abs(md["wn"] / wt - 1) < 0.01)
    cn = md["contrib"][near]
    out["o12_poles_near"] = [float(v) for v in md["wn"][near]]
    out["o12_significant"] = int((cn > 0.1 * cn.max()).sum()) if cn.size else 0
    out.update(group=group, k3=float(k3), A=float(A), mu=float(k3 * A ** 2),
               r_era=c / ct, era_stiff_pct=100 * (w / wt - 1), era_pole_share=sh,
               r_era_nearest=c_near / ct, r_era_modal_input=c_modal / ct,
               stabilization=stab)
    return out


def main():
    jobs = [("grid", k3, A) for k3 in bm.K3S for A in bm.AMPS]
    jobs += [("threshold_sweep", mu / bm.A_THRESHOLD ** 2, bm.A_THRESHOLD)
             for mu in bm.THRESHOLD_MUS]
    with Pool(min(8, os.cpu_count() or 1)) as p:
        rows = [r for r in p.map(one, jobs) if r is not None]
    res = dict(config=dict(dec=DEC, base=BASE, orders=ORDERS, hankel=HANKEL, band=BAND, span=bm.SPAN,
                           dt=bm.DT, input="physical displacement, all 24 masses",
                           pole="largest contribution within band of the tracked linear frequency"),
               records=rows)
    out = os.path.join(HERE, "..", "..", "checkpoints", "era_benchmark_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}  ({len(rows)} records)")


if __name__ == "__main__":
    main()
