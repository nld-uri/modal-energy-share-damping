"""A second, deliberately unfavourable configuration on the same chain.

Every quantitative result in the paper comes from one (station, mode) choice:
the cubic spring at mass 2 and mode 7, for which psi_7 = 0.2778 is near the
largest coupling the station offers. That choice was made because it shows the
effect clearly, which is a reason to ask whether the effect is a property of
the choice.

This runs the same protocol on the same chain at the same station, tracking
mode 11 instead, for which psi_11 = 0.1041 -- 0.37 times the coupling, so the
nonlinear drive of the tracked mode is weaker by an order of magnitude in
energy at equal amplitude. If the paper's three claims are properties of the
system rather than of the configuration, then on this configuration too:

  1. below the onset both estimates return the true damping coefficient;
  2. the onset is located by the Floquet threshold of the periodic orbit,
     with no identification in the loop;
  3. above it the estimand is recovered by summing the carrier with the pair
     the leading Floquet eigenvector names.

Nothing here is tuned to make those come out: the mu grid is set once from the
Floquet threshold and the same estimators are applied.

Usage:  python second_configuration.py   (writes second_configuration_results.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("models", "identification", "experiments"):
    sys.path.insert(0, os.path.join(HERE, "..", _sub))
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402
import onset_mechanism as om  # noqa: E402

J0 = 1          # same station as the main configuration
MODE = 10       # 0-based; mode 11, against mode 7 in the main configuration
A_FIX = 2e-2
MUS_FLOQ = (4.0e-4, 8.0e-4, 1.2e-3, 1.6e-3, 2.0e-3, 2.4e-3, 3.2e-3, 4.0e-3)
MUS_GRID = (1.0e-4, 4.0e-4, 8.0e-4, 1.2e-3, 1.6e-3, 2.0e-3, 2.4e-3, 3.2e-3)


def floquet_threshold(mus=MUS_FLOQ):
    rows = []
    for mu in mus:
        r = om.floquet_true_orbit(mu, A=A_FIX, mode=MODE, j0=J0)
        rows.append(r)
        c = r["unstable_content"]["content"][:2]
        print(f"    floquet mu={mu:.1e}  |lam|={r['max_multiplier']:.5f}  "
              + ", ".join(f"m{d['mode']}:{d['share']:.3f}" for d in c), flush=True)
    mx = [r["max_multiplier"] for r in rows]
    thr = None
    for i in range(len(mus) - 1):
        if (mx[i] - 1.0) * (mx[i + 1] - 1.0) < 0:
            f = (1.0 - mx[i]) / (mx[i + 1] - mx[i])
            thr = float(np.exp(np.log(mus[i])
                               + f * (np.log(mus[i + 1]) - np.log(mus[i]))))
            break
    return rows, thr


def mu_sweep(mus=MUS_GRID):
    out = []
    for mu in mus:
        c = bm.controlled_release(mu / A_FIX ** 2, A_FIX, mode=MODE, j0=J0)
        if c is None:
            continue
        c["mu"] = float(mu)
        out.append(c)
        print(f"    sweep mu={mu:.1e}  share={c['e_frac']:.4f}  "
              f"env={c['r_env']:.4f}  gcvd={c['r_gcvd']:.4f}  "
              f"R2={c['env_r2']:.5f}", flush=True)
    return out


def recipe(mu, pair):
    """Recover the estimand from summed modal energy over widening subsets."""
    L = ln.Lattice(n=bm.N, j0=J0, k3=mu / A_FIX ** 2, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A_FIX, MODE)
    tt = np.arange(0.0, bm.SPAN, bm.DT)
    aa, vv, _ = L.decay(y0, tt, rtol=bm.RTOL)
    Ei = 0.5 * (vv ** 2 + (L.wn ** 2)[None, :] * aa ** 2)
    k = int(round(bm.TRIM / bm.DT))
    sl = slice(k, len(tt) - k)
    subsets = {"carrier only": [MODE],
               "carrier +/- 1": [MODE - 1, MODE, MODE + 1],
               f"carrier and the pair {pair}": sorted({MODE, *pair}),
               "carrier +/- 2": [MODE - 2, MODE - 1, MODE, MODE + 1, MODE + 2],
               "all modes": list(range(L.n))}
    out = []
    for name, idx in subsets.items():
        E = Ei[:, [i for i in idx if 0 <= i < L.n]].sum(axis=1)
        g = -np.polyfit(tt[sl], np.log(E[sl]), 1)[0]
        out.append(dict(subset=name, modes=[i + 1 for i in idx],
                        gamma_over_ctrue=float(g / L.ci[MODE])))
        print(f"    recipe {name:32s} {out[-1]['gamma_over_ctrue']:.4f}", flush=True)
    return out


def main():
    L = ln.Lattice(n=bm.N, j0=J0, k3=0.0, zeta1=bm.ZETA1)
    psi_main = float(abs(L.psi[bm.MODE]))
    psi_this = float(abs(L.psi[MODE]))
    print(f"configuration: station {J0 + 1}, mode {MODE + 1}, "
          f"psi = {psi_this:.4f} against {psi_main:.4f} for the main "
          f"configuration ({psi_this / psi_main:.2f}x)", flush=True)

    floq, thr = floquet_threshold()
    print(f"  Floquet threshold mu = "
          f"{'none in range' if thr is None else f'{thr:.3e}'}", flush=True)
    sweep = mu_sweep()

    pair = sorted(d["mode"] - 1 for d in
                  floq[-1]["unstable_content"]["content"][:2])
    above = [c for c in sweep if thr is not None and c["mu"] > 2 * thr]
    rec = recipe(above[-1]["mu"], pair) if above else []

    below = [c for c in sweep if thr is None or c["mu"] < thr]
    out = dict(config=dict(n=bm.N, j0=J0, mode_1based=MODE + 1, A=A_FIX,
                           zeta1=bm.ZETA1, span=bm.SPAN, dt=bm.DT,
                           psi_tracked=psi_this, psi_main_config=psi_main,
                           psi_ratio=psi_this / psi_main,
                           omega_tracked=float(L.wn[MODE])),
               floquet=floq, floquet_threshold_mu=thr,
               sweep=sweep,
               unstable_pair_1based=[p + 1 for p in pair],
               recipe=rec,
               below_onset=dict(
                   n=len(below),
                   worst_env=float(max((abs(c["r_env"] - 1) for c in below),
                                       default=float("nan"))),
                   worst_gcvd=float(max((abs(c["r_gcvd"] - 1) for c in below),
                                        default=float("nan")))))
    dst = os.path.join(HERE, "..", "..", "checkpoints",
                       "second_configuration_results.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nbelow onset: {out['below_onset']['n']} cells, worst envelope "
          f"{out['below_onset']['worst_env']:.4f}, worst decomposition "
          f"{out['below_onset']['worst_gcvd']:.4f}")
    print(f"wrote {os.path.normpath(dst)}")


if __name__ == "__main__":
    main()
