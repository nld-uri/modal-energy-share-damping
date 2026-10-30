"""gamma_tot/c_true for the CONSERVATIVE CUBIC, with friction_boundary.py's own estimator.

The paper states this quantity two ways that cannot both be right: the body says
"gamma_tot/c_true = 0.999 at every amplitude", the caption of Fig. friction says the
total-energy decay "returns the true modal damping to six decimals". No stored checkpoint
carries the cubic arm, so it is recomputed here on the same protocol as the hysteretic arm
(settle under a tone at the tracked linear frequency, then fit log(Etot) over the free decay
with the same EDGE trim), the only change being a grounded cubic in place of the Bouc-Wen
element.
"""
import json, os, sys
import numpy as np
from scipy.integrate import solve_ivp

P3 = sys.argv[1]
sys.path.insert(0, os.path.join(P3, "code/experiments"))
sys.path.insert(0, os.path.join(P3, "code/models"))
sys.path.insert(0, os.path.join(P3, "code/identification"))
import lattice_nl as ln
import bias_map as bm

T_SETTLE, SPAN, DT, RTOL, EDGE = 4000.0, 400.0, 0.05, 1e-9, 20
MODE = bm.MODE


def run(k3, A):
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=k3, zeta1=bm.ZETA1)
    n, W2, c, psi = bm.N, L.wn ** 2, float(L.ci[0]), L.psi

    def f(t, y, A=0.0, Om=0.0):
        p, v = y[:n], y[n:]
        w = psi @ p
        acc = -c * v - W2 * p - psi * (k3 * w ** 3) + (psi * (A * np.cos(Om * t)) if A else 0.0)
        return np.concatenate([v, acc])

    s = solve_ivp(lambda t, y: f(t, y, A, L.wn[MODE]), (0.0, T_SETTLE), np.zeros(2 * n),
                  method="DOP853", rtol=RTOL, atol=RTOL * 1e-2)
    tt = np.arange(0.0, SPAN, DT)
    d = solve_ivp(f, (0.0, SPAN), s.y[:, -1], method="DOP853",
                  rtol=RTOL, atol=RTOL * 1e-2, t_eval=tt)
    pp, vv = d.y[:n].T, d.y[n:].T
    Ei = 0.5 * (vv ** 2 + W2[None, :] * pp ** 2)
    Etot = Ei.sum(axis=1)
    k = len(tt) // EDGE
    sl = slice(k, len(tt) - k)
    return dict(k3=k3, A=A, mu=k3 * A ** 2,
                share=float(Ei[:, MODE].mean() / Etot.mean()),
                gamma_tot_over_c=float(-np.polyfit(tt[sl], np.log(Etot[sl]), 1)[0] / c))


out = [run(k3, A) for k3, A in ((0.0, 0.02), (1.0, 0.02), (2.0, 0.02), (4.0, 0.02),
                                (4.0, 0.01), (4.0, 0.005))]
for r in out:
    print("  k3 %-4.1f A %-6.3f mu %9.2e  share %.4f  gamma_tot/c = %.6f  (|1-x| = %.2e)"
          % (r["k3"], r["A"], r["mu"], r["share"], r["gamma_tot_over_c"],
             abs(1 - r["gamma_tot_over_c"])), flush=True)
g = [r["gamma_tot_over_c"] for r in out]
print("\n  range %.6f to %.6f | rounded to 3 dp: %s | matches 1 to six decimals: %s"
      % (min(g), max(g), sorted({round(x, 3) for x in g}),
         all(abs(1 - x) < 5e-7 for x in g)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "cubic_gamma_tot.json"), "w"), indent=1)
