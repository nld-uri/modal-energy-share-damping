"""Three GCVD readouts on paper 3's records, cell by cell.

  diagonal  characteristic values read off the diagonal of Sigma_2 (pre-2026)
  rotated   eigenvalues of Sigma_2, modes rotated by its eigenvectors
            (the version used to produce paper 3's stored results)
  final     eigenvalues of Sigma_2, modes left as the eigenvectors of Q, each
            eigenvalue assigned one-to-one to the Q-mode carrying most of its
            eigenvector's weight (the superseded readout: shapes left as the eigenvectors of Q, each
       eigenvalue assigned one-to-one to the Q-mode carrying most of its
       eigenvector's weight; the programme has since adopted the canonical
       regression shapes Phi = V T^-H, under which no assignment heuristic
       is needed)

For every record of the 5x5 grid and the window-span sweep, reports
c_gcvd/c_true under each readout, the frequency difference between readouts,
the off-diagonal leakage of Sigma_2 relative to its diagonal, and the
pole-to-mode assignment share. Writes checkpoints/readout_final_check.json.

    cd checkpoints && OMP_NUM_THREADS=1 PYTHONPATH=../code/models:../code/identification:../code/experiments \
        python ../code/experiments/readout_final_check.py
"""
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

import gcvd as gc
import bias_map as bmp

LOG = []
# paths are taken from the location of this file, so the stage runs from any working
# directory (run_paper3.py starts stages in code/experiments, not in checkpoints)
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")


def _core(disp, vel, acc, center):
    X1, X2 = np.hstack([disp, vel]), np.hstack([vel, acc])
    if center:
        X1, X2 = X1 - X1.mean(0), X2 - X2.mean(0)
    K11, K12, K21, K22 = X1.T @ X1, X1.T @ X2, X2.T @ X1, X2.T @ X2
    Q = 0.25 * (gc._rdiv(K21, K11) + gc._rdiv(K22, K12) + gc._rdiv(K11, K21) + gc._rdiv(K12, K22))
    _, V = np.linalg.eig(Q)
    VmH = np.linalg.inv(V).conj().T
    Xi = X1 @ VmH
    return V, VmH, Xi, np.linalg.pinv(Xi) @ (X2 @ VmH)


def _assign(lam, T):
    P = np.abs(T) ** 2
    P = P / P.sum(0, keepdims=True)
    r, c = linear_sum_assignment(-P)
    out = np.empty_like(lam); sh = np.empty(len(lam))
    out[r] = lam[c]; sh[r] = P[r, c]
    return out, sh


def make(kind):
    def g(disp, vel, acc, center=True):
        V, VmH, Xi, S2 = _core(disp, vel, acc, center)
        d = np.diag(S2).copy()
        lam, T = np.linalg.eig(S2)
        la, sh = _assign(lam, T)
        off = np.abs(S2 - np.diag(d)).max(1) / np.abs(d)
        good = np.abs(d) > 1e-12 * np.abs(d).max()
        LOG.append(dict(share_min=float(sh.min()),
                        freq_diff=float(np.max(np.abs(np.abs(la[good]) / np.abs(d[good]) - 1))),
                        leak_min=float(off[good].min()), leak_max=float(off[good].max())))
        if kind == "diagonal":
            return d, V, Xi
        if kind == "final":
            return la, V, Xi
        W = VmH @ T
        return lam, np.linalg.inv(W).conj().T, Xi @ T
    return g


def cell(args):
    kind, k3, A, span = args
    gc.gcvd = make(kind)
    LOG.clear()
    r = bmp.controlled_release(k3, A, span=span)
    lg = LOG[-1] if LOG else {}
    return dict(kind=kind, k3=k3, A=A, span=span, r_gcvd=float(r["r_gcvd"]), **lg)


def main():
    B = json.load(open(os.path.join(CK, "bias_map_results.json")))
    recs = [("grid", g["k3"], g["A"], g["span"]) for g in B["grid"]] + \
           [("threshold", g["k3"], g["A"], g["span"]) for g in B["threshold_sweep"]] + \
           [("span", s["k3"], s["A"], s["span"]) for s in B["span_sweep"]]
    efrac = [g["e_frac"] for g in B["grid"] + B["threshold_sweep"]] + [None] * len(B["span_sweep"])
    jobs = [(kind, k3, A, sp) for kind in ("diagonal", "rotated", "final") for _, k3, A, sp in recs]
    with Pool(10) as p:
        res = p.map(cell, jobs)
    n = len(recs)
    out = dict(records=[dict(set=s, k3=k3, A=A, span=sp, e_frac=ef)
                        for (s, k3, A, sp), ef in zip(recs, efrac)])
    for i, (kind) in enumerate(("diagonal", "rotated", "final")):
        out[kind] = res[i * n:(i + 1) * n]
    stored = [g["r_gcvd"] for g in B["grid"] + B["threshold_sweep"]] + [s["r_gcvd"] for s in B["span_sweep"]]
    sm = np.array([ef is not None and ef > 0.99 for ef in efrac])
    for kind in ("diagonal", "rotated", "final"):
        e = np.array([abs(x["r_gcvd"] - 1) for x in out[kind]])
        out[f"{kind}_single_mode_worst"] = float(e[sm].max())
        print(f"{kind:9s} single-mode cells ({int(sm.sum())}): worst {100*e[sm].max():.2f}%")
    for kind in ("diagonal", "rotated", "final"):
        e = np.array([abs(x["r_gcvd"] - 1) for x in out[kind]])
        sets = np.array([r["set"] for r in out["records"]])
        print(f"{kind:9s} grid: median {100*np.median(e[sets=='grid']):.3f}%  worst {100*e[sets=='grid'].max():.2f}%   "
              f"span sweep: worst {100*e[sets=='span'].max():.2f}%   all: worst {100*e.max():.2f}%")
    rot = np.array([x["r_gcvd"] for x in out["rotated"]])
    fin = np.array([x["r_gcvd"] for x in out["final"]])
    print("rotated vs stored max |diff|", float(np.max(np.abs(rot - np.array(stored)))))
    print("final vs rotated max |diff|", float(np.max(np.abs(fin - rot))))
    L = out["final"]
    print("assignment share min", min(x["share_min"] for x in L),
          " freq diff diag vs eig max", max(x["freq_diff"] for x in L),
          " leakage range", min(x["leak_min"] for x in L), max(x["leak_max"] for x in L))
    json.dump(out, open(os.path.join(CK, "readout_final_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
