"""Step 1 of the protocols track: phase-resonance forced orbits and their Floquet stability.

For zeta1 in (0.005, 0.010, 0.020): sweep the forcing frequency Omega upward
along the damped backbone of mode 7; at each Omega Newton-shoot the
phase-resonant orbit (unknowns y0 and A), compute the monodromy, record
max|multiplier|, the leading-eigenvector modal content and kappa. Crossings of
|lam|=1 are refined by bisection in Omega. The averaged combination-resonance
margin is evaluated at each orbit's (Omega, kappa).

Writes protocols_phase_resonance.json (next to this script's ../../checkpoints).
"""
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import protocols_core as pc  # noqa: E402
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402

OUTDIR = os.path.abspath(os.path.join(HERE, "..", "..", "checkpoints"))
NNM_JSON = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTDIR, "nnm.json")
ZETAS = (0.005, 0.010, 0.020)
KAP_TARGETS = np.concatenate([np.linspace(0.005, 0.02, 4), np.linspace(0.0225, 0.06, 16),
                              np.linspace(0.064, 0.15, 22)])
M = bm.MODE


def backbone_omega(kap_targets, nnm):
    k = np.array([r["kappa"] for r in nnm]); w = np.array([r["omega"] for r in nnm])
    o = np.argsort(k); k, w = k[o], w[o]
    p = np.polyfit(k[-10:], w[-10:], 1)
    return np.where(kap_targets <= k[-1], np.interp(kap_targets, k, w), np.polyval(p, kap_targets))


def one(L, Om, y, A):
    r = pc.shoot_phase_resonance(L, Om, y, A, m=M)
    met = pc.orbit_metrics(L, r["A"], Om, r["y0"], m=M)
    mx, cont, ev = pc.floquet(r["M"], L.wn)
    marg = pc.averaged_pairs(L, Om, met["kappa"], exclude_mode=M)
    best = max(marg, key=marg.get)
    rec = dict(Omega=float(Om), Omega_over_wlin=float(Om / L.wn[M]), A=float(r["A"]),
               mu=float(L.k3 * r["A"] ** 2), residual=r["residual"],
               quad_residual=r["quad_residual"], newton_iters=r["newton_iters"],
               max_multiplier=mx, unstable=bool(mx > 1.0), content=cont,
               avg_best_pair=list(best), avg_best_margin=float(marg[best]),
               avg_margin_6_8=float(marg[(6, 8)]), avg_margin_5_9=float(marg[(5, 9)]),
               avg_unstable_pairs=[list(k) for k, v in marg.items() if v > 0], **met)
    return rec, r["y0"], r["A"]


def sweep(zeta):
    nnm = json.load(open(NNM_JSON))
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=1.0, zeta1=zeta)
    Oms = backbone_omega(KAP_TARGETS, nnm)
    recs, states = [], []
    y, A = pc.guess_state(L, Oms[0], KAP_TARGETS[0], m=M)
    for i, Om in enumerate(Oms):
        if i > 0:     # warm start: scale previous orbit to the target kappa
            s = np.sqrt(KAP_TARGETS[i] / recs[-1]["kappa"])
            y, A = states[-1][0] * s, states[-1][1] * s
        rec, y1, A1 = one(L, Om, y, A)
        recs.append(rec); states.append((y1, A1))
        print(f"z={zeta} Om={Om:.6f} kap={rec['kappa']:.5f} lam={rec['max_multiplier']:.6f} "
              f"pair={[c['mode'] for c in rec['content']['content'][:2]]} res={rec['residual']:.1e}",
              flush=True)
    # refine every |lam|=1 crossing by bisection in Omega
    bounds = []
    for i in range(len(recs) - 1):
        a, b = recs[i], recs[i + 1]
        if (a["max_multiplier"] - 1) * (b["max_multiplier"] - 1) < 0:
            lo, hi = (i, a), (i + 1, b)
            ya, Aa = states[i]
            Olo, Ohi, rlo, rhi = a["Omega"], b["Omega"], a, b
            for _ in range(7):
                Om = 0.5 * (Olo + Ohi)
                s = 1.0
                rec, y1, A1 = one(L, Om, ya, Aa)
                if (rec["max_multiplier"] - 1) * (rlo["max_multiplier"] - 1) > 0:
                    Olo, rlo, ya, Aa = Om, rec, y1, A1
                else:
                    Ohi, rhi = Om, rec
                recs.append(rec)
            f = (1 - rlo["max_multiplier"]) / (rhi["max_multiplier"] - rlo["max_multiplier"])
            bounds.append(dict(type="onset" if rlo["max_multiplier"] < 1 else "restabilisation",
                               kappa=float(rlo["kappa"] + f * (rhi["kappa"] - rlo["kappa"])),
                               kappa_bracket=[rlo["kappa"], rhi["kappa"]],
                               Omega=float(rlo["Omega"] + f * (rhi["Omega"] - rlo["Omega"])),
                               mu=float(rlo["mu"] + f * (rhi["mu"] - rlo["mu"])),
                               pair_unstable_side=[c["mode"] for c in
                                                   (rhi if rhi["max_multiplier"] > 1 else rlo)
                                                   ["content"]["content"][:2]]))
    recs.sort(key=lambda r: r["Omega"])
    # averaged-theory boundaries along the protocol's own Omega(kappa)
    kk = np.array([r["kappa"] for r in recs]); OO = np.array([r["Omega"] for r in recs])
    kd = np.linspace(kk.min(), kk.max(), 6000); Od = np.interp(kd, kk, OO)
    avg = {}
    for key in ((6, 8), (5, 9), (4, 10)):
        mg = np.array([pc.averaged_pairs(L, o, k, exclude_mode=M)[key] for o, k in zip(Od, kd)])
        un = kd[mg > 0]
        avg[f"{key[0]}-{key[1]}"] = None if len(un) == 0 else [float(un.min()), float(un.max())]
    # same averaged theory at Omega = omega_lin (cross-check vs. stored m13 numbers)
    avg_lin = {}
    for key in ((6, 8), (5, 9)):
        mg = np.array([pc.averaged_pairs(L, L.wn[M], k, exclude_mode=M)[key] for k in kd])
        un = kd[mg > 0]
        avg_lin[f"{key[0]}-{key[1]}"] = None if len(un) == 0 else [float(un.min()), float(un.max())]
    return dict(zeta1=zeta, c=float(L.ci[0]), omega_lin=float(L.wn[M]), records=recs,
                floquet_boundaries=bounds, averaged_windows_at_protocol_Omega=avg,
                averaged_windows_at_omega_lin=avg_lin,
                worst_residual=float(max(r["residual"] for r in recs)),
                worst_quad_residual=float(max(r["quad_residual"] for r in recs)))


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    with Pool(3) as p:
        res = p.map(sweep, ZETAS)
    out = dict(config=dict(N=bm.N, j0=bm.J0, mode_1based=M + 1, k3=1.0,
                           protocol="phase resonance: first harmonic of a_7 in quadrature with station force",
                           newton_tol=1e-12, integrator="DOP853 rtol 1e-12 atol 1e-14",
                           kappa_def="k3*max|psi.a|^2 over the orbit", mu_def="k3*A^2"),
               sweeps=res)
    json.dump(out, open(os.path.join(OUTDIR, "protocols_phase_resonance.json"), "w"), indent=1)
    print("done")
