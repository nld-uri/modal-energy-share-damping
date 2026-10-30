"""Step 2, second pass: preparations that actually place the state on the target orbit.

The first pass (protocols_release_records.py -> *_fromrest.json) settled the
lattice FROM REST under the fixed phase-resonance force (Omega_pr, A_pr). For
kappa >= ~0.06 that lands on a different (lower-amplitude) solution of the
hardening response: distance from the PR orbit at release 0.2 -> 1.0. So those
records do not test the PR boundary. Three preparations here, zeta1 = 0.010:

  PR-ramp : start on the exact PR orbit at kappa = 0.005, then sweep (Omega, A)
            along the PR family, linear in kappa, to the target over T_RAMP,
            then hold the PR force for T_SETTLE = 6000; release.  (A slow
            upward-swept sine: realisable without a phase-locked loop.)
  PR-hold : start on the exact PR orbit at the target + a fixed perturbation
            (every modal displacement and velocity offset by EPS * X_7 * r_i,
            r_i ~ N(0,1), seed 0), hold the PR force for T_SETTLE; release.
  LIN-hold: identical construction on the exact omega_lin orbit (A = 2e-2).
The from-rest LIN records (the paper's protocol) are carried over from pass 1.

Per record: kappa of the PRE-RELEASE state (k3 max|w| over the last forcing
period), distance from the target orbit at release, bias_map._three_estimates,
total-energy decay rate over all modes, energy destination.
"""
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import protocols_core as pc  # noqa: E402
import protocols_release_records as r1  # noqa: E402
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

OUTDIR = r1.OUTDIR
ZETA, M, A_LIN = r1.ZETA, r1.M, r1.A_LIN
KAPPAS = r1.KAPPAS
T_RAMP = 4000.0
EPS = 1e-3
KAP0 = 0.005


def pr_family():
    d = json.load(open(os.path.join(OUTDIR, "protocols_phase_resonance.json")))
    sw = [s for s in d["sweeps"] if abs(s["zeta1"] - ZETA) < 1e-12][0]
    rr = sorted(sw["records"], key=lambda r: r["kappa"])
    return (np.array([r["kappa"] for r in rr]), np.array([r["Omega"] for r in rr]),
            np.array([r["A"] for r in rr]), d)


def forced_rhs_phase(L, A_of_t, Om_of_t):
    """Force A(t) cos(theta), theta' = Omega(t); theta appended as last state."""
    n, W2, C, PSI, k3 = L.n, L.wn ** 2, L.ci, L.psi, L.k3

    def f(t, z):
        a, v, th = z[:n], z[n:2 * n], z[2 * n]
        w = PSI @ a
        return np.concatenate([v, -C * v - W2 * a - PSI * (k3 * w ** 3)
                               + PSI * (A_of_t(t) * np.cos(th)), [Om_of_t(t)]])
    return f


def last_period_kappa(L, A, Om, y, t0):
    T = 2 * np.pi / Om
    s = solve_ivp(L.rhs_forced(A, Om), (t0, t0 + T), y, method="DOP853", rtol=1e-11,
                  atol=1e-14, t_eval=np.linspace(t0, t0 + T, 801))
    return float(L.k3 * np.max(np.abs(s.y[:L.n].T @ L.psi)) ** 2)


def perturbation(L, y_orb):
    rng = np.random.default_rng(0)
    X7 = np.hypot(y_orb[M], y_orb[L.n + M] / L.wn[M])
    d = EPS * X7 * rng.standard_normal(2 * L.n)
    d[L.n:] *= L.wn                      # velocity offsets scaled to each mode
    return d


def hold(L, A, Om, y_start):
    s = solve_ivp(L.rhs_forced(A, Om), (0.0, bm.T_SETTLE), y_start, method="DOP853",
                  rtol=bm.RTOL, atol=bm.RTOL * 1e-2)
    return s.y[:, -1]


def job(args):
    prep, kappa = args
    kk, OO, AA, _ = pr_family()
    if prep in ("PR-ramp", "PR-hold"):
        L, Om, r, k, mx = r1.pr_orbit(kappa, json.load(open(os.path.join(
            OUTDIR, "protocols_phase_resonance.json"))))
        A, y_orb = r["A"], r["y0"]
    else:
        omr = json.load(open(os.path.join(HERE, "..", "..", "checkpoints",
                                          "onset_mechanism_results.json")))
        stored = [q for c in omr["converged"] if abs(c["zeta1"] - ZETA) < 1e-12
                  for q in c["records"]]
        L, Om, r, k, mx, mu = r1.lin_orbit(kappa, stored)
        A, y_orb = A_LIN, r["y0"]
    if prep == "PR-ramp":
        L0, Om0, r0, k0, _ = r1.pr_orbit(KAP0, json.load(open(os.path.join(
            OUTDIR, "protocols_phase_resonance.json"))))
        # map the start orbit (k3 = 1 throughout the PR family)
        kap_t = lambda t: KAP0 + (kappa - KAP0) * min(t / T_RAMP, 1.0)  # noqa: E731
        Om_t = lambda t: (np.interp(kap_t(t), kk, OO) if t < T_RAMP else Om)  # noqa: E731
        A_t = lambda t: (np.interp(kap_t(t), kk, AA) if t < T_RAMP else A)  # noqa: E731
        # interpolated family endpoints differ from the shot values by <1e-4;
        # blend them so Omega(t), A(t) are continuous at T_RAMP
        dOm0, dA0 = Om0 - np.interp(KAP0, kk, OO), r0["A"] - np.interp(KAP0, kk, AA)
        dOm1, dA1 = Om - np.interp(kappa, kk, OO), A - np.interp(kappa, kk, AA)
        Om_c = lambda t: (np.interp(kap_t(t), kk, OO) + dOm0 + (dOm1 - dOm0) * min(t / T_RAMP, 1))  # noqa: E731,E501
        A_c = lambda t: (np.interp(kap_t(t), kk, AA) + dA0 + (dA1 - dA0) * min(t / T_RAMP, 1))  # noqa: E731,E501
        Tend = T_RAMP + bm.T_SETTLE
        s = solve_ivp(forced_rhs_phase(L, A_c, Om_c), (0.0, Tend),
                      np.concatenate([r0["y0"], [0.0]]), method="DOP853",
                      rtol=bm.RTOL, atol=bm.RTOL * 1e-2)
        ys, th_end = s.y[:-1, -1], s.y[-1, -1]
        # the hold phase of the force: equivalent time origin for the orbit comparison
        t_equiv = (th_end % (2 * np.pi)) / Om
    else:
        ys = hold(L, A, Om, y_orb + perturbation(L, y_orb))
        t_equiv = bm.T_SETTLE
    yo = r1.orbit_state_at(L, A, Om, y_orb, t_equiv)
    k_rel = last_period_kappa(L, A, Om, ys, t_equiv)   # periodic-forcing time shift
    out = dict(preparation=prep, kappa_target=kappa, kappa_orbit=k, Omega=float(Om),
               Omega_over_wlin=float(Om / L.wn[M]), A=float(A), k3=float(L.k3),
               mu=float(L.k3 * A ** 2), orbit_max_multiplier=mx,
               orbit_residual=float(r["residual"]),
               kappa_pre_release=k_rel,
               release_to_orbit_reldist=float(np.linalg.norm(ys - yo) / np.linalg.norm(yo)),
               settled=r1.analyse(L, ys))
    print(f"{prep} kap={k:.4f} lam={mx:.5f} kap_rel={k_rel:.4f} "
          f"dist={out['release_to_orbit_reldist']:.2e} s7={out['settled']['share_7']:.4f} "
          f"renv={out['settled']['r_env']:.4f} rg={out['settled']['r_gcvd']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    jobs = [(p, k) for k in KAPPAS for p in ("PR-ramp", "PR-hold", "LIN-hold")]
    with Pool(10) as p:
        recs = p.map(job, jobs)
    fr = json.load(open(os.path.join(OUTDIR, "protocols_release_records_fromrest.json")))
    _, _, _, prj = pr_family()
    sw = [s for s in prj["sweeps"] if abs(s["zeta1"] - ZETA) < 1e-12][0]
    omr = json.load(open(os.path.join(HERE, "..", "..", "checkpoints",
                                      "onset_mechanism_results.json")))
    cz = [c for c in omr["converged"] if abs(c["zeta1"] - ZETA) < 1e-12][0]
    out = dict(config=dict(fr["config"], T_ramp=T_RAMP, eps=EPS, kappa_ramp_start=KAP0,
                           preparations=dict(
                               LIN_fromrest="paper protocol: rest -> A=2e-2 cos(omega_7 t) for 6000",
                               PR_fromrest="rest -> A_pr cos(Omega_pr t) for 6000 (does NOT reach the PR orbit for kappa>=0.06)",
                               PR_ramp="exact PR orbit at kappa 0.005, swept along PR family over T_ramp, held 6000",
                               PR_hold="exact PR orbit + eps perturbation, held 6000",
                               LIN_hold="exact omega_lin orbit + same eps perturbation, held 6000")),
               predicted_boundaries=dict(
                   PR_floquet_kappa=[b["kappa"] for b in sw["floquet_boundaries"]],
                   PR_floquet_pair=[b["pair_unstable_side"] for b in sw["floquet_boundaries"]],
                   PR_averaged=sw["averaged_windows_at_protocol_Omega"],
                   LIN_averaged=sw["averaged_windows_at_omega_lin"],
                   LIN_floquet_threshold_mu=cz["threshold_mu"],
                   LIN_floquet_kappa_bracket=[max(r["k3_wpeak2"] for r in cz["records"] if r["max_multiplier"] < 1 and r["k3_wpeak2"] < 1),
                                              min(r["k3_wpeak2"] for r in cz["records"] if r["max_multiplier"] > 1)]),
               records_fromrest=fr["records"], records=recs)
    json.dump(out, open(os.path.join(OUTDIR, "protocols_release_records.json"), "w"), indent=1)
    print("done")
