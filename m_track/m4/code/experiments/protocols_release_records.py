"""Step 2 of the protocols track: release records under phase-resonance preparation,
side by side with the omega_lin protocol at the SAME kappa (zeta1 = 0.010).

Phase-resonance (PR) protocol: for a target kappa, the phase-resonant orbit
(Omega_pr, A_pr) is Newton-shot (protocols_core); the lattice (k3 = 1) is then
settled FROM REST under A_pr cos(Omega_pr t) for T_SETTLE = 6000, released, and a
400-unit decay at dt = 0.05 is analysed with bias_map._three_estimates.
omega_lin (LIN) protocol: Omega = omega_7, A = 2e-2, k3 = mu/A^2 with mu chosen
by secant so the Newton-shot orbit has the target kappa; same settle/release.
Extra per record: distance of the settled state from the shot orbit at the
release phase, energy destination, total-energy decay rate over all modes
(quadratic modal energies, and with the quartic potential k3 w^4/4 included),
and for PR a control release taken directly from the exact orbit.
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
import onset_mechanism as om  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

OUTDIR = os.path.abspath(os.path.join(HERE, "..", "..", "checkpoints"))
ZETA = 0.010
M = bm.MODE
A_LIN = 2e-2
KAPPAS = (0.010, 0.020, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.070, 0.080,
          0.090, 0.100, 0.120, 0.150)


def orbit_kappa(L, A, Om, y0):
    T = 2 * np.pi / Om
    sp = solve_ivp(L.rhs_forced(A, Om), (0.0, T), y0, method="DOP853", rtol=1e-12,
                   atol=1e-14, t_eval=np.linspace(0, T, 1601))
    return float(L.k3 * np.max(np.abs(sp.y[:L.n].T @ L.psi)) ** 2)


def orbit_state_at(L, A, Om, y0, t):
    T = 2 * np.pi / Om
    tau = t % T
    if tau == 0:
        return y0
    s = solve_ivp(L.rhs_forced(A, Om), (0.0, tau), y0, method="DOP853", rtol=1e-12, atol=1e-14)
    return s.y[:, -1]


def pr_orbit(kappa, pr_json):
    """Phase-resonant orbit with kappa = target (secant in Omega on the shot family)."""
    sw = [s for s in pr_json["sweeps"] if abs(s["zeta1"] - ZETA) < 1e-12][0]
    rr = sorted(sw["records"], key=lambda r: r["kappa"])
    kk = np.array([r["kappa"] for r in rr]); OO = np.array([r["Omega"] for r in rr])
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=1.0, zeta1=ZETA)
    Om = float(np.interp(kappa, kk, OO))
    y, A = pc.guess_state(L, Om, kappa, m=M)
    hist = []
    for _ in range(6):
        r = pc.shoot_phase_resonance(L, Om, y, A, m=M)
        k = orbit_kappa(L, r["A"], Om, r["y0"])
        hist.append((Om, k))
        if abs(k / kappa - 1) < 1e-4:
            break
        if len(hist) >= 2 and hist[-1][1] != hist[-2][1]:
            (O1, k1), (O2, k2) = hist[-2], hist[-1]
            Om = O2 + (kappa - k2) * (O2 - O1) / (k2 - k1)
        else:
            Om = float(np.interp(kappa, kk, OO)) * (1 + 1e-5)
        y, A = r["y0"] * np.sqrt(kappa / k), r["A"] * np.sqrt(kappa / k)
    mx = float(np.max(np.abs(np.linalg.eigvals(r["M"]))))
    return L, Om, r, k, mx


def lin_orbit(kappa, stored):
    """omega_lin orbit at A = 2e-2 with kappa = target (secant in log mu)."""
    rec = sorted(stored, key=lambda r: r["k3_wpeak2"])
    lk = np.log([r["k3_wpeak2"] for r in rec]); lm = np.log([r["mu"] for r in rec])
    p = np.polyfit(lk[:4], lm[:4], 1)       # small-kappa extrapolation (kappa ~ mu^2/3 law)
    mu = float(np.exp(np.interp(np.log(kappa), lk, lm) if np.log(kappa) >= lk[0]
                      else np.polyval(p, np.log(kappa))))
    hist = []
    for _ in range(8):
        L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A_LIN ** 2, zeta1=ZETA)
        Om = L.wn[M]
        s = solve_ivp(L.rhs_forced(A_LIN, Om), (0.0, 300.0), np.zeros(2 * L.n),
                      method="DOP853", rtol=1e-10, atol=1e-13)
        y0, res, Mono = om.shoot_orbit(L, A_LIN, Om, s.y[:, -1])
        k = orbit_kappa(L, A_LIN, Om, y0)
        hist.append((np.log(mu), np.log(k)))
        if abs(k / kappa - 1) < 1e-4:
            break
        if len(hist) >= 2:
            (m1, k1), (m2, k2) = hist[-2], hist[-1]
            mu = float(np.exp(m2 + (np.log(kappa) - k2) * (m2 - m1) / (k2 - k1)))
        else:
            mu = mu * (kappa / k) ** 1.0
    mx = float(np.max(np.abs(np.linalg.eigvals(Mono))))
    return L, Om, dict(y0=y0, A=A_LIN, residual=res, M=Mono), k, mx, mu


def analyse(L, y0):
    tt = np.arange(0.0, bm.SPAN, bm.DT)
    aa, vv, ac = L.decay(y0, tt, rtol=bm.RTOL)
    est = bm._three_estimates(L, aa, vv, ac, M)
    Ei = 0.5 * (vv ** 2 + (L.wn ** 2)[None, :] * aa ** 2)
    w = aa @ L.psi
    k = int(round(bm.TRIM / bm.DT)); sl = slice(k, len(tt) - k)
    Eq = Ei.sum(1); H = Eq + 0.25 * L.k3 * w ** 4
    pq, rq = np.polyfit(tt[sl], np.log(Eq[sl]), 1, full=True)[:2]
    pH = np.polyfit(tt[sl], np.log(H[sl]), 1)
    lgq = np.log(Eq[sl])
    r2q = float(1 - rq[0] / ((lgq - lgq.mean()) ** 2).sum())
    Em = Ei.mean(0); Em = Em / Em.sum()
    top = np.argsort(-Em)[:5]
    # energy-weighted mean damping over all modes, window mean (= c for uniform damping)
    est.update(r_Etot_quadratic=float(-pq[0] / L.ci[M]), r_Etot_quadratic_R2=r2q,
               r_Etot_with_quartic=float(-pH[0] / L.ci[M]),
               share_7=float(Em[M]), share_6_8=float(Em[5] + Em[7]),
               share_5_9=float(Em[4] + Em[8]),
               top_modes=[dict(mode=int(i) + 1, share=float(Em[i])) for i in top],
               decimated=dict(t=tt[::40].tolist(), log_env7=np.log(np.abs(
                   __import__("scipy.signal", fromlist=["hilbert"]).hilbert(aa[:, M])))[::40].tolist(),
                   share7=(Ei[:, M] / Eq)[::40].tolist()))
    return est


def settle(L, A, Om):
    s = solve_ivp(L.rhs_forced(A, Om), (0.0, bm.T_SETTLE), np.zeros(2 * L.n),
                  method="DOP853", rtol=bm.RTOL, atol=bm.RTOL * 1e-2)
    return s.y[:, -1]


def job(args):
    proto, kappa, pr_json, stored = args
    if proto == "PR":
        L, Om, r, k, mx = pr_orbit(kappa, pr_json)
        A, mu = r["A"], float(L.k3 * r["A"] ** 2)
    else:
        L, Om, r, k, mx, mu = lin_orbit(kappa, stored)
        A = A_LIN
    ys = settle(L, A, Om)
    yo = orbit_state_at(L, A, Om, r["y0"], bm.T_SETTLE)
    out = dict(protocol=proto, kappa_target=kappa, kappa_orbit=k, Omega=float(Om),
               Omega_over_wlin=float(Om / L.wn[M]), A=float(A), k3=float(L.k3), mu=mu,
               orbit_residual=float(r["residual"]), orbit_max_multiplier=mx,
               settled_to_orbit_reldist=float(np.linalg.norm(ys - yo) / np.linalg.norm(yo)),
               settled=analyse(L, ys))
    if proto == "PR":
        out["orbit_release"] = analyse(L, yo)
    print(f"{proto} kap={k:.4f} lam={mx:.5f} dist={out['settled_to_orbit_reldist']:.2e} "
          f"s7={out['settled']['share_7']:.4f} renv={out['settled']['r_env']:.4f} "
          f"rg={out['settled']['r_gcvd']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    pr_json = json.load(open(os.path.join(OUTDIR, "protocols_phase_resonance.json")))
    omr = json.load(open(os.path.join(HERE, "..", "..", "checkpoints",
                                      "onset_mechanism_results.json")))
    stored = [r for c in omr["converged"] if abs(c["zeta1"] - ZETA) < 1e-12
              for r in c["records"]]
    jobs = [(p, k, pr_json, stored) for k in KAPPAS for p in ("PR", "LIN")]
    with Pool(10) as p:
        recs = p.map(job, jobs)
    pr = [s for s in pr_json["sweeps"] if abs(s["zeta1"] - ZETA) < 1e-12][0]
    out = dict(config=dict(N=bm.N, j0=bm.J0, mode_1based=M + 1, zeta1=ZETA,
                           T_settle=bm.T_SETTLE, span=bm.SPAN, dt=bm.DT, trim=bm.TRIM,
                           A_lin=A_LIN, k3_pr=1.0),
               predicted_boundaries=dict(
                   PR_floquet=pr["floquet_boundaries"],
                   PR_averaged=pr["averaged_windows_at_protocol_Omega"],
                   LIN_averaged=pr["averaged_windows_at_omega_lin"]),
               records=recs)
    json.dump(out, open(os.path.join(OUTDIR, "protocols_release_records_fromrest.json"), "w"), indent=1)
    print("done")
