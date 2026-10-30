"""What sets the damping-bias onset: a sideband instability of the tracked NNM.

Paper 3, Section 1b.8.  `bias_map.py` locates an onset in the similarity
parameter mu = k3 A^2, between 4.0e-4 and 5.0e-4, past which the tracked mode
loses its share of the window energy and the identified damping departs from
the true value.  This script identifies the mechanism.

Three findings, each computed here:

  1. The energy goes to a SYMMETRIC SIDEBAND PAIR about the tracked mode
     (modes m-2 and m+2 for m = 7), whose frequencies satisfy
     omega_{m-2} + omega_{m+2} = 1.98423 omega_m, i.e. 2 omega_m detuned by
     -0.00789 (relative) -- a three-wave condition.
     It is not a harmonic resonance: 3 omega_7 lies outside the chain passband.

  2. The pair is NOT selected by detuning.  The k=1 pair (modes 6, 8) is four
     times closer to exact resonance (-0.0020 against -0.0079) and takes no
     energy, so the coupling coefficients decide.

  3. Above onset the sideband energy GROWS EXPONENTIALLY during the forced
     buildup and then saturates -- the signature of a modulational instability
     settling on a modulated branch -- while below onset it is merely driven at
     a non-growing level two decades lower.

Consequence for the paper: the onset is the NNM's stability boundary, so it is
predictable from a Floquet analysis of the periodic orbit rather than being an
empirical threshold.  Damping identifiability and nonlinear-normal-mode
stability are the same question.

Usage:  python onset_mechanism.py      (writes onset_mechanism_results.json)
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("models", "identification", "experiments"):
    sys.path.insert(0, os.path.join(HERE, "..", _sub))
import lattice_nl as ln  # noqa: E402
import bias_map as bm  # noqa: E402

A_FIX = 2e-2
# Straddle the CONVERGED Floquet boundary (3.85e-4 at the nominal zeta), not the
# bias-onset band: 4.0e-4 is above the boundary and does grow slowly, which made
# the growth panel contradict its own caption.
MU_BELOW, MU_ABOVE = 3.0e-4, 5.0e-4
MUS_DEST = (4.0e-4, 5.0e-4, 8.0e-4, 1.6e-3)
T_SETTLE = 6000.0


def spectrum_relations(n=bm.N, j0=bm.J0, mode=bm.MODE):
    """Sideband detunings and harmonic-resonance candidates about the tracked mode."""
    L = ln.Lattice(n=n, j0=j0, k3=1.0, zeta1=bm.ZETA1)
    wm = L.wn[mode]
    pairs = []
    for k in range(1, 6):
        lo, hi = mode - k, mode + k
        if lo < 0 or hi >= n:
            continue
        pairs.append(dict(k=k, modes=[lo + 1, hi + 1],
                          detuning=float((L.wn[lo] + L.wn[hi]) / (2 * wm) - 1),
                          omega_lo=float(L.wn[lo]), omega_hi=float(L.wn[hi]),
                          # mean modulation offset the pair imposes on the
                          # carrier, [(w_hi - w_m) + (w_m - w_lo)] / 2
                          mean_offset=float((L.wn[hi] - L.wn[lo]) / 2),
                          psi_lo=float(L.psi[lo]), psi_hi=float(L.psi[hi]),
                          psi_product=float(L.psi[lo] * L.psi[hi])))
    harm = [dict(p=p, q=q, mode=i + 1, ratio=float(p * L.wn[i] / (q * wm)))
            for i in range(n) for p, q in ((3, 1), (1, 3), (2, 1), (1, 2), (3, 2), (2, 3))
            if abs(p * L.wn[i] / (q * wm) - 1) < 0.06]
    return dict(omega_tracked=float(wm), third_harmonic=float(3 * wm),
                band_edge=float(L.wn[-1]),
                third_harmonic_in_band=bool(3 * wm <= L.wn[-1]),
                sideband_pairs=pairs, harmonic_candidates=harm)


def energy_destination(mu, A=A_FIX, mode=bm.MODE, n_top=4):
    """Where the window energy sits, once the state has been prepared at this mu."""
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A, mode)
    aa, vv, _ = L.decay(y0, np.arange(0.0, bm.SPAN, bm.DT), rtol=bm.RTOL)
    E = np.mean(0.5 * (vv ** 2 + (L.wn ** 2)[None, :] * aa ** 2), axis=0)
    E = E / E.sum()
    top = np.argsort(E)[::-1][:n_top]
    return dict(mu=float(mu), k3=float(mu / A ** 2),
                top=[dict(mode=int(t) + 1, share=float(E[t]),
                          ratio=float(L.wn[t] / L.wn[mode])) for t in top])


def sideband_growth(mu, A=A_FIX, mode=bm.MODE, dt_out=2.0, fit_window=(1000.0, 4000.0)):
    """Sideband energy history during forced buildup: growing, or merely driven?"""
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
    ts = np.arange(0.0, T_SETTLE, dt_out)
    s = solve_ivp(L.rhs_forced(A, L.wn[mode]), (0.0, T_SETTLE), np.zeros(2 * L.n),
                  method="DOP853", rtol=1e-11, atol=1e-14, t_eval=ts)
    a, v = s.y[:L.n].T, s.y[L.n:].T
    E = 0.5 * (v ** 2 + (L.wn ** 2)[None, :] * a ** 2)
    lo, hi = mode - 2, mode + 2
    seg = (ts > fit_window[0]) & (ts < fit_window[1])
    rates = {f"mode_{m + 1}": float(np.polyfit(ts[seg], np.log(E[seg, m]), 1)[0])
             for m in (lo, hi)}
    # Decimate to ~300 points and store the whole history, so the growth panel
    # of the stability figure is plotted from disk rather than recomputed.
    step = max(1, len(ts) // 300)
    hist = dict(t=[float(v) for v in ts[::step]],
                E_carrier=[float(v) for v in E[::step, mode]],
                E_lower=[float(v) for v in E[::step, lo]],
                E_upper=[float(v) for v in E[::step, hi]])
    return dict(mu=float(mu), sideband_modes=[lo + 1, hi + 1],
                log_slope_energy=rates, passive_rate=float(-L.ci[mode]),
                net_growth_upper=float(rates[f"mode_{hi + 1}"] + L.ci[mode]),
                history=hist)


def _monodromy(L, wq_of_tau, tau, T, return_spectrum=False):
    """Max Floquet multiplier of the linearisation about a periodic carrier.

    The cubic element modulates the tangent stiffness by 3 k3 (psi.q)^2 psi psi^T,
    which is periodic at the carrier period, so the sidebands see a parametric
    excitation.  No identification enters this computation.
    """
    n = L.n
    W2, C, PSI, k3 = L.wn ** 2, L.ci, L.psi, L.k3

    def rhs(t, M):
        M = M.reshape(2 * n, 2 * n)
        dK = 3 * k3 * np.interp(t % T, tau, wq_of_tau) ** 2
        return np.vstack([M[n:],
                          -(W2[:, None] * M[:n])
                          - dK * np.outer(PSI, PSI @ M[:n])
                          - C[:, None] * M[n:]]).ravel()

    s = solve_ivp(rhs, (0.0, T), np.eye(2 * n).ravel(), method="DOP853",
                  rtol=1e-10, atol=1e-12)
    ev, evec = np.linalg.eig(s.y[:, -1].reshape(2 * n, 2 * n))
    mx = float(np.max(np.abs(ev)))
    return (mx, ev, evec) if return_spectrum else mx


def unstable_mode_content(ev, evec, wn):
    """Modal energy distribution of the leading Floquet eigenvector.

    The monodromy does not merely say that the orbit is unstable; its leading
    eigenvector says WHICH degrees of freedom grow. In modal coordinates the
    displacement half of that vector, weighted into energy, is the selection
    rule itself -- no argument from detuning or from mode-shape overlap is
    needed, and any such argument has to agree with it.
    """
    j = int(np.argmax(np.abs(ev)))
    v = evec[:, j]
    n = len(wn)
    e = 0.5 * (np.abs(v[n:]) ** 2 + wn ** 2 * np.abs(v[:n]) ** 2)
    e = e / e.sum()
    order = np.argsort(-e)
    return dict(leading_multiplier_abs=float(abs(ev[j])),
                leading_multiplier_arg=float(np.angle(ev[j])),
                content=[dict(mode=int(i) + 1, share=float(e[i]))
                         for i in order[:6]],
                participation=float(1.0 / np.sum(e ** 2)))


def floquet_true_orbit(mu, A=A_FIX, mode=bm.MODE, t_probe=2000.0, npts=801,
                       zeta1=None, j0=None):
    """Stability of the actual settled orbit, probed before sidebands saturate."""
    L = ln.Lattice(n=bm.N, j0=bm.J0 if j0 is None else j0, k3=mu / A ** 2,
                   zeta1=bm.ZETA1 if zeta1 is None else zeta1)
    Om = L.wn[mode]
    T = 2 * np.pi / Om
    s = solve_ivp(L.rhs_forced(A, Om), (0.0, t_probe + T), np.zeros(2 * L.n),
                  method="DOP853", rtol=1e-11, atol=1e-14,
                  t_eval=np.linspace(t_probe, t_probe + T, npts))
    wq = s.y[:L.n].T @ L.psi
    tau = np.linspace(0.0, T, npts)
    mx, ev, evec = _monodromy(L, wq, tau, T, return_spectrum=True)
    w_pk = float(np.abs(wq).max())
    # Store the WHOLE spectrum, not just its maximum: 48 complex numbers per mu
    # is negligible on disk and it is what the stability figure plots, so the
    # figure never has to re-integrate the monodromy.
    order = np.argsort(-np.abs(ev))
    return dict(mu=float(mu), k3=float(L.k3), w_peak=w_pk,
                k3_wpeak2=float(L.k3 * w_pk ** 2), max_multiplier=mx,
                unstable=bool(mx > 1.0),
                growth_rate_energy=float(2 * np.log(mx) / T),
                crest_factor=float(w_pk / np.sqrt(np.mean(wq ** 2))),
                period=float(T),
                multipliers_re=[float(z.real) for z in ev[order]],
                multipliers_im=[float(z.imag) for z in ev[order]],
                unstable_content=unstable_mode_content(ev, evec, L.wn))


def floquet_single_harmonic(k3_wpeak2, k3=1.0, mode=bm.MODE, npts=801):
    """Same stability question under a purely sinusoidal carrier — a cross-check."""
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=k3, zeta1=bm.ZETA1)
    Om = L.wn[mode]
    T = 2 * np.pi / Om
    tau = np.linspace(0.0, T, npts)
    wq = np.sqrt(k3_wpeak2 / k3) * np.cos(Om * tau)
    mx = _monodromy(L, wq, tau, T)
    return dict(k3_wpeak2=float(k3_wpeak2), max_multiplier=mx,
                unstable=bool(mx > 1.0))


def shoot_orbit(L, A, Om, y0, iters=12, tol=1e-12):
    """Newton-shoot the forced period-T orbit, returning (y0, residual, M).

    Probing a settling transient is not the same as evaluating the periodic
    orbit. The forced response approaches the orbit on the decay time, which
    here is two orders longer than the period, so a probe taken at any finite
    time carries a return residual that is large compared with the margin the
    stability call turns on. Newton on the period map removes that entirely:

        F(y) = phi_T(y) - y,      dF/dy = M - I,

    where M is the monodromy of the FORCED system (the same variational
    integration the stability calculation needs anyway). Two or three
    iterations take the residual to integration tolerance, after which
    |lambda|max is a property of the orbit rather than of when it was probed.
    """
    n2 = 2 * L.n
    f = L.rhs_forced(A, Om)

    def step(y):
        def rhs(t, z):
            yy, M = z[:n2], z[n2:].reshape(n2, n2)
            a = yy[:L.n]
            dK = 3 * L.k3 * (L.psi @ a) ** 2
            J = np.zeros((n2, n2))
            J[:L.n, L.n:] = np.eye(L.n)
            J[L.n:, :L.n] = -(np.diag(L.wn ** 2) + dK * np.outer(L.psi, L.psi))
            J[L.n:, L.n:] = -np.diag(L.ci)
            return np.concatenate([f(t, yy), (J @ M).ravel()])

        z0 = np.concatenate([y, np.eye(n2).ravel()])
        s = solve_ivp(rhs, (0.0, 2 * np.pi / Om), z0, method="DOP853",
                      rtol=1e-12, atol=1e-14)
        yT = s.y[:n2, -1]
        M = s.y[n2:, -1].reshape(n2, n2)
        return yT, M

    for _ in range(iters):
        yT, M = step(y0)
        r = yT - y0
        res = float(np.linalg.norm(r) / max(np.linalg.norm(y0), 1e-300))
        if res < tol:
            break
        y0 = y0 - np.linalg.solve(M - np.eye(n2), r)
    yT, M = step(y0)
    res = float(np.linalg.norm(yT - y0) / max(np.linalg.norm(y0), 1e-300))
    return y0, res, M


def converged_threshold(mus=(2.8e-4, 3.0e-4, 3.2e-4, 3.4e-4, 3.6e-4, 4.0e-4),
                        A=A_FIX, mode=bm.MODE, t_seed=2000.0):
    """Threshold from Newton-shot orbits rather than from probed transients."""
    out = []
    for mu in mus:
        L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
        Om = L.wn[mode]
        T = 2 * np.pi / Om
        s = solve_ivp(L.rhs_forced(A, Om), (0.0, t_seed), np.zeros(2 * L.n),
                      method="DOP853", rtol=1e-11, atol=1e-14)
        y0, res, M = shoot_orbit(L, A, Om, s.y[:, -1])
        ev = np.linalg.eigvals(M)
        out.append(dict(mu=float(mu), return_residual=res,
                        max_multiplier=float(np.max(np.abs(ev))),
                        period=float(T)))
        print(f"    shot mu={mu:.2e}: residual {res:.2e}, "
              f"|lam|max {out[-1]['max_multiplier']:.6f}", flush=True)
    thr = None
    for a, b in zip(out, out[1:]):
        if (a["max_multiplier"] - 1) * (b["max_multiplier"] - 1) < 0:
            f_ = (1 - a["max_multiplier"]) / (b["max_multiplier"] - a["max_multiplier"])
            thr = float(np.exp(np.log(a["mu"])
                               + f_ * (np.log(b["mu"]) - np.log(a["mu"]))))
            break
    return dict(records=out, threshold_mu=thr,
                worst_residual=float(max(r["return_residual"] for r in out)))


def converged_zeta_sweep(zetas=(0.005, 0.010, 0.020),
                         mus=(1.0e-4, 1.5e-4, 2.0e-4, 2.5e-4,
                              3.0e-4, 3.4e-4, 3.6e-4, 3.8e-4, 4.0e-4,
                              5.0e-4, 6.0e-4, 8.0e-4, 1.0e-3, 1.2e-3, 1.6e-3,
                              2.0e-3),
                         A=A_FIX, mode=bm.MODE, t_seed=300.0):
    """Threshold and unstable eigenvector from Newton-shot periodic orbits.

    This supersedes the probed-transient version: see `orbit_accuracy` for why
    a probe cannot resolve the threshold, and the Newton residuals below
    (order 1e-13) for what replaces it.
    """
    out = []
    for z in zetas:
        rows = []
        for mu in mus:
            L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=z)
            Om = L.wn[mode]
            s = solve_ivp(L.rhs_forced(A, Om), (0.0, t_seed), np.zeros(2 * L.n),
                          method="DOP853", rtol=1e-10, atol=1e-13)
            y0, res, M = shoot_orbit(L, A, Om, s.y[:, -1])
            ev, evec = np.linalg.eig(M)
            order = np.argsort(-np.abs(ev))
            # Modal amplitude of the CONVERGED orbit over one period. The
            # single-harmonic comparison of Sec 6 needs k3*w_peak^2 from an
            # orbit that closes, not from a probe that has not settled.
            T_ = 2 * np.pi / Om
            sp = solve_ivp(L.rhs_forced(A, Om), (0.0, T_), y0, method="DOP853",
                           rtol=1e-11, atol=1e-14,
                           t_eval=np.linspace(0.0, T_, 801))
            wq = sp.y[:L.n].T @ L.psi
            w_pk = float(np.abs(wq).max())
            rows.append(dict(mu=float(mu), return_residual=res,
                             k3=float(L.k3), w_peak=w_pk,
                             k3_wpeak2=float(L.k3 * w_pk ** 2),
                             crest_factor=float(w_pk / np.sqrt(np.mean(wq ** 2))),
                             period=float(2 * np.pi / Om),
                             max_multiplier=float(np.max(np.abs(ev))),
                             unstable=bool(np.max(np.abs(ev)) > 1.0),
                             multipliers_re=[float(z.real) for z in ev[order]],
                             multipliers_im=[float(z.imag) for z in ev[order]],
                             content=unstable_mode_content(ev, evec, L.wn)))
        mx = [r["max_multiplier"] for r in rows]
        thr = None
        for i in range(len(mus) - 1):
            if (mx[i] - 1.0) * (mx[i + 1] - 1.0) < 0:
                f_ = (1.0 - mx[i]) / (mx[i + 1] - mx[i])
                thr = float(np.exp(np.log(mus[i])
                                   + f_ * (np.log(mus[i + 1]) - np.log(mus[i]))))
                break
        unst = [r for r in rows if r["max_multiplier"] > 1.0]
        out.append(dict(zeta1=float(z), threshold_mu=thr, records=rows,
                        worst_residual=float(max(r["return_residual"] for r in rows)),
                        n_unstable=len(unst)))
        print(f"    zeta1={z:.3f}: threshold mu = "
              f"{'none in range' if thr is None else f'{thr:.3e}'}"
              f"  (worst Newton residual {out[-1]['worst_residual']:.1e})",
              flush=True)
    return out


def orbit_accuracy(mus=(3.0e-4, 3.2e-4), A=A_FIX, mode=bm.MODE,
                   probes=(1000.0, 2000.0, 3000.0),
                   rtols=(1e-10, 1e-11, 1e-12), nptss=(401, 801, 1601)):
    """How well is the periodic orbit resolved, and does the boundary survive it?

    The threshold call turns on |lambda|max = 1.0030 against 0.99948 at
    adjacent mu, a margin of 0.3 per cent. That margin is only meaningful if
    the orbit the monodromy is linearized about is itself resolved to better
    than it. Two quantities settle that:

      * the Poincare return residual ||y(t+T) - y(t)|| / ||y(t)||, which says
        how nearly periodic the probed segment actually is (the forced
        response is still settling and the sidebands have not saturated, so
        it is not exactly periodic);
      * the spread of |lambda|max as the probe time, the integration
        tolerance and the number of samples per period are varied over
        ranges that change the residual by orders of magnitude.

    If the second is far smaller than the 0.3 per cent margin, the boundary
    call stands on the numerics; if not, it does not.
    """
    out = []
    for mu in mus:
        L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
        Om = L.wn[mode]
        T = 2 * np.pi / Om
        rows = []
        for t_probe in probes:
            for rtol in rtols:
                for npts in nptss:
                    s = solve_ivp(L.rhs_forced(A, Om), (0.0, t_probe + T),
                                  np.zeros(2 * L.n), method="DOP853",
                                  rtol=rtol, atol=rtol * 1e-3,
                                  t_eval=np.linspace(t_probe, t_probe + T, npts))
                    y0, yT = s.y[:, 0], s.y[:, -1]
                    res = float(np.linalg.norm(yT - y0) / np.linalg.norm(y0))
                    wq = s.y[:L.n].T @ L.psi
                    mx = _monodromy(L, wq, np.linspace(0.0, T, npts), T)
                    rows.append(dict(t_probe=float(t_probe), rtol=float(rtol),
                                     npts=int(npts), return_residual=res,
                                     max_multiplier=float(mx)))
        m = [r["max_multiplier"] for r in rows]
        res = [r["return_residual"] for r in rows]
        out.append(dict(mu=float(mu), n_variants=len(rows),
                        residual_min=float(min(res)), residual_max=float(max(res)),
                        multiplier_min=float(min(m)), multiplier_max=float(max(m)),
                        multiplier_spread=float(max(m) - min(m)),
                        unstable_unanimous=bool(all(v > 1.0 for v in m)
                                                or all(v <= 1.0 for v in m)),
                        variants=rows))
        print(f"    mu={mu:.1e}: residual {min(res):.2e}-{max(res):.2e}, "
              f"|lam|max {min(m):.6f}-{max(m):.6f} "
              f"(spread {max(m)-min(m):.2e}) over {len(rows)} variants",
              flush=True)
    return out


def threshold_vs_zeta(zetas=(0.005, 0.010, 0.020),
                      mus=(1.5e-4, 2.0e-4, 3.0e-4, 4.0e-4, 5.0e-4, 6.0e-4,
                           8.0e-4, 9.0e-4, 1.0e-3, 1.2e-3, 1.4e-3),
                      A=A_FIX, mode=bm.MODE):
    """Onset threshold as a function of modal damping.

    The instability is parametric, so the threshold is set by the competition
    between sideband growth and sideband damping and must depend on zeta as
    well as on mu. Reporting a single threshold from a single zeta leaves that
    dependence unmeasured. Here the same Floquet calculation is run at half
    and twice the nominal damping, and the crossing of |lambda| = 1 is located
    by linear interpolation in log mu.
    """
    out = []
    for z in zetas:
        rows = [floquet_true_orbit(m, A=A, mode=mode, zeta1=z) for m in mus]
        mx = [r["max_multiplier"] for r in rows]
        thr = None
        for i in range(len(mus) - 1):
            if (mx[i] - 1.0) * (mx[i + 1] - 1.0) < 0:
                f = (1.0 - mx[i]) / (mx[i + 1] - mx[i])
                thr = float(np.exp(np.log(mus[i])
                                   + f * (np.log(mus[i + 1]) - np.log(mus[i]))))
                break
        out.append(dict(zeta1=float(z), mus=[float(m) for m in mus],
                        max_multiplier=[float(v) for v in mx],
                        threshold_mu=thr,
                        content_at_top=rows[-1]["unstable_content"]))
        print(f"    zeta1={z:.3f}: threshold mu = "
              f"{'none in range' if thr is None else f'{thr:.3e}'}", flush=True)
    return out


def neimark_sacker(floq, spec, k=2):
    """Classify the crossing and compare its rotation number with the pair.

    A conjugate pair leaving the unit circle at finite argument is a
    Neimark-Sacker bifurcation to modulated motion; the modulation frequency is
    |arg lambda| / T.  If the sidebands are what goes unstable, that frequency
    must equal the mean offset of the k-th pair from the carrier.
    """
    rows = []
    for e in sorted(floq, key=lambda r: r["mu"]):
        lam = np.asarray(e["multipliers_re"]) + 1j * np.asarray(e["multipliers_im"])
        out = lam[np.abs(lam) > 1.0]
        if out.size == 0:
            continue
        arg = float(np.abs(np.angle(out)).mean())
        rows.append(dict(mu=e["mu"], n_unstable=int(out.size),
                         arg_over_pi=arg / np.pi, rotation=arg / e["period"]))
    pair = next(p for p in spec["sideband_pairs"] if p["k"] == k)
    rot = float(np.mean([r["rotation"] for r in rows])) if rows else float("nan")
    return dict(records=rows, n_unstable_max=max((r["n_unstable"] for r in rows),
                                                 default=0),
                mean_rotation=rot, pair_modes=pair["modes"],
                pair_mean_offset=pair["mean_offset"],
                rel_error=abs(rot / pair["mean_offset"] - 1.0))


def main():
    res = dict(config=dict(N=bm.N, j0=bm.J0, mode_1based=bm.MODE + 1,
                           zeta1=bm.ZETA1, A_fixed=A_FIX, t_settle=T_SETTLE),
               spectrum=spectrum_relations(),
               destinations=[energy_destination(m) for m in MUS_DEST],
               growth=[sideband_growth(m) for m in (MU_BELOW, MU_ABOVE)],
               floquet_true_orbit=[floquet_true_orbit(m) for m in
                                   (1.0e-4, 2.0e-4, 2.5e-4, 3.0e-4, 3.2e-4,
                                    3.4e-4, 3.6e-4, 3.8e-4, 4.0e-4, 5.0e-4,
                                    6.0e-4, 8.0e-4, 1.2e-3, 1.6e-3)],
               floquet_single_harmonic=[floquet_single_harmonic(k) for k in
                                        (0.01, 0.02, 0.04, 0.06, 0.08, 0.081,
                                         0.082, 0.083, 0.084, 0.085, 0.09,
                                         0.10, 0.12)])
    # The PRIMARY stability result is computed on Newton-shot periodic orbits.
    # The probed-transient sweep above is retained only because
    # `orbit_accuracy` needs it to show why probing cannot resolve the
    # threshold; no claim in the paper rests on it.
    res["converged"] = converged_zeta_sweep()
    nominal = next(b for b in res["converged"] if abs(b["zeta1"] - bm.ZETA1) < 1e-12)
    res["neimark_sacker"] = neimark_sacker(nominal["records"], res["spectrum"])
    # The scaling quantities the text quotes, stored rather than recomputed in
    # prose: an exponent worked out by hand is a number with no source.
    _z = np.array([b["zeta1"] for b in res["converged"]])
    _m = np.array([b["threshold_mu"] for b in res["converged"]])
    res["threshold_scaling"] = dict(
        zeta1=[float(v) for v in _z], threshold_mu=[float(v) for v in _m],
        overall_ratio=float(_m[-1] / _m[0]),
        fitted_exponent=float(np.polyfit(np.log(_z), np.log(_m), 1)[0]),
        local_exponents=[float(v) for v in
                         np.diff(np.log(_m)) / np.diff(np.log(_z))])
    res["threshold_vs_zeta_probed"] = threshold_vs_zeta()
    res["orbit_accuracy"] = orbit_accuracy()
    out = os.path.join(HERE, "..", "..", "checkpoints", "onset_mechanism_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
