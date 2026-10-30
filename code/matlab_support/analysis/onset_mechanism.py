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
MU_BELOW, MU_ABOVE = 4.0e-4, 5.0e-4
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
                          psi_lo=float(L.psi[lo]), psi_hi=float(L.psi[hi])))
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
    hist = [dict(t=float(tq), E_carrier=float(E[int(tq / dt_out), mode]),
                 E_lower=float(E[int(tq / dt_out), lo]),
                 E_upper=float(E[int(tq / dt_out), hi]),
                 ratio_upper_carrier=float(E[int(tq / dt_out), hi]
                                           / E[int(tq / dt_out), mode]))
            for tq in (500, 1500, 2500, 3500, 4500, 5500)]
    return dict(mu=float(mu), sideband_modes=[lo + 1, hi + 1],
                log_slope_energy=rates, passive_rate=float(-L.ci[mode]),
                net_growth_upper=float(rates[f"mode_{hi + 1}"] + L.ci[mode]),
                history=hist)


def _monodromy(L, wq_of_tau, tau, T):
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
    ev = np.linalg.eigvals(s.y[:, -1].reshape(2 * n, 2 * n))
    return float(np.max(np.abs(ev)))


def floquet_true_orbit(mu, A=A_FIX, mode=bm.MODE, t_probe=2000.0, npts=801):
    """Stability of the actual settled orbit, probed before sidebands saturate."""
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
    Om = L.wn[mode]
    T = 2 * np.pi / Om
    s = solve_ivp(L.rhs_forced(A, Om), (0.0, t_probe + T), np.zeros(2 * L.n),
                  method="DOP853", rtol=1e-11, atol=1e-14,
                  t_eval=np.linspace(t_probe, t_probe + T, npts))
    wq = s.y[:L.n].T @ L.psi
    tau = np.linspace(0.0, T, npts)
    mx = _monodromy(L, wq, tau, T)
    w_pk = float(np.abs(wq).max())
    return dict(mu=float(mu), k3=float(L.k3), w_peak=w_pk,
                k3_wpeak2=float(L.k3 * w_pk ** 2), max_multiplier=mx,
                unstable=bool(mx > 1.0),
                growth_rate_energy=float(2 * np.log(mx) / T),
                crest_factor=float(w_pk / np.sqrt(np.mean(wq ** 2))))


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


def main():
    res = dict(config=dict(N=bm.N, j0=bm.J0, mode_1based=bm.MODE + 1,
                           zeta1=bm.ZETA1, A_fixed=A_FIX, t_settle=T_SETTLE),
               spectrum=spectrum_relations(),
               destinations=[energy_destination(m) for m in MUS_DEST],
               growth=[sideband_growth(m) for m in (MU_BELOW, MU_ABOVE)],
               floquet_true_orbit=[floquet_true_orbit(m) for m in
                                   (3.0e-4, 3.2e-4, 3.4e-4, 3.6e-4, 3.8e-4,
                                    4.0e-4, 5.0e-4, 6.0e-4, 8.0e-4)],
               floquet_single_harmonic=[floquet_single_harmonic(k) for k in
                                        (0.01, 0.02, 0.04, 0.06, 0.08, 0.12)])
    out = os.path.join(HERE, "..", "..", "checkpoints", "onset_mechanism_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
