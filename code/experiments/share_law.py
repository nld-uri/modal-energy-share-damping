"""The damping-bias law, derived rather than fitted.

Paper 3, Section 7.  `bias_map.py` found empirically that past the instability
onset the identified-to-true modal damping ratio is close to the tracked mode's
share of the record energy.  That is a proxy, not the law, and reporting it as a
fit invites an objection a referee will certainly raise: a single exponential
fitted to a multi-mode decay returns an energy-weighted mixture of the modal
damping values, so is the "law" just mixing?

TWO ANSWERS, both computed here.

(1) It cannot be mixing in this system.  The modal damping coefficients are
    UNIFORM, c_i = 2 zeta_1 omega_1 for every i, so any energy-weighted mixture
    of them is exactly c.  Mixing can produce no bias at all here.

(2) The actual law follows in two lines.  Write the tracked mode's energy as
    E_m(t) = f(t) E_tot(t) with f the energy share.  The total decays at the
    energy-weighted mean damping cbar(t) = sum_i c_i E_i / sum_i E_i.  An
    envelope fit to mode m reports whatever exponential rate E_m shows, so

        c_ident  =  cbar  -  d(ln f)/dt .

    With uniform damping cbar == c_true identically, and the ENTIRE bias is the
    logarithmic drift of the energy share.  The system therefore isolates the
    second term by construction, which is why the effect is visible cleanly.

The sign is informative.  During the analysed window the share RISES (energy
returns to the carrier, because the decaying amplitude drops the orbit back
below the instability threshold and the sidebands stop being pumped), so
d(ln f)/dt > 0 and the identified damping is too SMALL -- which is the sign
observed.

Verified against the Hilbert reference over eight records spanning the
transition band: mean absolute error 0.0057, worst 0.0092, correlation 1.0000,
against 0.0359 / 0.0604 / 0.9898 for the empirical share proxy.

Usage:  python share_law.py       (writes share_law_results.json)
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
MUS = (4.0e-4, 5.0e-4, 6.0e-4, 7.0e-4, 8.0e-4, 1.0e-3, 1.3e-3, 1.6e-3)
EDGE = 20  # trim 1/EDGE of the window at each end before fitting


def one_record(mu, A=A_FIX, mode=bm.MODE):
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A, mode)
    tt = np.arange(0.0, bm.SPAN, bm.DT)
    aa, vv, _ = L.decay(y0, tt, rtol=bm.RTOL)

    Ei = 0.5 * (vv ** 2 + (L.wn ** 2)[None, :] * aa ** 2)
    Etot = Ei.sum(axis=1)
    f = Ei[:, mode] / Etot
    cbar = (Ei * L.ci[None, :]).sum(axis=1) / Etot     # energy-weighted mean damping

    k = len(tt) // EDGE
    sl = slice(k, len(tt) - k)
    dlnf = float(np.polyfit(tt[sl], np.log(f[sl]), 1)[0])

    env = np.abs(hilbert(aa[:, mode]))
    r_env = float(-2 * np.polyfit(tt[sl], np.log(env[sl]), 1)[0] / L.ci[mode])

    r_derived = float((cbar[sl].mean() - dlnf) / L.ci[mode])
    return dict(mu=float(mu), k3=float(L.k3),
                share_mean=float(f.mean()), share_start=float(f[0]),
                share_end=float(f[-1]), dlnf_dt=dlnf,
                cbar_over_ctrue=float(cbar[sl].mean() / L.ci[mode]),
                r_env=r_env, r_derived=r_derived, r_share_proxy=float(f.mean()))


def main():
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=1.0, zeta1=bm.ZETA1)
    recs = [one_record(m) for m in MUS]
    e = np.array([r["r_env"] for r in recs])
    d = np.array([r["r_derived"] for r in recs])
    s = np.array([r["r_share_proxy"] for r in recs])
    res = dict(
        config=dict(N=bm.N, j0=bm.J0, mode_1based=bm.MODE + 1, zeta1=bm.ZETA1,
                    A_fixed=A_FIX, span=bm.SPAN, dt=bm.DT, edge_trim=EDGE),
        uniform_damping=dict(
            c_min=float(L.ci.min()), c_max=float(L.ci.max()),
            all_equal=bool(np.allclose(L.ci, L.ci[0])),
            note=("modal damping is uniform, so an energy-weighted mixture of "
                  "modal damping values equals the true value and mixing cannot "
                  "produce bias in this system")),
        records=recs,
        derived_law=dict(mean_abs_err=float(np.mean(np.abs(e - d))),
                         worst_abs_err=float(np.max(np.abs(e - d))),
                         correlation=float(np.corrcoef(e, d)[0, 1])),
        share_proxy=dict(mean_abs_err=float(np.mean(np.abs(e - s))),
                         worst_abs_err=float(np.max(np.abs(e - s))),
                         correlation=float(np.corrcoef(e, s)[0, 1])))
    out = os.path.join(HERE, "..", "..", "checkpoints", "share_law_results.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
