"""M6: model-informed recovery of the common viscous coefficient c and k3 from the port's own records.

Port protocol (bias_map._settle then L.decay; A = 0.02, zeta1 = 0.010, 400-unit window, dt = 0.05).
Joint least squares  -qdd_i - w_i^2 q_i = c qd_i + k3 psi_i w^3  over all modes and samples.
(a) model acceleration (the record's own acc), (b) 4th-order centred difference of sampled velocity.
Also recomputes the port's three estimates on the same record. Writes checkpoints/m_track/m6_coefficients.json."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.abspath(os.path.join(HERE, '..'))  # repository root
sys.path[:0] = [os.path.join(P3, 'code', 'models'), os.path.join(P3, 'code', 'experiments'), os.path.join(P3, 'code', 'identification')]
import lattice_nl as ln
import bias_map as bm
A = 0.02
rows = []
t0 = time.time()
for mu in (0.0, 3e-4, 1.6e-3):
    k3 = mu / A ** 2
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=k3, zeta1=bm.ZETA1)
    y0 = bm._settle(L, A, bm.MODE)
    aa, vv, ac = L.decay(y0, np.arange(0.0, bm.SPAN, bm.DT), rtol=bm.RTOL)
    est = bm._three_estimates(L, aa, vv, ac, bm.MODE)
    c_true = float(L.ci[0])
    def fit(q, qd, qdd):
        w = q @ L.psi
        y = (-qdd - q * L.wn ** 2).ravel()
        X = np.column_stack([qd.ravel(), (np.outer(w ** 3, L.psi)).ravel()])
        if k3 == 0.0:
            th, *_ = np.linalg.lstsq(X, y, rcond=None)
        else:
            th, *_ = np.linalg.lstsq(X / np.array([1.0, 1.0]), y, rcond=None)
        return float(th[0]), float(th[1])
    ca, ka = fit(aa, vv, ac)
    h = bm.DT
    acc_fd = (-vv[4:] + 8 * vv[3:-1] - 8 * vv[1:-3] + vv[:-4]) / (12 * h)
    cb, kb = fit(aa[2:-2], vv[2:-2], acc_fd)
    rows.append(dict(mu=mu, k3=k3, c_true=c_true, r_env=est['r_env'], r_gcvd=est['r_gcvd'], share=est['e_frac'],
                     a_model=dict(c=ca, k3=ka, c_rel_err=ca / c_true - 1, k3_rel_err=(ka / k3 - 1) if k3 else None, k3_abs=ka),
                     b_fd4=dict(c=cb, k3=kb, c_rel_err=cb / c_true - 1, k3_rel_err=(kb / k3 - 1) if k3 else None, k3_abs=kb)))
    print(json.dumps(rows[-1]), flush=True)
json.dump(dict(protocol='port bias_map._settle + Lattice.decay, A=0.02, zeta1=0.010, span 400, dt 0.05',
               wall_s=time.time() - t0, records=rows), open(os.path.join(HERE, '..', 'checkpoints', 'm_track', 'm6_coefficients.json'), 'w'), indent=1)
