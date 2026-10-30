"""M1: averaged combination-resonance condition on the port's omega_lin protocol.

Evaluates lib/averaged.py (copy of Alt B's theory module) on the port's Lattice,
maps kappa thresholds to the port's mu via the port's own converged Newton-shot
orbits (checkpoints/onset_mechanism_results.json), and compares with the port's
Floquet boundary and multipliers. Writes checkpoints/m_track/m1_averaged.json.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.abspath(os.path.join(HERE, '..'))  # repository root
sys.path[:0] = [os.path.join(HERE, 'lib'), os.path.join(P3, 'code', 'models'), os.path.join(P3, 'code', 'experiments')]
import lattice_nl as ln
import bias_map as bm
import averaged as av

ON = json.load(open(os.path.join(P3, 'checkpoints', 'onset_mechanism_results.json')))
MODE = bm.MODE
out = dict(source='port checkpoints/onset_mechanism_results.json (converged) + lib/averaged.py', zetas=[])
for Z in ON['converged']:
    z = Z['zeta1']
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=1.0, zeta1=z)
    Om = float(L.wn[MODE])
    law = lambda k: Om
    W = av.all_windows(L, MODE, law, kmax=0.3, nscan=600)
    wins = {f'{a}-{b}': v for (a, b), v in W.items()}
    recs = [r for r in Z['records'] if r['return_residual'] < 1e-8]   # excludes the spurious orbit
    mu = np.array([r['mu'] for r in recs]); ka = np.array([r['k3_wpeak2'] for r in recs])
    mx = np.array([r['max_multiplier'] for r in recs])
    o = np.argsort(mu); mu, ka, mx = mu[o], ka[o], mx[o]
    assert np.all(np.diff(ka) > 0)
    k2mu = lambda k: float(np.exp(np.interp(np.log(k), np.log(ka), np.log(mu))))
    # Floquet bracket of the (5,9) boundary the port reports (first stable->unstable crossing at mu >= 1e-4)
    i = next(j for j in range(len(mx) - 1) if mx[j] <= 1 < mx[j + 1])
    br_k, br_mu = (float(ka[i]), float(ka[i + 1])), (float(mu[i]), float(mu[i + 1]))
    w59 = W.get((5, 9), [])
    kc59 = next(lo for lo, hi in w59 if lo <= br_k[1] and hi >= br_k[0]) if w59 else None
    mu_pred = k2mu(kc59) if kc59 is not None and ka[0] <= kc59 <= ka[-1] else None
    # pair that the theory says goes unstable first at kappa >= ka[0] (the sampled range)
    firsts = sorted((lo, p) for p, ws in W.items() for lo, hi in ws)
    # multipliers at every valid record
    mrows = []
    for r in recs:
        Lk = ln.Lattice(n=bm.N, j0=bm.J0, k3=r['k3'], zeta1=z)
        lam, best = av.predicted_max_multiplier(Lk, r['k3_wpeak2'], Om, MODE)
        flo_pair = sorted(c['mode'] for c in r['content']['content'][:2])
        mrows.append(dict(mu=r['mu'], kappa=r['k3_wpeak2'], floquet=r['max_multiplier'], predicted=lam,
                          diff=lam - r['max_multiplier'], pred_pair=list(best['pair']),
                          floquet_leading_two=flo_pair,
                          side_agree=bool((lam > 1) == (r['max_multiplier'] > 1))))
    out['zetas'].append(dict(zeta1=z, Omega=Om, windows=wins, first_edges=[[float(lo), f'{p[0]}-{p[1]}'] for lo, p in firsts[:6]],
                             floquet_bracket_kappa=br_k, floquet_bracket_mu=br_mu,
                             port_mu_c=Z['threshold_mu'], kappa_c_59=kc59, mu_c_pred=mu_pred,
                             mu_c_rel_err=(mu_pred / Z['threshold_mu'] - 1) if mu_pred else None,
                             kappa_c_59_in_bracket=bool(kc59 is not None and br_k[0] <= kc59 <= br_k[1]),
                             kappa_range_sampled=[float(ka[0]), float(ka[-1])],
                             multipliers=mrows,
                             max_abs_mult_diff=float(max(abs(m['diff']) for m in mrows)),
                             side_disagreements=[m['mu'] for m in mrows if not m['side_agree']]))
# P1.4 at the zeta=0.010 boundary: raw and shifted detunings, Gamma, required
Z = next(z for z in out['zetas'] if z['zeta1'] == 0.010)
L = ln.Lattice(n=bm.N, j0=bm.J0, k3=1.0, zeta1=0.010); Om = float(L.wn[MODE])
def row(tab, p): return next(r for r in tab if tuple(r['pair']) == p)
kc = Z['kappa_c_59']
sh = av.pair_table(L, kc, Om, MODE); raw = av.pair_table(L, kc, Om, MODE, raw=True)
p14 = {}
for p in ((5, 9), (6, 8)):
    s, r_ = row(sh, p), row(raw, p)
    req = np.sqrt(s['mu_j'] * s['mu_k'] * (1 + (s['sigma'] / (s['mu_j'] + s['mu_k'])) ** 2))
    p14[f'{p[0]}-{p[1]}'] = dict(sigma_raw=r_['sigma'], sigma_shifted=s['sigma'], Gamma=s['Gamma'], Gamma_required=float(req),
                                 margin=s['margin'], psi_product_raw=float(abs(L.psi[p[0]-1] * L.psi[p[1]-1])))
# where the mean shift carries each pair through exact tuning
from scipy.optimize import brentq
def tune(p):
    f = lambda k: (lambda ms: 2 * Om - ms['wt'][p[0]-1] - ms['wt'][p[1]-1])(av.mean_shift(L, k))
    return float(brentq(f, 1e-6, 1.0))
out['p14'] = dict(kappa=kc, pairs=p14, kappa_exact_tuning={'6-8': tune((6, 8)), '5-9': tune((5, 9))})
r_k = out['zetas'][2]['kappa_c_59'] / out['zetas'][0]['kappa_c_59']
r_mu = ON['threshold_scaling']['overall_ratio']
out['damping_dependence'] = dict(kappa_ratio_020_005=r_k, mu_ratio_020_005=r_mu,
                                 floquet_kappa_mid_ratio=float(np.mean(out['zetas'][2]['floquet_bracket_kappa']) / np.mean(out['zetas'][0]['floquet_bracket_kappa'])))
# lower intervals (below the sampled range) predicted by the theory
out['lower_intervals'] = {f"{z['zeta1']}": {p: [w for w in ws if w[0] < z['kappa_range_sampled'][0]] for p, ws in z['windows'].items()
                                            if any(w[0] < z['kappa_range_sampled'][0] for w in ws)} for z in out['zetas']}
os.makedirs(os.path.join(HERE, '..', 'checkpoints', 'm_track'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, '..', 'checkpoints', 'm_track', 'm1_averaged.json'), 'w'), indent=1)
for z in out['zetas']:
    print(f"zeta {z['zeta1']}: kc59 {z['kappa_c_59']:.5f} bracket {z['floquet_bracket_kappa']} in={z['kappa_c_59_in_bracket']} "
          f"mu_pred {z['mu_c_pred']:.4e} port {z['port_mu_c']:.4e} err {100*z['mu_c_rel_err']:+.2f}% maxdiff {z['max_abs_mult_diff']:.2e} sidedis {z['side_disagreements']}")
    print('   first edges', z['first_edges'][:4])
print('P1.4', json.dumps(out['p14'], indent=0)[:900])
print('damping', out['damping_dependence']); print('lower', out['lower_intervals'])
