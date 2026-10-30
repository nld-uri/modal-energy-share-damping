"""Linear growth budget for the hold preparations (step 2): does seed*|lam|^(2N) predict the
sideband share after the 6000-unit hold, and where does it reach 0.1?  Writes
checkpoints/protocols_growth_budget.json.  (Same code that was run interactively.)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import protocols_release_records_v2 as v2, lattice_nl as ln, bias_map as bm
R = os.path.join(HERE, '..', '..', 'checkpoints')
rr = json.load(open(os.path.join(R, 'protocols_release_records.json')))
PRJ = json.load(open(os.path.join(R, 'protocols_phase_resonance.json')))
omr = json.load(open(os.path.join(HERE, '..', '..', 'checkpoints', 'onset_mechanism_results.json')))
out = []
for r in rr['records']:
    if r['preparation'] == 'PR-ramp':
        continue
    if r['preparation'] == 'PR-hold':
        L, Om, orb, k, mx = v2.r1.pr_orbit(r['kappa_target'], PRJ); pair = (5, 7)
    else:
        st = [q for c in omr['converged'] if c['zeta1'] == 0.01 for q in c['records']]
        L, Om, orb, k, mx, mu = v2.r1.lin_orbit(r['kappa_target'], st); pair = (4, 8)
    y = orb['y0']; d = v2.perturbation(L, y)
    E = lambda z: 0.5 * (z[L.n:] ** 2 + L.wn ** 2 * z[:L.n] ** 2)
    seed = float((E(d)[pair[0]] + E(d)[pair[1]]) / E(y).sum())
    N = bm.T_SETTLE / (2 * np.pi / Om)
    s_obs = r['settled']['share_6_8'] if pair == (5, 7) else r['settled']['share_5_9']
    out.append(dict(prep=r['preparation'], kappa=r['kappa_target'], lam=mx, seed_pair_share=seed,
                    growth_factor_energy=float(mx ** (2 * N)), pred_pair_share_linear=float(seed * mx ** (2 * N)),
                    obs_pair_share=s_obs))
seed = out[0]['seed_pair_share']
def kfail(ks, lams, T):
    g = np.log(np.maximum(seed * np.array(lams) ** (2 * 6000 / T), 1e-300)) - np.log(0.1)
    ks = np.array(ks); i = np.where((g[:-1] < 0) & (g[1:] >= 0))[0][0]
    return float(ks[i] - g[i] * (ks[i + 1] - ks[i]) / (g[i + 1] - g[i]))
sw = [s for s in PRJ['sweeps'] if s['zeta1'] == 0.01][0]; pr = sorted(sw['records'], key=lambda r: r['kappa'])
lin = sorted([q for c in omr['converged'] if c['zeta1'] == 0.01 for q in c['records']], key=lambda r: r['k3_wpeak2'])
res = dict(method="pair share after hold = seed share * max|lam|^(2*T_hold/T); failure where it reaches 0.1",
           seed_pair_energy_share=seed, threshold_pair_share=0.1, T_hold=6000.0,
           kappa_fail_pred_PR=kfail([r['kappa'] for r in pr], [r['max_multiplier'] for r in pr],
                                    2 * np.pi / np.mean([r['Omega'] for r in pr])),
           kappa_fail_pred_LIN=kfail([r['k3_wpeak2'] for r in lin], [r['max_multiplier'] for r in lin], lin[0]['period']),
           records=out)
json.dump(res, open(os.path.join(R, 'protocols_growth_budget.json'), 'w'), indent=1)
print(res['kappa_fail_pred_PR'], res['kappa_fail_pred_LIN'])
