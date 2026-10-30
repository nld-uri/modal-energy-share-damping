"""M5: share threshold against damping error on the port's own 35-cell map (stored results only).
Writes checkpoints/m_track/m5_share_vs_error.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.abspath(os.path.join(HERE, '..'))  # repository root
d = json.load(open(os.path.join(P3, 'checkpoints', 'bias_map_results.json')))
cells = d['grid'] + d['threshold_sweep']
TAU, S0 = 0.005, 0.99
out = dict(n_cells=len(cells), tau=TAU, share_threshold=S0, estimators={})
for est in ('r_env', 'r_gcvd'):
    hi_bad = [c for c in cells if c['e_frac'] >= S0 and abs(c[est] - 1) > TAU]
    lo_ok = [c for c in cells if c['e_frac'] < S0 and abs(c[est] - 1) <= TAU]
    out['estimators'][est] = dict(
        n_share_hi=sum(c['e_frac'] >= S0 for c in cells), n_share_lo=sum(c['e_frac'] < S0 for c in cells),
        N1_hi_share_but_error=len(hi_bad), N2_lo_share_but_accurate=len(lo_ok),
        N1_cells=[dict(k3=c['k3'], A=c['A'], mu=c['k3'] * c['A'] ** 2, share=c['e_frac'], err=c[est] - 1) for c in hi_bad],
        N2_cells=[dict(k3=c['k3'], A=c['A'], mu=c['k3'] * c['A'] ** 2, share=c['e_frac'], err=c[est] - 1) for c in lo_ok],
        max_err_hi_share=max(abs(c[est] - 1) for c in cells if c['e_frac'] >= S0))
json.dump(out, open(os.path.join(HERE, '..', 'checkpoints', 'm_track', 'm5_share_vs_error.json'), 'w'), indent=1)
print(json.dumps(out, indent=1)[:2500])
