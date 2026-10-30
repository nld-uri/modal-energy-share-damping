"""M7 summary: evaluates addendum-3 rules P7.1-P7.3 on m7/checkpoints/theory_second_config.json.
Writes checkpoints/m_track/m7_second_config.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'm7', 'checkpoints', 'theory_second_config.json')))
out = {}
for t in ('primary', 'secondary'):
    v = d[t]; b = v['floquet_bracket']; th = v['theory']
    mub = b['mu_interp']
    rel = sorted(v['release'], key=lambda r: r['mu'])
    below = [r for r in rel if r['mu'] < mub]; dep = [r for r in rel if abs(r['r_env'] - 1) > 0.005]
    flo_pair = sorted(m for m, _ in b['unstable_content'][:2])
    out[t] = dict(station_1based=v['station_1based'], mode_1based=v['mode_1based'],
                  kappa_c_pred=th['kappa_c'], pair_pred=th['pair'], kappa_bracket=[b['kappa_stable'], b['kappa_unstable']],
                  mu_boundary=mub, floquet_pair=flo_pair, floquet_shares=[s for _, s in b['unstable_content'][:2]],
                  kappa_pred_rel_to_bracket_low=th['kappa_c'] / b['kappa_stable'] - 1,
                  P7_1=bool(b['kappa_stable'] <= th['kappa_c'] <= b['kappa_unstable']),
                  P7_2=flo_pair == sorted(th['pair']),
                  P7_3=all(r['mu'] >= mub for r in dep),
                  max_env_err_below=max(abs(r['r_env'] - 1) for r in below),
                  first_departure=dict(mu=dep[0]['mu'], env_err=abs(dep[0]['r_env'] - 1)) if dep else None)
json.dump(out, open(os.path.join(HERE, '..', 'checkpoints', 'm_track', 'm7_second_config.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
