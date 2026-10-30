"""M3: new Newton-shot Floquet orbits below the port's sampled range (mu < 1e-4), omega_lin protocol,
using the port's own converged_zeta_sweep (code/experiments/onset_mechanism.py). Writes checkpoints/m_track/m3_lowmu_floquet.json."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.abspath(os.path.join(HERE, '..'))  # repository root
sys.path[:0] = [os.path.join(P3, 'code', 'models'), os.path.join(P3, 'code', 'experiments')]
import onset_mechanism as om
MUS = (5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5)
t0 = time.time()
res = om.converged_zeta_sweep(zetas=(0.005, 0.010), mus=MUS)
for z in res:
    for r in z['records']:
        r.pop('multipliers_re', None); r.pop('multipliers_im', None)
json.dump(dict(mus=MUS, protocol='omega_lin, A=0.02, port onset_mechanism.converged_zeta_sweep', wall_s=time.time() - t0,
               sweep=res), open(os.path.join(HERE, '..', 'checkpoints', 'm_track', 'm3_lowmu_floquet.json'), 'w'), indent=1)
for z in res:
    for r in z['records']:
        print(z['zeta1'], f"{r['mu']:.1e} kappa {r['k3_wpeak2']:.5f} |lam| {r['max_multiplier']:.6f} res {r['return_residual']:.1e} lead",
              [(c['mode'], round(c['share'], 3)) for c in r['content']['content'][:2]], flush=True)
print('wall', time.time() - t0)
