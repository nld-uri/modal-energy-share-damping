"""M3b (addendum 1): does the lower (6,8) instability window at zeta1 = 0.005 bias the port's estimates?
Writes checkpoints/m_track/m3b_lowwindow_identification.json."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.abspath(os.path.join(HERE, '..'))  # repository root
sys.path[:0] = [os.path.join(P3, 'code', 'models'), os.path.join(P3, 'code', 'experiments'), os.path.join(P3, 'code', 'identification')]
import lattice_nl as ln
import bias_map as bm
from multiprocessing import Pool
A, Z = 0.02, 0.005
def one(mu):
    L = ln.Lattice(n=bm.N, j0=bm.J0, k3=mu / A ** 2, zeta1=Z)
    y0 = bm._settle(L, A, bm.MODE)
    aa, vv, ac = L.decay(y0, np.arange(0.0, bm.SPAN, bm.DT), rtol=bm.RTOL)
    e = bm._three_estimates(L, aa, vv, ac, bm.MODE)
    return dict(mu=mu, r_env=float(e['r_env']), r_gcvd=float(e['r_gcvd']), share=float(e['e_frac']))
if __name__ == '__main__':
    t0 = time.time()
    with Pool(4) as p: rows = p.map(one, [1e-5, 2.5e-5, 4e-5, 6e-5, 1.5e-4])
    win = [r for r in rows if r['mu'] <= 6e-5]
    q1 = all(r['share'] >= 0.99 and abs(r['r_env'] - 1) <= 0.005 for r in win)
    out = dict(zeta1=Z, A=A, records=rows, Q1_identification_silent=q1, wall_s=time.time() - t0)
    json.dump(out, open(os.path.join(HERE, '..', 'checkpoints', 'm_track', 'm3b_lowwindow_identification.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
