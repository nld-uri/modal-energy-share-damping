"""Run the MATLAB back-translation against reference.mat, mirroring selftest.m.

    cd python/verification && python check_mirror.py
"""
import os
import sys

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlmirror as m  # noqa: E402

_REF = os.path.join(HERE, "..", "..", "matlab", "reference.mat")
R = {k: v for k, v in loadmat(_REF).items() if not k.startswith("__")}
S = lambda k: float(np.asarray(R[k]).ravel()[0])
V = lambda k: np.asarray(R[k], dtype=float).ravel()

npass = nfail = 0
fails = []


def chk(name, got, ref, tol, kind="rel"):
    global npass, nfail
    g = np.asarray(got, dtype=complex).ravel()
    r = np.asarray(ref, dtype=complex).ravel()
    if g.size != r.size:
        ad = rd = np.inf
    else:
        ad = float(np.max(np.abs(g - r)))
        den = float(np.max(np.abs(r))) or 1.0
        rd = ad / den
    val = rd if kind == "rel" else ad
    ok = np.isfinite(val) and val <= tol
    if ok:
        npass += 1
    else:
        nfail += 1
        fails.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} |d| {ad:9.2e}  rel {rd:9.2e}  "
          f"tol {tol:8.1e} ({kind})")


N, J0, MODE = 24, int(S("cfg_j0_matlab1based")), int(S("cfg_mode_matlab1based"))
DT, SPAN, RTOL = S("cfg_dt"), S("cfg_span"), S("cfg_rtol")
ZET, TSET, EDGE = S("cfg_zeta1"), S("cfg_t_settle"), int(S("cfg_edge"))
print(f"\nMATLAB back-translation vs reference.mat  (j0={J0}, mode={MODE}, 1-based)\n")

print("-- 1. chain structure")
L0 = m.lattice(N, J0, 0.0, ZET)
chk("chain_wn", L0["wn"], V("chain_wn"), 1e-12)
chk("chain_Phi", np.asarray(R["chain_Phi"]).ravel(), L0["Phi"].ravel(), 1e-12)
chk("chain_psi", L0["psi"], V("chain_psi"), 1e-12)
chk("chain_ci", L0["ci"], V("chain_ci"), 1e-12)
chk("psi == Phi(j0,:).'", L0["psi"], L0["Phi"][J0 - 1, :], 0.0, "abs")

print("-- 2. analytic signal")
ze, zo = m.hilbert_fft(V("hilb_even_x")), m.hilbert_fft(V("hilb_odd_x"))
chk("hilbert even real", ze.real, V("hilb_even_re"), 1e-12)
chk("hilbert even imag", ze.imag, V("hilb_even_im"), 1e-12)
chk("hilbert even env", np.abs(ze), V("hilb_even_env"), 1e-12)
chk("hilbert odd real", zo.real, V("hilb_odd_re"), 1e-12)
chk("hilbert odd imag", zo.imag, V("hilb_odd_im"), 1e-12)
chk("hilbert odd env", np.abs(zo), V("hilb_odd_env"), 1e-12)

print("-- 3. GCVD on the synthetic record")
D, Vv, Ac = R["gcvd_disp"], R["gcvd_vel"], R["gcvd_acc"]
lam, _, _ = m.gcvd(D, Vv, Ac, True)
chk("gcvd |lam| sorted", np.sort(np.abs(lam)), np.sort(V("gcvd_lam_abs")), 1e-10)
g1 = m.gcvd_windowed(D, Vv, Ac, 1, 6, True, np.eye(6))
chk("tracked wn", g1["wn"][0], V("gcvd_track_wn"), 1e-10)
chk("tracked zeta", g1["zeta"][0], V("gcvd_track_zeta"), 1e-10)
chk("tracked mpc", g1["mpc"][0], V("gcvd_track_mpc"), 1e-10)
chk("tracked energy", g1["energy"][0], V("gcvd_track_energy"), 1e-10)
chk("tracked amp", g1["amp"][0], V("gcvd_track_amp"), 1e-10)
chk("tracked shapes", g1["shape"][0].ravel(), np.asarray(R["gcvd_track_shape"]).ravel(), 1e-10)
chk("recovers true wn", g1["wn"][0], V("gcvd_true_wn"), 1e-8)
chk("recovers true zeta", g1["zeta"][0], V("gcvd_true_zeta"), 1e-8)
g3 = m.gcvd_windowed(D, Vv, Ac, 3, 3, True, np.eye(6))
chk("3-window wn", g3["wn"].ravel(), np.asarray(R["gcvd_w3_wn"]).ravel(), 1e-10)
chk("3-window zeta", g3["zeta"].ravel(), np.asarray(R["gcvd_w3_zeta"]).ravel(), 1e-10)
chk("window edges", g3["edges"], V("gcvd_w3_edges"), 0.0, "abs")
ur, mr = m.realify(V("realify_v_re") + 1j * V("realify_v_im"))
chk("realify shape", ur, V("realify_u"), 1e-12)
chk("realify mpc", mr, S("realify_mpc"), 1e-12)

print("-- 4./5. RHS evaluations")
Lr = m.lattice(N, J0, S("cfg_k3_ref"), ZET)
chk("rhs_free", m.rhs_free(Lr)(0.0, V("rhs_state")), V("rhs_free_out"), 1e-12)
chk("rhs_forced", m.rhs_forced(Lr, S("rhs_forced_A"), S("rhs_forced_Om"))(
    S("rhs_forced_t"), V("rhs_state")), V("rhs_forced_out"), 1e-12)
Sb = m.boucwen()
chk("bw wn", Sb["wn"], V("bw_wn"), 1e-12)
chk("bw psit", Sb["psit"], V("bw_psit"), 1e-12)
chk("bw rhs free", Sb["rhs"](0, 0)(0.0, V("bw_rhs_state")), V("bw_rhs_free_out"), 1e-12)
chk("bw rhs forced", Sb["rhs"](2e-2, Sb["wn"][MODE - 1])(3.25, V("bw_rhs_state")),
    V("bw_rhs_forced_out"), 1e-12)

print("-- 6. estimators on the supplied sub-record")
es = m.estimates(Lr, R["sub_disp"], R["sub_vel"], R["sub_acc"], MODE, DT)
for k in ("r_gcvd", "r_env", "r_flux", "stiff_pct", "a_m", "w_rms", "e_frac", "cond"):
    chk(f"sub {k}", es[k], S(f"subest_{k}"), 1e-8)

print("-- 7. sensor index sets and conditioning")
sub5 = np.arange(MODE - 2, MODE + 3)
cnds, idx_ok = [], True
ref_idx = np.asarray(R["sensor_idx_matlab"], dtype=int)
for q, ms in enumerate(V("sensor_counts").astype(int)):
    # MATLAB round() is half-AWAY-FROM-ZERO; numpy's is half-to-even.  Emulate
    # MATLAB's rule, otherwise this check silently tests the wrong thing.
    _x = np.linspace(1, N, ms)
    idx = np.unique((np.sign(_x) * np.floor(np.abs(_x) + 0.5)).astype(int))
    ri = ref_idx[q][ref_idx[q] > 0]
    if not np.array_equal(idx, ri):
        idx_ok = False
        print(f"      sensor-index MISMATCH at m={ms}: MATLAB {idx} vs Python {ri}")
    cnds.append(np.linalg.cond(L0["Phi"][idx - 1][:, sub5 - 1]))
chk("sensor index sets identical", float(idx_ok), 1.0, 0.0, "abs")
chk("cond(Phi_sensed)", cnds, V("sensor_cond"), 1e-10)

print("-- 8./9./10. integrated: settle, release, share law, subsets")
tt, aa, vv, ac, y0 = m.settle_release(Lr, S("cfg_A_ref"), MODE, SPAN, DT, RTOL, TSET)
chk("settled state y0", y0, V("settle_y0"), 1e-6)
ef = m.estimates(Lr, aa, vv, ac, MODE, DT)
for k in ("r_gcvd", "r_env", "r_flux", "stiff_pct", "e_frac", "w_rms"):
    chk(k, ef[k], S(f"est_{k}"), 1e-6, "abs")
er = m.energy_rates(Lr, tt, aa, vv, MODE, None, EDGE)
chk("share mean", er["share_mean"], S("share_mean"), 1e-6, "abs")
chk("share start", er["share_start"], S("share_start"), 1e-6, "abs")
chk("share end", er["share_end"], S("share_end"), 1e-6, "abs")
chk("d(ln f)/dt", er["dlnf_tot"], S("share_dlnf_dt"), 1e-9, "abs")
chk("cbar/c", er["cbar_mean_over_c"], S("share_cbar_over_ctrue"), 1e-6, "abs")
chk("r_derived", er["r_derived"], S("share_r_derived"), 1e-6, "abs")
lo, hi = V("subset_lo_matlab").astype(int), V("subset_hi_matlab").astype(int)
gs = [m.energy_rates(Lr, tt, aa, vv, MODE, np.arange(lo[q], hi[q] + 1), EDGE)
      for q in range(5)]
chk("subset gamma/c", [g["gamma_over_c"] for g in gs], V("subset_gamma_over_c"), 1e-6, "abs")
chk("subset vacuous", [g["vacuous_over_c"] for g in gs], V("subset_vacuous"), 1e-6, "abs")

print("-- 11. Floquet")
for tag in ("stable", "unstable"):
    fl = m.floquet(S(f"floq_{tag}_mu"), S("floq_A"), MODE, N, J0, ZET,
                   S("floq_t_probe"), int(S("floq_npts")))
    chk(f"floquet {tag} w_peak", fl["w_peak"], S(f"floq_{tag}_wpeak"), 1e-6, "abs")
    chk(f"floquet {tag} max|mult|", fl["max_multiplier"],
        S(f"floq_{tag}_max_mult"), 1e-6, "abs")

print("-- 12. Bouc-Wen record")
bw = m.boucwen_record(S("bw_A"), MODE, S("bw_t_settle"), S("bw_span"),
                      S("bw_dt"), S("bw_rtol"), EDGE)
for k in ("slip_ratio", "w_rms", "e_frac", "gamma_tot_over_c",
          "gamma_sub_over_c", "r_env_over_c"):
    chk(f"bw {k}", bw[k], S(f"bw_{k}"), 1e-3)
chk("bw settled state y0", bw["y0"], V("bw_y0"), 1e-3)

print("-- 13. band-forced cell")
from scipy.integrate import solve_ivp
Lb = m.lattice(N, J0, S("band_k3"), ZET)
tev = S("band_t_settle") + V("band_releases")
sb = solve_ivp(m.rhs_forced(Lb, S("band_a"), V("band_wk"), V("band_ph")),
               (0.0, tev[-1]), np.zeros(2 * N), method="DOP853", rtol=1e-9,
               atol=1e-11, t_eval=np.concatenate([[0.0], tev]))
y0b = sb.y[:, 1]
chk("band release state", y0b, V("band_y0_release0"), 1e-6)
ttb = np.arange(round(SPAN / DT)) * DT
db = solve_ivp(m.rhs_free(Lb), (ttb[0], ttb[-1]), y0b, method="DOP853",
               rtol=RTOL, atol=RTOL * 1e-2, t_eval=ttb)
ab, vb = db.y[:N].T, db.y[N:].T
cb = (-(Lb["ci"][None, :]) * vb - ((Lb["wn"] ** 2)[None, :]) * ab
      - (Lb["k3"] * (ab @ Lb["psi"]) ** 3)[:, None] * Lb["psi"][None, :])
eb = m.estimates(Lb, ab, vb, cb, MODE, DT)
rb = m.energy_rates(Lb, ttb, ab, vb, MODE, None, EDGE)
chk("band r_env", eb["r_env"], S("band_r_env"), 1e-6, "abs")
chk("band r_gcvd", eb["r_gcvd"], S("band_r_gcvd"), 1e-6, "abs")
chk("band share mean", rb["share_mean"], S("band_share_mean"), 1e-6, "abs")
chk("band d(ln f)/dt", rb["dlnf_tot"], S("band_dlnf_dt"), 1e-9, "abs")

print(f"\n==== mirror: {npass + nfail} checks, {npass} PASS, {nfail} FAIL ====")
if fails:
    print("FAILED:", ", ".join(fails))
