"""Export reference vectors for the MATLAB port of the Paper-3 damping study.

Runs the VERBATIM Python sources (pysrc/) and writes matlab/reference.mat
(MATLAB v5 + zlib == what MATLAB calls a v7 file, loadable by any release)
plus matlab/reference_scalars.csv for human inspection.

Everything written here is produced by the same code paths the paper used, so
a MATLAB/Python disagreement is a port bug (or a solver-tolerance difference),
never a difference in what was asked.
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
# The distributed bundle keeps the verbatim sources under pysrc/<area>; in the
# working tree the same directories sit directly under matlab_support/<area>.
# Prefer pysrc/ when it is present so the bundle still exports from the frozen
# snapshot, and fall back to the tree layout otherwise -- without the fallback
# this script cannot run from the repository at all (ModuleNotFoundError).
for _s in ("models", "identification", "experiments", "analysis"):
    _snap = os.path.join(HERE, "pysrc", _s)
    sys.path.insert(0, _snap if os.path.isdir(_snap) else os.path.join(HERE, _s))

import lattice_nl as ln          # noqa: E402
import bias_map as bm            # noqa: E402
import onset_mechanism as om     # noqa: E402
import friction_boundary as fb   # noqa: E402
import bandwidth_axis as ba      # noqa: E402
from gcvd import gcvd, windowed_gcvd, realify   # noqa: E402

N, J0, MODE = bm.N, bm.J0, bm.MODE      # 24, 1 (0-based), 6 (0-based)

# ---------------------------------------------------------------------------
# Canonicalise the chain eigenvectors to FIRST ROW POSITIVE, globally, before
# anything is computed.  numpy eigh and MATLAB eig each pick an arbitrary sign
# per eigenvector and disagree on 12 of the 24 chain modes, so every VECTOR-
# valued reference (psi, state vectors, the modal sub-record) would otherwise
# be exported in a basis the MATLAB port cannot reproduce -- while every
# scalar still matched, which is the worst kind of mismatch to debug.
# The change is bit-exact neutral for every scalar result (verified: 0.0
# difference with 12 signs flipped), and reproduces the analytic chain modes
# sqrt(2/(n+1)) sin(j i pi/(n+1)) to 5e-15.
_orig_chain_modes = ln.chain_modes


def _chain_canon(n=N):
    w, P = _orig_chain_modes(n)
    sg = np.sign(P[0])
    sg[sg == 0] = 1.0
    return w, P * sg


ln.chain_modes = _chain_canon
# ---------------------------------------------------------------------------
ZETA1, SPAN, DT, RTOL = bm.ZETA1, bm.SPAN, bm.DT, bm.RTOL
MU_REF, A_REF = 1.0e-3, 2.0e-2
K3_REF = MU_REF / A_REF ** 2            # 2.5
EDGE = 20
R = {}


def put(name, value):
    if isinstance(value, str) or (isinstance(value, np.ndarray) and value.dtype == object):
        R[name] = value
    else:
        R[name] = np.asarray(value, dtype=float)


# ---------------------------------------------------------------- 1. chain ---
wn, Phi_c = ln.chain_modes(N)           # already canonical (patched above)
_, Phi_raw = _orig_chain_modes(N)
e = np.zeros(N); e[J0] = 1.0
put("chain_wn", wn)
put("chain_Phi", Phi_c)
put("chain_psi", Phi_c.T @ e)           # canonical-sign psi, j0 = 1 (0-based)
put("chain_psi_raw", Phi_raw.T @ e)     # numpy eigh signs, for reference only
put("chain_ci", np.full(N, 2 * ZETA1 * wn[0]))
put("cfg_N", N); put("cfg_j0_python0based", J0); put("cfg_j0_matlab1based", J0 + 1)
put("cfg_mode_python0based", MODE); put("cfg_mode_matlab1based", MODE + 1)
put("cfg_zeta1", ZETA1); put("cfg_span", SPAN); put("cfg_dt", DT)
put("cfg_rtol", RTOL); put("cfg_t_settle", bm.T_SETTLE)
put("cfg_mu_ref", MU_REF); put("cfg_A_ref", A_REF); put("cfg_k3_ref", K3_REF)
put("cfg_edge", EDGE)

# ------------------------------------------------- 2. hilbert, even and odd ---
# Decaying chirp -- broadband, so a mishandled Nyquist bin or a wrong even/odd
# branch shows up immediately. Both parities are exported: the even case is the
# one with the extra h[N/2] = 1 rule.
for tag, m in (("even", 512), ("odd", 511)):
    t = np.arange(m) * 0.01
    x = np.exp(-0.7 * t) * np.cos(2 * np.pi * (1.3 * t + 0.45 * t ** 2)) + 0.15 * np.cos(2 * np.pi * 9.0 * t)
    h = hilbert(x)
    put(f"hilb_{tag}_x", x)
    put(f"hilb_{tag}_re", h.real)
    put(f"hilb_{tag}_im", h.imag)
    put(f"hilb_{tag}_env", np.abs(h))

# ------------------------------------- 3. gcvd on a synthetic exact response ---
# Six well-separated modes with known poles and orthogonal shapes: gcvd must
# return those poles. Well conditioned by construction, so this isolates the
# algorithm from the ill-conditioning of a real decay window.
ns = 6
wns, Phis = ln.chain_modes(ns)          # canonical
wns = np.array([0.40, 0.95, 1.55, 2.10, 2.70, 3.30])
zes = np.array([0.010, 0.015, 0.008, 0.020, 0.012, 0.025])
amps = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.18])
phs = np.array([0.0, 0.6, 1.2, 1.9, 2.5, 3.1])
ts = np.arange(1024) * 0.05
wd = wns * np.sqrt(1 - zes ** 2)
sig = -zes * wns
E = np.exp(sig[None, :] * ts[:, None])
C, S = np.cos(wd[None, :] * ts[:, None] + phs[None, :]), np.sin(wd[None, :] * ts[:, None] + phs[None, :])
q = amps[None, :] * E * C
qd = amps[None, :] * E * (sig[None, :] * C - wd[None, :] * S)
qdd = amps[None, :] * E * ((sig ** 2 - wd ** 2)[None, :] * C - (2 * sig * wd)[None, :] * S)
Ds, Vs, As_ = q @ Phis.T, qd @ Phis.T, qdd @ Phis.T
lam_s, V_s, Xi_s = gcvd(Ds, Vs, As_, center=True)
ordr = np.argsort(np.abs(lam_s))
put("gcvd_disp", Ds); put("gcvd_vel", Vs); put("gcvd_acc", As_)
put("gcvd_lam_re", lam_s[ordr].real); put("gcvd_lam_im", lam_s[ordr].imag)
put("gcvd_lam_abs", np.abs(lam_s[ordr]))
put("gcvd_true_wn", wns); put("gcvd_true_zeta", zes)
gw = windowed_gcvd(Ds, Vs, As_, n_windows=1, n_track=ns, M=np.eye(ns))
put("gcvd_track_wn", gw["wn"][0]); put("gcvd_track_zeta", gw["zeta"][0])
put("gcvd_track_mpc", gw["mpc"][0]); put("gcvd_track_shape", gw["shape"][0])
put("gcvd_track_amp", gw["amp"][0]); put("gcvd_track_energy", gw["energy"][0])
gw3 = windowed_gcvd(Ds, Vs, As_, n_windows=3, n_track=3, M=np.eye(ns))
put("gcvd_w3_wn", gw3["wn"]); put("gcvd_w3_zeta", gw3["zeta"])
put("gcvd_w3_edges", gw3["edges"])

# realify on a known complex vector (phase rotation + MPC)
vtest = np.array([0.3 + 0.05j, -0.8 - 0.10j, 1.0 + 0.0j, 0.45 - 0.07j, -0.2 + 0.03j, 0.11 + 0.01j])
u_t, mpc_t = realify(vtest)
put("realify_v_re", vtest.real); put("realify_v_im", vtest.imag)
put("realify_u", u_t); put("realify_mpc", mpc_t)

# --------------------------------------------------- 4. rhs evaluations -------
Lr = ln.Lattice(n=N, j0=J0, k3=K3_REF, zeta1=ZETA1)
zt = np.concatenate([0.01 * np.cos(np.arange(N) * 0.7), 0.003 * np.sin(np.arange(N) * 1.1)])
put("rhs_state", zt)
put("rhs_free_out", Lr.rhs_free()(0.0, zt))
put("rhs_forced_t", 3.25); put("rhs_forced_A", A_REF)
put("rhs_forced_Om", Lr.wn[MODE])
put("rhs_forced_out", Lr.rhs_forced(A_REF, Lr.wn[MODE])(3.25, zt))

# ------------------------------------------- 5. settle + release at mu = 1e-3 --
y0 = bm._settle(Lr, A_REF, MODE)
tt = np.arange(0.0, SPAN, DT)
aa, vv, ac = Lr.decay(y0, tt, rtol=RTOL)
put("settle_y0", y0)
est = bm._three_estimates(Lr, aa, vv, ac, MODE)
for k, v in est.items():
    put(f"est_{k}", v)

# a compact real sub-record so estimates.m can be tested WITHOUT integrating
NSUB = 2000
put("sub_disp", aa[:NSUB]); put("sub_vel", vv[:NSUB]); put("sub_acc", ac[:NSUB])
est_sub = bm._three_estimates(Lr, aa[:NSUB], vv[:NSUB], ac[:NSUB], MODE)
for k, v in est_sub.items():
    put(f"subest_{k}", v)

# --------------------------------------------------- 6. share law -------------
Ei = 0.5 * (vv ** 2 + (Lr.wn ** 2)[None, :] * aa ** 2)
Etot = Ei.sum(axis=1)
fsh = Ei[:, MODE] / Etot
cbar = (Ei * Lr.ci[None, :]).sum(axis=1) / Etot
kk = len(tt) // EDGE
sl = slice(kk, len(tt) - kk)
dlnf = float(np.polyfit(tt[sl], np.log(fsh[sl]), 1)[0])
env = np.abs(hilbert(aa[:, MODE]))
put("share_mean", fsh.mean()); put("share_start", fsh[0]); put("share_end", fsh[-1])
put("share_dlnf_dt", dlnf)
put("share_cbar_over_ctrue", cbar[sl].mean() / Lr.ci[MODE])
put("share_r_env", -2 * np.polyfit(tt[sl], np.log(env[sl]), 1)[0] / Lr.ci[MODE])
put("share_r_derived", (cbar[sl].mean() - dlnf) / Lr.ci[MODE])
put("share_r_proxy", fsh.mean())
put("share_Ei_first", Ei[0]); put("share_Ei_last", Ei[-1])

# ------------------------------------- 7. recovery-recipe subset table --------
subs = [("carrier only", [MODE]),
        ("carrier +/- 1", list(range(MODE - 1, MODE + 2))),
        ("carrier +/- 2", list(range(MODE - 2, MODE + 3))),
        ("carrier +/- 4", list(range(MODE - 4, MODE + 5))),
        ("all modes", list(range(N)))]
gam, vac = [], []
for nm, ss in subs:
    Es = Ei[:, ss].sum(axis=1)
    g = float(-np.polyfit(tt[sl], np.log(Es[sl]), 1)[0])
    f2 = Ei[:, MODE] / Es
    dl = float(np.polyfit(tt[sl], np.log(f2[sl]), 1)[0])
    gam.append(g / Lr.ci[MODE]); vac.append((g - dl) / Lr.ci[MODE])
put("subset_gamma_over_c", gam)
put("subset_vacuous", vac)
put("subset_names", np.array([s[0] for s in subs], dtype=object))
put("subset_lo_matlab", [min(s[1]) + 1 for s in subs])
put("subset_hi_matlab", [max(s[1]) + 1 for s in subs])

# --------------------------------------------------- 8. Floquet ---------------
for tag, mu in (("stable", 3.0e-4), ("unstable", 5.0e-4)):
    d = om.floquet_true_orbit(mu)
    put(f"floq_{tag}_mu", d["mu"]); put(f"floq_{tag}_k3", d["k3"])
    put(f"floq_{tag}_wpeak", d["w_peak"])
    put(f"floq_{tag}_max_mult", d["max_multiplier"])
    put(f"floq_{tag}_crest", d["crest_factor"])
put("floq_t_probe", 2000.0); put("floq_npts", 801)
put("floq_A", om.A_FIX)

# --------------------------------------------------- 9. Bouc-Wen -------------
S = fb.system()
# Canonical first-row-positive basis for the stuck-element reference structure.
# Verified bit-exact neutral: 11 of 24 sign flips change no exported result.
_L0 = ln.Lattice(n=N, j0=J0, k3=0.0, zeta1=ZETA1)
_Km = np.diag(_L0.wn ** 2) + fb.KH * np.outer(_L0.psi, _L0.psi)
_lm, _V = np.linalg.eigh(_Km)
_V = _V * np.sign(_V[0])
put("bw_wn", np.sqrt(_lm)); put("bw_psit", _V.T @ _L0.psi); put("bw_c", S["c"])
put("bw_psit_raw", S["psit"])
put("bw_kh", S["kh"]); put("bw_alpha", S["alpha"]); put("bw_uy", S["uy"])
put("bw_beta", S["beta"]); put("bw_gamma", S["gamma"])
put("bw_t_settle", fb.T_SETTLE); put("bw_span", fb.SPAN)
put("bw_dt", fb.DT); put("bw_rtol", fb.RTOL)
Sc = dict(S); Sc["wn"] = np.sqrt(_lm); Sc["psit"] = _V.T @ _L0.psi
ybw = np.concatenate([zt, [0.4 * S["uy"]]])
put("bw_rhs_state", ybw)
put("bw_rhs_free_out", fb.rhs(Sc)(0.0, ybw))
put("bw_rhs_forced_out", fb.rhs(Sc, 2e-2, Sc["wn"][MODE])(3.25, ybw))

_sbw = solve_ivp(fb.rhs(Sc, 2e-2, Sc["wn"][MODE]), (0.0, fb.T_SETTLE),
                 np.zeros(2 * N + 1), method="DOP853", rtol=fb.RTOL, atol=fb.RTOL * 1e-2)
put("bw_y0", _sbw.y[:, -1])
bwrec = fb.one_amplitude(2e-2)
for k, v in bwrec.items():
    put(f"bw_{k}", v)

# ------------------------------- 10. bandwidth cell (phases exported) ---------
# The band forcing uses numpy PCG64 phases, which base MATLAB cannot reproduce;
# exporting the 25 phases makes the cell exactly reproducible anyway.
BETA_REF, TGT_REF, SEED_REF = 0.25, 0.45, 1
Lb = ln.Lattice(n=N, j0=J0, k3=ba.K3, zeta1=ZETA1)
wk, ph, aamp = ba.band_force(Lb, BETA_REF, MODE, TGT_REF, seed=SEED_REF)
put("band_beta", BETA_REF); put("band_target", TGT_REF); put("band_k3", ba.K3)
put("band_wk", wk); put("band_ph", ph); put("band_a", aamp)
put("band_t_settle", ba.T_SETTLE); put("band_releases", np.array(ba.RELEASES))
put("band_n_neighbours", ba.neighbours_in_band(Lb, BETA_REF, MODE))
n_ = Lb.n
W2b, Cb, PSIb = Lb.wn ** 2, Lb.ci, Lb.psi


def f_band(t, z):
    u = aamp * np.cos(wk * t + ph).sum()
    return np.concatenate([z[n_:], -Cb * z[n_:] - W2b * z[:n_]
                           - PSIb * (ba.K3 * (PSIb @ z[:n_]) ** 3) + PSIb * u])


sb = solve_ivp(f_band, (0.0, ba.T_SETTLE + max(ba.RELEASES)), np.zeros(2 * n_),
               method="DOP853", rtol=1e-9, atol=1e-11,
               t_eval=ba.T_SETTLE + np.array(ba.RELEASES))
put("band_y0_release0", sb.y[:, 0])
aab, vvb, acb = Lb.decay(sb.y[:, 0], tt, rtol=RTOL)
eb = bm._three_estimates(Lb, aab, vvb, acb, MODE)
for k, v in eb.items():
    put(f"band_{k}", v)
Eib = 0.5 * (vvb ** 2 + W2b[None, :] * aab ** 2)
fb_ = Eib[:, MODE] / Eib.sum(axis=1)
put("band_share_mean", fb_.mean())
put("band_dlnf_dt", float(np.polyfit(tt[sl], np.log(fb_[sl]), 1)[0]))

# ------------------------- 11. sensor-projection conditioning (deterministic) --
sub5 = list(range(MODE - 2, MODE + 3))
conds, idxs = [], []
for m in (24, 16, 12, 8, 6, 5):
    idx = np.unique(np.linspace(0, N - 1, m).round().astype(int))
    conds.append(float(np.linalg.cond(Lr.Phi[idx][:, sub5])))
    idxs.append((idx + 1).tolist() + [0] * (24 - len(idx)))
put("sensor_counts", [24, 16, 12, 8, 6, 5])
put("sensor_cond", conds)
put("sensor_idx_matlab", np.array(idxs, dtype=float))

# ------------------------------------------------------------- write ----------
# Same bundle-versus-tree split as the source path above: the bundle keeps
# matlab/ beside this script, the working tree keeps it at the paper root.
_MDIR = os.path.join(HERE, "matlab")
if not os.path.isdir(_MDIR):
    _MDIR = os.path.abspath(os.path.join(HERE, "..", "..", "matlab"))
out_mat = os.path.join(_MDIR, "reference.mat")
savemat(out_mat, R, format="5", do_compression=True, oned_as="column")
scal = {k: float(v) for k, v in R.items()
        if isinstance(v, np.ndarray) and v.dtype != object and v.size == 1}
with open(os.path.join(_MDIR, "reference_scalars.csv"), "w") as fh:
    fh.write("name,value\n")
    for k in sorted(scal):
        fh.write(f"{k},{scal[k]!r}\n")
print(f"wrote {out_mat}  ({os.path.getsize(out_mat)/1e6:.2f} MB, {len(R)} variables)")
for k in ("est_r_gcvd", "est_r_env", "est_r_flux", "est_e_frac", "share_dlnf_dt",
          "share_r_derived", "floq_stable_max_mult", "floq_unstable_max_mult",
          "bw_gamma_tot_over_c", "band_r_env", "band_share_mean"):
    print(f"  {k:28s} {float(R[k]):.10f}")
print("  subset gamma/c:", np.array2string(np.asarray(gam), precision=6))
