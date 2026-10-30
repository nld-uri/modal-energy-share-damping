"""Mechanical back-translation of the MATLAB port, used to verify it.

MATLAB cannot be run here, so every .m file is transliterated LINE BY LINE
into Python with its ONE-BASED indexing kept explicit (`rng1(a, b)` is the
MATLAB range a:b), and the result is checked against the same reference.mat
the MATLAB self-test loads.

This cannot catch MATLAB syntax errors -- the user's first run does that --
but it does catch the failures that would otherwise be silent: an off-by-one
in a window slice, a transposed psi, a wrong even/odd branch in the analytic
signal, ddof on a standard deviation, a column-vs-row-major reshape.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.linalg import svdvals


def rng1(a, b):
    """MATLAB a:b as a 0-based index array (empty when b < a)."""
    return np.arange(a - 1, b, dtype=int)


# --------------------------------------------------------------- lattice.m --
def lattice(n=24, j0=2, k3=0.0, zeta1=0.010):
    K = (np.diag(2 * np.ones(n)) - np.diag(np.ones(n - 1), 1)
         - np.diag(np.ones(n - 1), -1))
    K = (K + K.T) / 2
    w2, V = np.linalg.eigh(K)                # MATLAB eig on a symmetric matrix
    ix = np.argsort(w2, kind="stable")       # MATLAB sort(...,'ascend')
    w2, V = w2[ix].real, V[:, ix].real
    sg = np.sign(V[0]); sg[sg == 0] = 1
    V = V * sg
    e = np.zeros(n); e[j0 - 1] = 1.0         # MATLAB e(j0) = 1
    L = dict(n=n, j0=j0, k3=k3, zeta1=zeta1, wn=np.sqrt(w2), Phi=V,
             psi=V.T @ e, ci=np.full(n, 2 * zeta1 * np.sqrt(w2[0])))
    return L


# -------------------------------------------------------------- rhs_free.m --
def rhs_free(L):
    n, w2, c, psi, k3 = L["n"], L["wn"] ** 2, L["ci"], L["psi"], L["k3"]

    def f(t, z):
        a, v = z[rng1(1, n)], z[rng1(n + 1, 2 * n)]
        return np.concatenate([v, -c * v - w2 * a - psi * (k3 * (psi @ a) ** 3)])
    return f


# ------------------------------------------------------------ rhs_forced.m --
def rhs_forced(L, A, Om, ph=None):
    Om = np.atleast_1d(np.asarray(Om, dtype=float))
    A = np.atleast_1d(np.asarray(A, dtype=float))
    ph = np.zeros_like(Om) if ph is None else np.atleast_1d(np.asarray(ph, float))
    n, w2, c, psi, k3 = L["n"], L["wn"] ** 2, L["ci"], L["psi"], L["k3"]

    def f(t, z):
        a, v = z[rng1(1, n)], z[rng1(n + 1, 2 * n)]
        return np.concatenate([v, -c * v - w2 * a - psi * (k3 * (psi @ a) ** 3)
                               + psi * np.sum(A * np.cos(Om * t + ph))])
    return f


# ----------------------------------------------------------- hilbert_fft.m --
def hilbert_fft(x):
    x = np.asarray(x).ravel()
    N = x.size
    h = np.zeros(N)
    if N % 2 == 0:
        h[rng1(1, 1)] = 1
        h[rng1(N // 2 + 1, N // 2 + 1)] = 1
        h[rng1(2, N // 2)] = 2
    else:
        h[rng1(1, 1)] = 1
        h[rng1(2, (N + 1) // 2)] = 2
    return np.fft.ifft(np.fft.fft(x) * h)


# --------------------------------------------------------------- realify.m --
def realify(v):
    v = np.asarray(v).ravel()
    i1 = int(np.argmax(np.abs(v)))           # MATLAB [~,i1]=max(abs(v))
    v = v / v[i1]
    R = np.column_stack([v.real, v.imag])
    R = R - R.mean(0)
    s = svdvals(R)
    if s.size < 2:
        s = np.append(s, 0.0)
    mpc = (s[0] ** 2 - s[1] ** 2) ** 2 / (s[0] ** 2 + s[1] ** 2) ** 2
    th = np.angle(np.sum((v / np.linalg.norm(v)) ** 2)) / 2
    u = (v * np.exp(-1j * th)).real
    u = u / np.linalg.norm(u)
    i2 = int(np.argmax(np.abs(u)))
    return u * np.sign(u[i2]), mpc


# ------------------------------------------------------------------ gcvd.m --
def _mrdivide(A, B):
    return np.linalg.solve(B.T, A.T).T        # MATLAB A/B


def gcvd(disp_, vel, acc, center=True):
    X1 = np.hstack([disp_, vel])
    X2 = np.hstack([vel, acc])
    if center:
        X1 = X1 - X1.mean(0)
        X2 = X2 - X2.mean(0)
    K11, K12 = X1.T @ X1, X1.T @ X2
    K21, K22 = X2.T @ X1, X2.T @ X2
    Q = 0.25 * (_mrdivide(K21, K11) + _mrdivide(K22, K12)
                + _mrdivide(K11, K21) + _mrdivide(K12, K22))
    _, V = np.linalg.eig(Q)
    VmH = np.linalg.inv(V).conj().T
    Xi = X1 @ VmH
    S2 = np.linalg.pinv(Xi) @ (X2 @ VmH)
    lam, T = np.linalg.eig(S2)                  # [T, D] = eig(S2); lam = diag(D)
    # Canonical shapes, mirroring gcvd.m since 2026-10-22:
    #   Phi = V/(T')  in MATLAB  ==  V @ inv(T).conj().T  here   (Phi = V T^-H)
    #   U   = Xi*T
    # No matchpairs/linear_sum_assignment: under Phi each pole already sits on
    # its own column, so the superseded one-to-one assignment is not performed.
    Phi = V @ np.linalg.inv(T).conj().T
    U = Xi @ T
    return lam, Phi, U


# --------------------------------------------------------- gcvd_windowed.m --
def gcvd_windowed(disp_, vel, acc, n_windows=48, n_track=3, center=True, M=None):
    m, n = disp_.shape
    ed = np.floor(np.linspace(0, m, n_windows + 1)).astype(int)
    out = {k: np.full((n_windows, n_track), np.nan)
           for k in ("wn", "zeta", "mpc", "amp", "energy")}
    out["shape"] = np.full((n_windows, n_track, n), np.nan)
    out["edges"] = ed
    prev_shape, prev_wn = None, None
    for w in range(1, n_windows + 1):
        rows = rng1(ed[w - 1] + 1, ed[w])
        lam, V, _ = gcvd(disp_[rows], vel[rows], acc[rows], center)
        ii = np.flatnonzero(lam.imag > 1e-6)
        o = np.argsort(np.abs(lam[ii]), kind="stable")
        idx = ii[o]
        nc = idx.size
        c_wn = np.zeros(nc); c_ze = np.zeros(nc)
        c_sh = np.zeros((n, nc)); c_mp = np.zeros(nc)
        for j in range(1, nc + 1):
            i = idx[j - 1]
            c_wn[j - 1] = abs(lam[i])
            c_ze[j - 1] = -lam[i].real / abs(lam[i])
            uu, mm = realify(V[rng1(1, n), i])
            c_sh[:, j - 1] = uu
            c_mp[j - 1] = mm
        if prev_wn is None:
            sel = np.arange(1, min(n_track, nc) + 1)
        else:
            npv = prev_wn.size
            cost = np.zeros((npv, nc))
            for k in range(1, npv + 1):
                for j in range(1, nc + 1):
                    cost[k - 1, j - 1] = (
                        (1 - (c_sh[:, j - 1] @ prev_shape[:, k - 1]) ** 2)
                        + abs(np.log(c_wn[j - 1] / prev_wn[k - 1])))
            from scipy.optimize import linear_sum_assignment
            sel = linear_sum_assignment(cost)[1] + 1     # matchpairs stand-in
        dX = disp_[rows] - disp_[rows].mean(0)
        for k in range(1, sel.size + 1):
            j = sel[k - 1]
            if j == 0:
                continue
            out["wn"][w - 1, k - 1] = c_wn[j - 1]
            out["zeta"][w - 1, k - 1] = c_ze[j - 1]
            out["shape"][w - 1, k - 1, :] = c_sh[:, j - 1]
            out["mpc"][w - 1, k - 1] = c_mp[j - 1]
            if M is not None:
                vm = c_sh[:, j - 1] / np.sqrt(c_sh[:, j - 1] @ M @ c_sh[:, j - 1])
                a = np.sqrt(2) * (dX @ (M @ vm)).std(ddof=0)   # MATLAB std(...,1)
                out["amp"][w - 1, k - 1] = a * abs(vm[-1])
                out["energy"][w - 1, k - 1] = 0.5 * (c_wn[j - 1] * a) ** 2
        keep = sel[sel > 0]
        prev_shape = c_sh[:, keep - 1]
        prev_wn = c_wn[keep - 1]
    return out


# ------------------------------------------------------------- estimates.m --
def estimates(L, aa, vv, ac, mode, dt=0.05):
    n, psi, wn2 = L["n"], L["psi"], L["wn"] ** 2
    m = aa.shape[0]
    w = aa @ psi
    tt = np.arange(m) * dt
    gc = gcvd_windowed(aa, vv, ac, 1, 8, True, np.eye(n))
    mq = int(np.argmax(np.abs(gc["shape"][0, :, mode - 1]))) + 1
    out = {}
    out["r_gcvd"] = 2 * gc["zeta"][0, mq - 1] * gc["wn"][0, mq - 1] / L["ci"][mode - 1]
    env = np.abs(hilbert_fft(aa[:, mode - 1]))
    k = m // 20
    sl = rng1(k + 1, m - k)
    p = np.polyfit(tt[sl], np.log(env[sl]), 1)
    out["r_env"] = -2 * p[0] / L["ci"][mode - 1]
    P = -(L["k3"] * w ** 3)[:, None] * vv * psi[None, :]
    out["r_flux"] = ((L["ci"][mode - 1] - P[:, mode - 1].mean()
                      / (vv[:, mode - 1] ** 2).mean()) / L["ci"][mode - 1])
    E = np.mean(0.5 * (vv ** 2 + wn2[None, :] * aa ** 2), axis=0)
    sv = svdvals(np.hstack([aa, vv]))
    out["stiff_pct"] = 100 * (gc["wn"][0, mq - 1] / L["wn"][mode - 1] - 1)
    out["a_m"] = np.sqrt(np.mean(aa[:, mode - 1] ** 2))
    out["w_rms"] = np.sqrt(np.mean(w ** 2))
    out["e_frac"] = E[mode - 1] / E.sum()
    out["cond"] = sv[0] / sv[-1]
    return out


# ----------------------------------------------------------- energy_rates.m --
def energy_rates(L, tt, aa, vv, mode, subset=None, edge=20):
    tt = np.asarray(tt).ravel()
    m = tt.size
    subset = np.arange(1, L["n"] + 1) if subset is None else np.asarray(subset)
    Ei = 0.5 * (vv ** 2 + (L["wn"] ** 2)[None, :] * aa ** 2)
    Etot = Ei.sum(1)
    Esub = Ei[:, subset - 1].sum(1)
    k = m // edge
    sl = rng1(k + 1, m - k)
    f_sub = Ei[:, mode - 1] / Esub
    f_tot = Ei[:, mode - 1] / Etot
    cbar = (Ei @ L["ci"]) / Etot
    o = {}
    o["dlnf_sub"] = np.polyfit(tt[sl], np.log(f_sub[sl]), 1)[0]
    o["dlnf_tot"] = np.polyfit(tt[sl], np.log(f_tot[sl]), 1)[0]
    o["gamma"] = -np.polyfit(tt[sl], np.log(Esub[sl]), 1)[0]
    o["gamma_over_c"] = o["gamma"] / L["ci"][mode - 1]
    o["vacuous_over_c"] = (o["gamma"] - o["dlnf_sub"]) / L["ci"][mode - 1]
    o["cbar_mean_over_c"] = cbar[sl].mean() / L["ci"][mode - 1]
    o["r_derived"] = (cbar[sl].mean() - o["dlnf_tot"]) / L["ci"][mode - 1]
    o["share_mean"] = f_tot.mean()
    o["share_start"] = f_tot[0]
    o["share_end"] = f_tot[-1]
    return o


# ------------------------------------------------------- settle_release.m ---
def settle_release(L, A, mode, span=400.0, dt=0.05, rtol=1e-9, t_settle=6000.0):
    n = L["n"]
    s = solve_ivp(rhs_forced(L, A, L["wn"][mode - 1]), (0.0, t_settle),
                  np.zeros(2 * n), method="DOP853", rtol=rtol, atol=rtol * 1e-2,
                  t_eval=[0.0, t_settle / 2, t_settle])
    y0 = s.y[:, -1]
    tt = np.arange(round(span / dt)) * dt
    d = solve_ivp(rhs_free(L), (tt[0], tt[-1]), y0, method="DOP853",
                  rtol=rtol, atol=rtol * 1e-2, t_eval=tt)
    aa = d.y[rng1(1, n)].T
    vv = d.y[rng1(n + 1, 2 * n)].T
    ac = (-(L["ci"][None, :]) * vv - ((L["wn"] ** 2)[None, :]) * aa
          - (L["k3"] * (aa @ L["psi"]) ** 3)[:, None] * L["psi"][None, :])
    return tt, aa, vv, ac, y0


# --------------------------------------------------------- monodromy.m ------
def monodromy(L, wq, tau, T, rtol=1e-10, atol=None):
    atol = rtol * 1e-2 if atol is None else atol
    n = L["n"]
    W2, C, PSI, k3 = L["wn"] ** 2, L["ci"], L["psi"], L["k3"]
    wq = np.asarray(wq).ravel(); tau = np.asarray(tau).ravel()

    def mono_rhs(t, y):
        M = y.reshape(2 * n, 2 * n, order="F")        # MATLAB column-major
        dK = 3 * k3 * np.interp(t % T, tau, wq) ** 2
        top = M[rng1(1, n), :]
        bot = M[rng1(n + 1, 2 * n), :]
        return np.vstack([bot, -(W2[:, None] * top)
                          - dK * np.outer(PSI, PSI @ top)
                          - C[:, None] * bot]).ravel(order="F")

    y0 = np.eye(2 * n).ravel(order="F")
    s = solve_ivp(mono_rhs, (0.0, T), y0, method="DOP853", rtol=rtol, atol=atol,
                  t_eval=[0.0, T / 2, T])
    Mm = s.y[:, -1].reshape(2 * n, 2 * n, order="F")
    return float(np.max(np.abs(np.linalg.eigvals(Mm))))


# ------------------------------------------------------------- floquet.m ----
def floquet(mu, A=2e-2, mode=7, n=24, j0=2, zeta1=0.010, t_probe=2000.0, npts=801):
    L = lattice(n, j0, mu / A ** 2, zeta1)
    Om = L["wn"][mode - 1]
    T = 2 * np.pi / Om
    tev = np.linspace(t_probe, t_probe + T, npts)
    s = solve_ivp(rhs_forced(L, A, Om), (0.0, tev[-1]), np.zeros(2 * n),
                  method="DOP853", rtol=1e-11, atol=1e-14,
                  t_eval=np.concatenate([[0.0], tev]))
    Y = s.y.T
    wq = Y[rng1(2, npts + 1)][:, rng1(1, n)] @ L["psi"]
    tau = np.linspace(0.0, T, npts)
    o = dict(mu=mu, k3=L["k3"], w_peak=float(np.abs(wq).max()))
    o["k3_wpeak2"] = L["k3"] * o["w_peak"] ** 2
    o["max_multiplier"] = monodromy(L, wq, tau, T)
    o["unstable"] = o["max_multiplier"] > 1.0
    o["crest_factor"] = o["w_peak"] / np.sqrt(np.mean(wq ** 2))
    return o


# ------------------------------------------------------------- boucwen.m ----
def boucwen(kh=1.0, alpha=0.1, uy=0.002, n=24, j0=2, zeta1=0.010):
    L = lattice(n, j0, 0.0, zeta1)
    Kmod = np.diag(L["wn"] ** 2) + kh * np.outer(L["psi"], L["psi"])
    Kmod = (Kmod + Kmod.T) / 2
    lam, V = np.linalg.eigh(Kmod)
    ix = np.argsort(lam, kind="stable")
    lam, V = lam[ix].real, V[:, ix].real
    sg = np.sign(V[0]); sg[sg == 0] = 1
    V = V * sg
    S = dict(n=n, wn=np.sqrt(lam), psit=V.T @ L["psi"], c=L["ci"][0], kh=kh,
             alpha=alpha, uy=uy, beta=0.5 / uy, gamma=0.5 / uy, V=V, chain=L)

    def make(A, Om):
        W2, c, pt = S["wn"] ** 2, S["c"], S["psit"]
        al, be, ga = alpha, S["beta"], S["gamma"]

        def f(t, y):
            p = y[rng1(1, n)]; v = y[rng1(n + 1, 2 * n)]; z = y[2 * n + 1 - 1]
            w = pt @ p; wd = pt @ v
            resid = (al * kh * w + (1 - al) * kh * z) - kh * w
            zd = wd - be * abs(wd) * z - ga * wd * abs(z)
            acc = -c * v - W2 * p - pt * resid + pt * (A * np.cos(Om * t))
            return np.concatenate([v, acc, [zd]])
        return f
    S["rhs"] = make
    return S


def boucwen_record(A, mode=7, t_settle=4000.0, span=400.0, dt=0.05,
                   rtol=1e-9, edge=20):
    S = boucwen()
    n = S["n"]
    s = solve_ivp(S["rhs"](A, S["wn"][mode - 1]), (0.0, t_settle),
                  np.zeros(2 * n + 1), method="DOP853", rtol=rtol, atol=rtol * 1e-2,
                  t_eval=[0.0, t_settle / 2, t_settle])
    y0 = s.y[:, -1]
    tt = np.arange(round(span / dt)) * dt
    d = solve_ivp(S["rhs"](0, 0), (tt[0], tt[-1]), y0, method="DOP853",
                  rtol=rtol, atol=rtol * 1e-2, t_eval=tt)
    pp = d.y[rng1(1, n)].T
    vv = d.y[rng1(n + 1, 2 * n)].T
    z = d.y[2 * n + 1 - 1]
    Ei = 0.5 * (vv ** 2 + (S["wn"] ** 2)[None, :] * pp ** 2)
    Etot = Ei.sum(1)
    m = tt.size
    k = m // edge
    sl = rng1(k + 1, m - k)
    sub = rng1(max(1, mode - 2), min(n, mode + 2)) + 1
    env = np.abs(hilbert_fft(pp[:, mode - 1]))
    w = pp @ S["psit"]
    return dict(
        A=A, slip_ratio=np.abs(z).max() / S["uy"],
        w_rms=np.sqrt(np.mean(w ** 2)),
        e_frac=Ei[:, mode - 1].mean() / Etot.mean(),
        gamma_tot_over_c=-np.polyfit(tt[sl], np.log(Etot[sl]), 1)[0] / S["c"],
        gamma_sub_over_c=-np.polyfit(tt[sl], np.log(Ei[:, sub - 1].sum(1)[sl]),
                                     1)[0] / S["c"],
        r_env_over_c=-2 * np.polyfit(tt[sl], np.log(env[sl]), 1)[0] / S["c"],
        y0=y0)
