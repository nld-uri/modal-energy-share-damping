"""Generalised characteristic value decomposition (GCVD) for energy-dependent
modal identification and linear-projection reduced-order modelling.

Implements Stein & Chelidze, Nonlinear Dyn 113(1):25-45 (2025), Sec. 4:

    X1 = Xi Sigma1 V^H ,   X2 = Xi Sigma2 V^H ,   X2 = dX1/dt

with X1 = [x, xdot], X2 = [xdot, xddot] stacked column-wise over snapshots.
V is recovered from the balanced eigenproblem

    Q = 1/4 (K21 K11^-1 + K22 K12^-1 + K11 K21^-1 + K12 K22^-1) = V Lam V^-1,
    Kij = Xi^H Xj ,

then Sigma1 = I, Xi = X1 V^-H and Sigma2 = Xi^+ X2 V^-H, whose EIGENVALUES are
the characteristic values (system poles at the response amplitude of the
window). Sigma2 is diagonal only for an exactly linear, noise-free record, so
reading its diagonal -- the natural reading of the relation above, and the one
used until this study -- corrupts the damping while leaving the frequency
intact; see the note in `gcvd` below.

Two adaptations beyond the reference implementation, both required for a
bistable structure and both switchable:
  * per-window mean removal, so each window is linearised about the *local*
    equilibrium rather than the origin (`center=True`);
  * mode tracking across windows by combined MAC and log-frequency proximity,
    solved as a linear assignment problem.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _rdiv(A, B):
    return np.linalg.solve(B.T, A.T).T                      # A @ inv(B)


def gcvd(disp, vel, acc, center=True):
    """Decompose one window of synchronously sampled response.

    Parameters
    ----------
    disp, vel, acc : (m, n) arrays
        Displacement, velocity and acceleration at n spatial stations over
        m snapshots. Velocity/acceleration may be measured or obtained by
        feeding the displacements back through the equations of motion.
    center : bool
        Remove each column's window mean before forming the covariances.

    Returns
    -------
    lam : (2n,) complex
        Characteristic values (poles); lam = -zeta*wn +/- i*wn*sqrt(1-zeta^2).
    V : (2n, 2n) complex
        Characteristic modes in the state space; rows 0..n-1 of a column give
        the displacement mode shape.
    Xi : (m, 2n) complex
        Common temporal basis.
    """
    X1 = np.hstack([disp, vel])
    X2 = np.hstack([vel, acc])
    if center:
        X1 = X1 - X1.mean(0)
        X2 = X2 - X2.mean(0)
    K11, K12 = X1.T @ X1, X1.T @ X2
    K21, K22 = X2.T @ X1, X2.T @ X2
    Q = 0.25*(_rdiv(K21, K11) + _rdiv(K22, K12) + _rdiv(K11, K21) + _rdiv(K12, K22))
    _, V = np.linalg.eig(Q)
    VmH = np.linalg.inv(V).conj().T                          # V^-H
    Xi = X1 @ VmH                                            # Sigma1 = I
    S2 = np.linalg.pinv(Xi) @ (X2 @ VmH)                     # Sigma2

    # The characteristic values are the EIGENVALUES of Sigma2, not its diagonal.
    # Sigma2 is diagonal only to the extent that the record is linear and
    # noise-free; measured leakage here is 1e-4 to 2e-3 of the diagonal entry.
    # That is negligible for the frequency, which is |lam| ~ O(1), but the
    # damping is Re(lam) ~ 1e-3 |lam|, i.e. the SAME order as the leakage, so
    # reading the diagonal corrupts the damping while leaving the frequency
    # correct to four digits. On this study's records the diagonal readout is
    # wrong by up to 111 per cent and swings with the analysis-window length;
    # diagonalising costs nothing and holds to 0.35 per cent, agreeing with an
    # independent ERA realisation on the same data to 0.4 per cent.
    #
    # SHAPES are the eigenvectors of Q by convention, not rotated by the
    # eigenvectors of Sigma2. An earlier comparison of rotated shapes with the
    # equivalent-linearised mode on the companion study's beam record used an
    # incorrect transform (V T instead of V T^-H) and does not support a
    # preference; with the corrected transform the outcome is not uniform
    # (corrected 2026-10-11). Each eigenvalue
    # is assigned one-to-one to the Q-mode carrying most of its eigenvector's
    # weight. On this study's records the tracked damping is identical under
    # either shape convention (readout_final_check.py: max difference 0.0 over
    # the 25 grid and 15 span-sweep records).
    lam, T = np.linalg.eig(S2)                               # Sigma2 = T Lam T^-1
    # Canonical shapes since 2026-10-22, matching matlab/gcvd.m and the
    # programme toolbox: Phi = V T^-H paired with U = Xi T. The one-to-one
    # assignment the unrotated-V convention needed is not performed, because
    # each pole already sits on its own column of Phi.
    Phi = V @ np.linalg.inv(T).conj().T
    U = Xi @ T
    return lam, Phi, U


def realify(v):
    """Phase-rotate a complex mode to maximum collinearity.

    Returns the unit-norm real mode shape and the modal phase collinearity
    (MPC), which is 1 for a purely normal (real) mode.
    """
    v = v / v[np.argmax(np.abs(v))]
    R = np.column_stack([v.real, v.imag]); R = R - R.mean(0)
    s = np.linalg.svd(R, compute_uv=False)
    mpc = (s[0]**2 - s[1]**2)**2 / (s[0]**2 + s[1]**2)**2
    th = np.angle(np.sum((v/np.linalg.norm(v))**2))/2
    u = (v*np.exp(-1j*th)).real
    u = u/np.linalg.norm(u)
    return u*np.sign(u[np.argmax(np.abs(u))]), mpc


def windowed_gcvd(disp, vel, acc, n_windows=48, n_track=3, center=True, M=None):
    """Split a free-decay record into equal windows, decompose each, and track
    `n_track` resonances across windows.

    `M` (n x n mass matrix) enables modal-amplitude and modal-energy estimates;
    omit it to get frequencies, damping ratios and shapes only.

    Returns a dict of (n_windows, n_track) arrays -- 'wn', 'zeta', 'mpc',
    'amp' and 'energy' -- plus 'shape' of size (n_windows, n_track, n) and the
    window index bounds 'edges'.
    """
    m, n = disp.shape
    ed = np.linspace(0, m, n_windows+1).astype(int)
    out = {k: np.full((n_windows, n_track), np.nan)
           for k in ('wn', 'zeta', 'mpc', 'amp', 'energy')}
    out['shape'] = np.full((n_windows, n_track, n), np.nan)
    out['edges'] = ed
    prev = None
    for w in range(n_windows):
        s = slice(ed[w], ed[w+1])
        lam, V, _ = gcvd(disp[s], vel[s], acc[s], center=center)
        idx = sorted([i for i in range(len(lam)) if lam[i].imag > 1e-6],
                     key=lambda i: abs(lam[i]))
        cand = [(abs(lam[i]), -lam[i].real/abs(lam[i])) + realify(V[:n, i]) for i in idx]
        if prev is None:
            sel = list(range(min(n_track, len(cand))))
        else:
            # Size the cost matrix by what was actually carried forward, not by
            # n_track: a window yielding fewer than n_track candidate poles
            # (rank-deficient or heavily filtered record) would otherwise index
            # past the end of `prev` on the next window.
            cost = np.array([[(1 - (cand[j][2] @ prev[k][2])**2)
                              + abs(np.log(cand[j][0]/prev[k][0]))
                              for j in range(len(cand))] for k in range(len(prev))])
            sel = linear_sum_assignment(cost)[1]
        dX = disp[s] - disp[s].mean(0)
        for k, j in enumerate(sel):
            wn, zeta, shape, mpc = cand[j]
            out['wn'][w, k], out['zeta'][w, k] = wn, zeta
            out['shape'][w, k], out['mpc'][w, k] = shape, mpc
            if M is not None:
                vm = shape/np.sqrt(shape @ M @ shape)        # mass-normalised
                a = np.sqrt(2)*(dX @ (M @ vm)).std()
                out['amp'][w, k] = a*abs(vm[-1])
                out['energy'][w, k] = 0.5*(wn*a)**2
        prev = [cand[j] for j in sel]
    return out


def robustness_basis(shapes, rank=None, tol=1e-3):
    """Resonance robustness matrix S = [V^m_1 ... V^m_p] for one resonance and
    its orthonormal span.

    `shapes` is the (n_windows, n) stack of characteristic modes for a single
    tracked resonance. Returns (basis, singular_values); `rank` overrides the
    automatic cut at `tol` times the leading singular value.
    """
    S = np.asarray(shapes).T
    U, sv, _ = np.linalg.svd(S)
    r = rank if rank is not None else max(1, int(np.sum(sv/sv[0] > tol)))
    return U[:, :r], sv/sv[0]


def project(M, K, fnl, P):
    """Galerkin projection onto a constant basis P (n x k): x = P q.

    Returns (Mr, Kr, fnl_r, Pinv) with Mr = P^+ M P, Kr = P^+ K P and
    fnl_r(q) = P^+ fnl(P q).
    """
    P = P.reshape(-1, 1) if P.ndim == 1 else P
    Pinv = np.linalg.pinv(P)
    return Pinv @ M @ P, Pinv @ K @ P, (lambda q: Pinv @ fnl(P @ q)), Pinv
