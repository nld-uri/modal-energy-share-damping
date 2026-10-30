"""NCVD toolbox: the single canonical decomposition used everywhere in this paper.

THE DECOMPOSITION
-----------------
For a state record X1 = [disp, vel] and its derivative X2 = [vel, acc],

    X1 = U Phi^H ,      X2 = U Lam Phi^H + R ,      Lam = diag(lam) ,

with U the temporal basis, Phi the characteristic mode shapes, lam the
characteristic values and R everything the diagonal form cannot represent.

WHY THIS AND NOT THE EARLIER GCVD READOUT
-----------------------------------------
The earlier readout returned the eigenvectors V of the balanced matrix Q as the
mode shapes, and paired the i-th characteristic value with the i-th column of V.
That pairing is not exact: V does not diagonalize Sigma_2. On an exactly linear,
noise-free 4-DOF decay with ANALYTIC derivatives the off-diagonal part of
Sigma_2 is 1.0e-1 of its diagonal, and the second identity above fails at the
5.2e-1 level if V is used in place of Phi. Because the pairing was approximate,
the earlier code needed an assignment heuristic (a one-to-one matching
maximizing sum |T_ij|^2) to decide which pole belonged to which shape.

Writing Sigma_2 = T Lam T^-1 and rotating the pair,

    U = Xi T ,          Phi = V T^-H ,

makes both identities exact: on the same ideal record the X2 residual is
2.1e-15 against 5.2e-1, and each lam[i] belongs to column i of Phi by
construction, so no assignment heuristic is required. Phi are the REGRESSION
shapes; they are the canonical shapes of this paper.

WHAT THE CHANGE COSTS
---------------------
The modal parameters do not move: natural frequencies are identical and
damping agrees to 2e-14. The near-linear shapes agree closely but not
uniformly: 1-MAC is 6.7e-16 on the lattice final readout and 1.6e-8 on
lattice_record, so 1.6e-8 is the bound observed, not the smaller figure. What moves
is the amplitude-dependent part. Measured on this paper's own systems, the
correction directions change by 1-MAC 0.80 on the lattice, correction
magnitudes by up to 221%, and the escape fractions rise steeply (beam canonical
release, mode 2: 0.0141 -> 0.101). Anything built on the corrections -- the
escape diagnostic, the correction ranking, the global robustness basis -- is
therefore quantitatively different under the canonical decomposition, and is
reported as such rather than carried over.

USAGE
-----
    rec = ncvd(disp, vel, acc)              # unwhitened, full state
    rec = ncvd(disp, vel, acc, rank=r)      # whitened to rank r (ill-conditioned
                                            # spectra; does not change lam)
    rec['lam'], rec['Phi'], rec['U'], rec['R']

`poles_and_shapes` is the thin adapter used by the window-tracking code.

There is ONE implementation of the algebra in this paper and it is here. The
`legacy_balanced` argument exists solely so the cost-of-change section can be
recomputed from this same code path; it must never be used to produce a result.
"""
import numpy as np

__all__ = ['ncvd', 'poles_and_shapes', 'windowed', 'realify', 'DEFAULT_RANK']

DEFAULT_RANK = None


def _rdiv(A, B):
    """A B^-1, solved rather than inverted."""
    return np.linalg.solve(B.T, A.T).T


def _balanced(K11, K12, K21, K22):
    """The balanced matrix Q, symmetric in the four second-moment blocks."""
    return 0.25*(_rdiv(K21, K11) + _rdiv(K22, K12) + _rdiv(K11, K21) + _rdiv(K12, K22))


def ncvd(disp, vel, acc, rank=None, center=False, legacy_balanced=False):
    """The canonical NCVD of a state record.

    Parameters
    ----------
    disp, vel, acc : (m, n) arrays
        Displacement, velocity and acceleration at n stations over m snapshots.
    rank : int or None
        If given, whiten X1 by its own SVD and keep `rank` directions before
        forming the decomposition. X2 = X1 A^T is invariant under invertible
        right-multiplication of the state coordinates, so whitening repairs the
        conditioning of a structure whose frequencies span decades without
        altering the characteristic values. Shapes are mapped back to the
        original coordinates before return.
    center : bool
        Remove each column mean before forming the second moments.
    legacy_balanced : bool
        Return the superseded balanced-matrix shapes V with the approximate
        pole-to-shape assignment, for the cost-of-change comparison ONLY.

    Returns
    -------
    dict with
        lam   (k,)    characteristic values, lam[i] paired with Phi[:, i]
        Phi   (n2, k) canonical characteristic shapes (regression shapes)
        U     (m, k)  temporal basis
        R     (m, n2) residual of the second identity
        V     (k, k)  eigenvectors of the balanced matrix (diagnostic)
        T     (k, k)  eigenvectors of Sigma_2 (diagnostic)
        S2            Sigma_2
        s             singular values of X1
        condT         conditioning of T
        res           ||R|| / ||X2||
        exact1        ||X1 - U Phi^H|| / ||X1||
        resid         relative least-squares residual of X2 on X1
        share  (k,)   pole-to-shape weight; identically 1 for the canonical form
    """
    X1 = np.hstack([disp, vel])
    X2 = np.hstack([vel, acc])
    if center:
        X1 = X1 - X1.mean(0)
        X2 = X2 - X2.mean(0)
    s = np.linalg.svd(X1, full_matrices=False, compute_uv=True)[1]
    if rank is None:
        Y1, Y2, back = X1, X2, None
    else:
        _, s_all, Wh = np.linalg.svd(X1, full_matrices=False)
        Wr, sr = Wh[:rank].T, s_all[:rank]
        Y1, Y2 = X1 @ (Wr/sr), X2 @ (Wr/sr)
        back, s = Wr*sr[None, :], s_all
    resid = float(np.linalg.norm(Y2 - Y1 @ np.linalg.lstsq(Y1, Y2, rcond=None)[0])
                  / max(np.linalg.norm(Y2), 1e-300))
    K11, K12 = Y1.conj().T @ Y1, Y1.conj().T @ Y2
    K21, K22 = Y2.conj().T @ Y1, Y2.conj().T @ Y2
    Q = _balanced(K11, K12, K21, K22)
    _, V = np.linalg.eig(Q)
    VmH = np.linalg.inv(V).conj().T
    Xi = Y1 @ VmH                                        # Sigma_1 = I
    S2 = np.linalg.pinv(Xi) @ (Y2 @ VmH)
    lam, T = np.linalg.eig(S2)                           # Sigma_2 = T Lam T^-1
    if legacy_balanced:
        lam, share = _assign_legacy(lam, T)
        W, U = V, Xi
    else:
        W, U = V @ np.linalg.inv(T).conj().T, Xi @ T     # Phi = V T^-H, U = Xi T
        share = np.ones(len(lam))
    Phi = W if back is None else back @ W
    R = Y2 - (U*lam) @ W.conj().T
    return dict(lam=lam, Phi=Phi, U=U, R=R, V=V, T=T, S2=S2, s=s, share=share,
                condT=float(np.linalg.cond(T)), resid=resid,
                res=float(np.linalg.norm(R)/max(np.linalg.norm(Y2), 1e-300)),
                exact1=float(np.linalg.norm(Y1 - U @ W.conj().T)
                             / max(np.linalg.norm(Y1), 1e-300)))


def _assign_legacy(lam, T):
    """Superseded one-to-one pole-to-shape assignment. Cost-of-change use only."""
    from scipy.optimize import linear_sum_assignment
    P = np.abs(T)**2
    P = P/P.sum(0, keepdims=True)
    rows, cols = linear_sum_assignment(-P)
    out = np.empty_like(lam)
    share = np.empty(len(lam))
    out[rows] = lam[cols]
    share[rows] = P[rows, cols]
    return out, share


def poles_and_shapes(disp, vel, acc, rank=None, center=False, legacy_balanced=False):
    """Adapter for the window-tracking code: (lam, shapes, resid, singular values)."""
    r = ncvd(disp, vel, acc, rank=rank, center=center, legacy_balanced=legacy_balanced)
    return r['lam'], r['Phi'], r['resid'], r['s']


def realify(v):
    """Phase-rotate a complex mode to maximum collinearity; return (shape, MPC).

    MPC is the modal phase collinearity: the eccentricity of the scatter of
    (Re v, Im v), 1 for a purely normal mode and falling toward 0 as the motion
    becomes genuinely complex.
    """
    v = np.asarray(v, complex)
    R, I = v.real - v.real.mean(), v.imag - v.imag.mean()
    S = np.array([[R @ R, R @ I], [R @ I, I @ I]])
    ev, evec = np.linalg.eigh(S)
    mpc = ((ev[1] - ev[0])/max(ev[1] + ev[0], 1e-300))**2
    th = np.arctan2(evec[1, 1], evec[0, 1])
    w = (v*np.exp(-1j*th)).real
    nrm = np.linalg.norm(w)
    if nrm < 1e-300:
        w, nrm = v.real, max(np.linalg.norm(v.real), 1e-300)
    return w/nrm, float(mpc)


def windowed(disp, vel, acc, n_windows=48, n_track=3, center=True, M=None, rank=None):
    """Split a free decay into equal windows, decompose each, track resonances.

    Interface-compatible with the cvd-nnm-rom skill's gcvd_windowed, but the
    decomposition is this toolbox's canonical NCVD rather than the skill's own
    implementation, which returns the superseded balanced-matrix shapes. The
    paper must not depend on a second implementation, so scripts that used the
    skill helper call this instead.

    Characteristic values emerge unordered, so consecutive windows are matched
    by minimising (1 - MAC) + |log(w/w_prev)| as a linear assignment problem.
    Pass M (n x n mass matrix) to also get modal amplitude and modal energy.
    Returns a dict of (n_windows, n_track) arrays 'wn', 'zeta', 'mpc', 'amp',
    'energy', plus 'shape' (n_windows, n_track, n) and 'edges'.
    """
    from scipy.optimize import linear_sum_assignment
    m, n = disp.shape
    ed = np.linspace(0, m, n_windows + 1).astype(int)
    out = {k: np.full((n_windows, n_track), np.nan)
           for k in ('wn', 'zeta', 'mpc', 'amp', 'energy')}
    out['shape'] = np.full((n_windows, n_track, n), np.nan)
    out['edges'] = ed
    prev = None
    for w in range(n_windows):
        s = slice(ed[w], ed[w + 1])
        rec = ncvd(disp[s], vel[s], acc[s], rank=rank, center=center)
        lam, V = rec['lam'], rec['Phi']
        idx = sorted([i for i in range(len(lam)) if lam[i].imag > 1e-6],
                     key=lambda i: abs(lam[i]))
        cand = [(abs(lam[i]), -lam[i].real/abs(lam[i])) + realify(V[:n, i]) for i in idx]
        if not cand:
            prev = None
            continue
        if prev is None:
            sel = list(range(min(n_track, len(cand))))
        else:
            cost = np.array([[(1 - (cand[j][2] @ prev[k][2])**2)
                              + abs(np.log(cand[j][0]/prev[k][0]))
                              for j in range(len(cand))] for k in range(n_track)])
            sel = linear_sum_assignment(cost)[1]
        chosen = []
        for k, j in enumerate(sel[:n_track]):
            if j >= len(cand):
                chosen.append(None)
                continue
            wn_, z_, shp, mpc = cand[j]
            out['wn'][w, k], out['zeta'][w, k], out['mpc'][w, k] = wn_, z_, mpc
            out['shape'][w, k] = shp
            q = disp[s] @ shp
            out['amp'][w, k] = float(np.sqrt(np.mean(q**2)))
            if M is not None:
                mk = float(shp @ M @ shp)
                out['energy'][w, k] = 0.5*mk*float(np.mean((vel[s] @ shp)**2)) \
                    + 0.5*mk*wn_**2*float(np.mean(q**2))
            chosen.append(cand[j])
        prev = [c for c in chosen if c is not None] or prev
    return out
