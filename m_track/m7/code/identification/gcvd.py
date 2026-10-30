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

The SHAPES are not the columns of V. With Sigma2 = T Lam2 T^-1 the canonical
characteristic shapes are Phi = V T^-H, paired with U = Xi T, which is the only
choice for which both X1 = U Phi^H and X2 = U Lam2 Phi^H hold -- i.e. for which
the decomposition is the diagonal form written above. This module is a thin
adapter: the decomposition is computed by the programme's canonical NCVD
toolbox (`ncvd.py`) so that no result depends on a second implementation. The
names `gcvd` and `windowed_gcvd` are kept for the call sites that predate the
toolbox.

Two adaptations beyond the reference implementation, both required for a
bistable structure and both switchable:
  * per-window mean removal, so each window is linearised about the *local*
    equilibrium rather than the origin (`center=True`);
  * mode tracking across windows by combined MAC and log-frequency proximity,
    solved as a linear assignment problem.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from ncvd import ncvd as _ncvd, realify  # noqa: F401  (realify re-exported)


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
    # ONE IMPLEMENTATION, for this study's Python results. The decomposition is not
    # recomputed here: it is delegated to the canonical NCVD toolbox (`ncvd.py`,
    # byte-identical to the copy in the companion subspace-closure study), so no
    # result reported by this study can depend on a second Python implementation
    # drifting from the first. Two arms are deliberately NOT routed through it and
    # still use the columns of V directly: `matlab/gcvd.m` with its back-translation
    # `matlab_support/verification/mlmirror.py`, whose job is to mirror the MATLAB
    # code rather than this one, and `matlab_support/identification/gcvd.py`, which
    # feeds that cross-check. Making those canonical additionally requires
    # regenerating `matlab/reference.mat` under MATLAB.
    #
    # The characteristic values are the EIGENVALUES of Sigma2, not its diagonal.
    # Sigma2 is diagonal only to the extent that the record is linear and
    # noise-free; measured leakage here is 1e-4 to 2e-3 of the diagonal entry.
    # That is negligible for the frequency, which is |lam| ~ O(1), but the
    # damping is Re(lam) ~ 1e-3 |lam|, i.e. the SAME order as the leakage, so
    # reading the diagonal corrupts the damping while leaving the frequency
    # correct to four digits. On this study's records the diagonal readout is
    # wrong by up to 128 per cent on the 25 single-mode records and swings with
    # the analysis-window length; the eigenvalue readout costs nothing and holds
    # to 0.35 per cent, agreeing with an independent ERA realisation on the same
    # data to 0.4 per cent.
    #
    # SHAPES are the canonical regression shapes Phi = V T^-H, not the
    # eigenvectors V of Q. The grounds are algebraic and are set out in the
    # companion study: Phi is the only choice for which X2 = U Lam Phi^H holds,
    # and under it each characteristic pair carries its own mode's entire
    # eigenvector weight, so the one-to-one assignment heuristic that V requires
    # is never invoked. Switching this study from V to Phi left every reported
    # quantity unchanged -- see verify_paper3.py, which passes unaltered.
    rec = _ncvd(disp, vel, acc, center=center)
    return rec['lam'], rec['Phi'], rec['U']


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
