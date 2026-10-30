"""Fixed-fixed mass-spring lattice with one grounded cubic spring.

The point of this system is its SPECTRUM. A beam's omega_i^2 grows as i^4, which
filters every high-mode component out of an amplitude-dependent mode-shape
correction (the correction goes like dK_im/(omega_m^2 - omega_i^2)). A lattice
dispersion relation saturates at the band edge, so the denominators stop growing
and the corrections keep their high-mode content. This is the system class where
robustness directions can add to a modal basis instead of duplicating it.

Modal coordinates throughout: M = I, K = diag(omega_i^2), and the nonlinear
station enters as the rank-one direction psi = Phi^T e_j0. That is the SAME
nonlinearity form as the attached beam, deliberately -- so any difference in
behaviour is attributable to the spectrum and not to the nonlinearity.

    a_ddot_i + c_i a_dot_i + omega_i^2 a_i + psi_i k3 (psi . a)^3
        = psi_i A cos(Omega t)

Forcing is applied at the nonlinear station and, following the convention used
throughout this project, at a frequency near a low natural frequency -- the free
decay that builds the basis does not excite high modes, so a ROM built from it
cannot be expected to hold for high-frequency excitation.
"""
import numpy as np
from scipy.integrate import solve_ivp

N_DEF, J0_DEF = 24, 7


def chain_modes(n=N_DEF):
    K = (np.diag(2*np.ones(n)) - np.diag(np.ones(n - 1), 1)
         - np.diag(np.ones(n - 1), -1))
    w2, Phi = np.linalg.eigh(K)
    # Canonicalise the sign of each eigenvector. numpy's eigh fixes the sign
    # arbitrarily (it differs between LAPACK builds, numpy versions and other
    # languages -- MATLAB's eig disagrees on 12 of these 24 modes), so every
    # SCALAR result is invariant but every VECTOR one is not reproducible
    # without this. First-row-positive is unambiguous here: the smallest
    # magnitude in the first row is 0.0354.
    Phi = Phi * np.sign(Phi[0])
    return np.sqrt(w2), Phi


class Lattice:
    def __init__(self, n=N_DEF, j0=J0_DEF, k3=0.0, zeta1=0.010):
        self.n, self.j0, self.k3 = int(n), int(j0), float(k3)
        self.wn, self.Phi = chain_modes(self.n)
        e = np.zeros(self.n); e[self.j0] = 1.0
        self.psi = self.Phi.T @ e
        self.zeta1 = float(zeta1)
        self.ci = np.full(self.n, 2*self.zeta1*self.wn[0])

    # ---- free response ------------------------------------------------------
    def rhs_free(self):
        wn2, ci, psi, k3, n = self.wn**2, self.ci, self.psi, self.k3, self.n

        def f(t, z):
            a, v = z[:n], z[n:]
            return np.concatenate([v, -ci*v - wn2*a - psi*(k3*(psi @ a)**3)])
        return f

    def decay(self, y0, t, rtol=1e-10):
        f = self.rhs_free()
        s = solve_ivp(f, (t[0], t[-1]), y0, t_eval=t, method='DOP853',
                      rtol=rtol, atol=rtol*1e-2)
        a, v = s.y[:self.n].T, s.y[self.n:].T
        acc = np.array([f(tt, yy)[self.n:] for tt, yy in zip(s.t, s.y.T)])
        return a, v, acc

    # ---- forced response, full and reduced ----------------------------------
    def rhs_forced(self, A, Om):
        wn2, ci, psi, k3, n = self.wn**2, self.ci, self.psi, self.k3, self.n

        def f(t, z):
            a, v = z[:n], z[n:]
            return np.concatenate([v, -ci*v - wn2*a - psi*(k3*(psi @ a)**3)
                                   + psi*(A*np.cos(Om*t))])
        return f

    def rom(self, Q):
        """Galerkin projection onto an orthonormal basis Q (n x k)."""
        Q = np.asarray(Q)
        Q = Q.reshape(-1, 1) if Q.ndim == 1 else Q
        Qo, _ = np.linalg.qr(Q)
        k = Qo.shape[1]
        Kr = Qo.T @ np.diag(self.wn**2) @ Qo
        Cr = Qo.T @ np.diag(self.ci) @ Qo
        row = self.psi @ Qo                      # station displacement from q
        pr = Qo.T @ self.psi

        def f(A, Om):
            def rhs(t, z):
                q, u = z[:k], z[k:]
                return np.concatenate([u, -Cr @ u - Kr @ q
                                       - pr*(self.k3*(row @ q)**3)
                                       + pr*(A*np.cos(Om*t))])
            return rhs
        return dict(Q=Qo, k=k, row=row, rhs=f)

    def settled_station(self, A, Om, Q=None, n_trans=400, n_per=6, npts=400,
                        rtol=1e-9):
        """Settled displacement of the nonlinear station over n_per periods."""
        T = 2*np.pi/Om
        if Q is None:
            f, dim, row, k = self.rhs_forced(A, Om), 2*self.n, self.psi, self.n
        else:
            rm = self.rom(Q)
            k = rm['k']; f = rm['rhs'](A, Om); dim = 2*k; row = rm['row']
        s = solve_ivp(f, (0, n_trans*T), np.zeros(dim), method='DOP853',
                      rtol=rtol, atol=rtol*1e-2)
        if s.status != 0:
            return None
        tt = np.linspace(0, n_per*T, n_per*npts, endpoint=False)
        d = solve_ivp(f, (0, tt[-1]), s.y[:, -1], t_eval=tt, method='DOP853',
                      rtol=rtol, atol=rtol*1e-2)
        if d.status != 0:
            return None
        return row @ d.y[:k]
