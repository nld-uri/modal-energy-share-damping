"""Periodic-orbit (NNM) continuation for conservative mechanical systems.

Shooting on the time-reversal symmetry of  M xdd + K x + fnl(x) = 0 :
an orbit launched from rest, x(0)=x0, xd(0)=0, is periodic with period T
iff xd(T/2)=0.  Unknowns (x0, T); pseudo-arc-length continuation with an
analytic Jacobian from the variational equations.  Floquet multipliers are
obtained from the monodromy matrix over the full period.
"""
import numpy as np
from scipy.integrate import solve_ivp


class ConservativeSystem:
    """M xdd + K x + fnl(x) = 0, with fnl localised through a single column."""

    def __init__(self, M, K, fnl, dfnl):
        self.M, self.K = M, K
        self.Minv = np.linalg.inv(M)
        self.fnl, self.dfnl = fnl, dfnl
        self.n = M.shape[0]

    def rhs(self, t, y):
        n = self.n
        x, v = y[:n], y[n:2*n]
        return np.concatenate([v, -self.Minv @ (self.K @ x + self.fnl(x))])

    def rhs_var(self, t, y):
        """State + variational flow; y = [x, v, vec(Phi)] with Phi (2n x 2n)."""
        n = self.n
        x, v = y[:n], y[n:2*n]
        Phi = y[2*n:].reshape(2*n, 2*n)
        A = np.zeros((2*n, 2*n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -self.Minv @ (self.K + self.dfnl(x))
        return np.concatenate([v, -self.Minv @ (self.K @ x + self.fnl(x)),
                               (A @ Phi).ravel()])

    def energy(self, x, v, Vnl):
        return 0.5*v @ self.M @ v + 0.5*x @ self.K @ x + Vnl(x)

    # ---- shooting residual and Jacobian -------------------------------------
    def _halfperiod(self, x0, T, rtol=1e-11, atol=1e-13):
        n = self.n
        y0 = np.concatenate([x0, np.zeros(n), np.eye(2*n).ravel()])
        s = solve_ivp(self.rhs_var, (0, T/2), y0, method='DOP853',
                      rtol=rtol, atol=atol, dense_output=False)
        yf = s.y[:, -1]
        return yf[:n], yf[n:2*n], yf[2*n:].reshape(2*n, 2*n)

    def shoot(self, x0, T):
        """Residual H = xd(T/2) and its Jacobian wrt (x0, T)."""
        n = self.n
        xh, vh, Phi = self._halfperiod(x0, T)
        H = vh
        dH_dx0 = Phi[n:2*n, :n]                       # d xd(T/2) / d x0
        acc = -self.Minv @ (self.K @ xh + self.fnl(xh))
        dH_dT = 0.5 * acc                             # d xd(T/2) / d T
        return H, dH_dx0, dH_dT, xh

    def monodromy(self, x0, T):
        n = self.n
        y0 = np.concatenate([x0, np.zeros(n), np.eye(2*n).ravel()])
        s = solve_ivp(self.rhs_var, (0, T), y0, method='DOP853',
                      rtol=1e-11, atol=1e-13)
        return s.y[2*n:, -1].reshape(2*n, 2*n)


def _newton(sys, u, w_T, tang=None, u_prev=None, ds=None, tol=1e-10, maxit=12):
    """Correct u=(x0,T); with tang/u_prev/ds appended as the arc-length equation."""
    n = sys.n
    for _ in range(maxit):
        H, Jx, JT, _ = sys.shoot(u[:n], u[n])
        J = np.column_stack([Jx, JT])
        if tang is None:                       # phase/amplitude pinned externally
            return None
        scal = np.concatenate([np.ones(n), [w_T]])
        r = np.concatenate([H, [np.dot(tang*scal, (u - u_prev)*scal) - ds]])
        Jf = np.vstack([J, (tang*scal*scal)])
        du = np.linalg.solve(Jf, -r)
        u = u + du
        if np.linalg.norm(H) < tol and np.linalg.norm(du) < tol:
            return u, J
    return (u, J) if np.linalg.norm(H) < 1e-6 else (None, None)


def refine_orbit(sys, x0, T, pin_idx, pin_val, tol=1e-11, maxit=40):
    """Newton on H(x0,T)=0 with the component x0[pin_idx] pinned (fixes the
    one-parameter family), used to land a seed orbit before continuation."""
    n = sys.n
    u = np.concatenate([x0, [T]])
    for _ in range(maxit):
        H, Jx, JT, _ = sys.shoot(u[:n], u[n])
        r = np.concatenate([H, [u[pin_idx] - pin_val]])
        row = np.zeros(n+1); row[pin_idx] = 1.0
        Jf = np.vstack([np.column_stack([Jx, JT]), row])
        du = np.linalg.lstsq(Jf, -r, rcond=None)[0]
        u = u + 0.7*du if np.linalg.norm(du) > 0.05 else u + du
        if np.linalg.norm(r) < tol:
            return u[:n], u[n]
    return (u[:n], u[n]) if np.linalg.norm(r) < 1e-7 else (None, None)


def continue_branch(sys, x0, T0, ds=2e-3, nsteps=400, w_T=1.0, direction=+1,
                    amp_max=np.inf, amp_fun=None, T_bounds=(1e-3, 1e4),
                    adapt=True, monodromy_every=0):
    """Pseudo-arc-length continuation from a seed periodic orbit (x0, T0)."""
    n = sys.n
    scal = np.concatenate([np.ones(n), [w_T]])
    u = np.concatenate([x0, [T0]])

    H, Jx, JT, _ = sys.shoot(u[:n], u[n])
    J = np.column_stack([Jx, JT])
    _, _, Vt = np.linalg.svd(J)
    tang = Vt[-1] / np.linalg.norm(Vt[-1]*scal)
    tang *= direction * np.sign(tang[n]) if abs(tang[n]) > 1e-12 else direction

    out, ds_max = [], ds*8
    for step in range(nsteps):
        u_prev, ok, J = u.copy(), None, None
        for trial in range(8):
            ok, J = _newton(sys, u_prev + ds*tang, w_T, tang, u_prev, ds)
            if ok is not None and T_bounds[0] < ok[n] < T_bounds[1]:
                break
            ds *= 0.4
        if ok is None:
            break
        u = ok
        _, _, Vt = np.linalg.svd(J)
        t_new = Vt[-1] / np.linalg.norm(Vt[-1]*scal)
        if np.dot(t_new*scal, tang*scal) < 0:
            t_new = -t_new
        tang = t_new
        rec = {'x0': u[:n].copy(), 'T': u[n], 'w': 2*np.pi/u[n], 'ds': ds}
        if monodromy_every and step % monodromy_every == 0:
            mu = np.linalg.eigvals(sys.monodromy(u[:n], u[n]))
            rec['floquet_max'] = np.abs(mu).max()
        out.append(rec)
        if amp_fun is not None and amp_fun(u[:n]) > amp_max:
            break
        if adapt:
            ds = min(ds*1.08, ds_max)
    return out
