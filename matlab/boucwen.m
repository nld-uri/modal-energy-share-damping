function S = boucwen(kh, alpha, uy, n, j0, zeta1)
%BOUCWEN  Hysteretic station element and its stuck-basis reference structure.
%
%   Port of  system()  and  rhs()  in  friction_boundary.py.
%
%   S = BOUCWEN()                              study defaults
%   S = BOUCWEN(kh, alpha, uy, n, j0, zeta1)
%   f = S.rhs(A, Om)   ->  @(t,y) for the 2n+1 state y = [p; p_dot; z]
%   f = S.rhs(0, 0)    ->  the free (released) system
%
%   WHY THE BASIS CHANGES.  At small amplitude a Bouc-Wen element is a linear
%   spring of stiffness kh, which changes the mode shapes.  The reference
%   structure must therefore be the chain PLUS the stuck element stiffness,
%
%       Kmod = diag(wn.^2) + kh*psi*psi.'  ->  eigen -> S.wn, S.psit
%
%   and the residual force driving the modal equations is the element's
%   DEPARTURE from that spring,
%
%       resid = (alpha*kh*w + (1-alpha)*kh*z) - kh*w .
%
%   Tracked on that basis the elastic limit recovers gamma_tot/c -> 1 and a
%   clean mode, which is the control the conservative/dissipative comparison
%   needs.  Getting the basis wrong (using the bare chain modes) does not
%   error -- it quietly shifts every identified frequency.
%
%   THE ELEMENT.  Bouc-Wen with exponent 1 and beta = gamma = 0.5/uy:
%
%       z_dot = w_dot - beta |w_dot| z - gamma w_dot |z| .
%
%   This nonlinearity does NET WORK over a cycle, so unlike the cubic it
%   changes the estimand itself: the identified damping is not biased, it is
%   measuring a different quantity (an amplitude-dependent effective damping).
%
%   The abs() makes the right-hand side continuous but NOT differentiable at
%   velocity and hysteretic-force reversals.  A Runge-Kutta pair loses its
%   formal order crossing each kink, and DOP853 and ode89 place their steps
%   differently there, so integrated Bouc-Wen results are the loosest check in
%   selftest.m.
%
%   Eigenvector signs are canonicalised to first-row-positive, as in
%   lattice.m; verified bit-exact neutral in Python (0.0 change with 11 of 24
%   signs flipped).  The smallest first-row entry of this basis is 1.1e-3,
%   far above the ~1e-14 eigenvector noise floor, so the sign is unambiguous.

if nargin < 1 || isempty(kh),    kh    = 1.0;   end
if nargin < 2 || isempty(alpha), alpha = 0.1;   end
if nargin < 3 || isempty(uy),    uy    = 0.002; end
if nargin < 4 || isempty(n),     n     = 24;    end
if nargin < 5 || isempty(j0),    j0    = 2;     end   % 1-BASED
if nargin < 6 || isempty(zeta1), zeta1 = 0.010; end

L    = lattice(n, j0, 0.0, zeta1);
Kmod = diag(L.wn.^2) + kh*(L.psi*L.psi.');
Kmod = (Kmod + Kmod.')/2;
[V, D]   = eig(Kmod);
[lam, ix] = sort(diag(D), 'ascend');
V  = V(:, ix);
sg = sign(V(1,:)); sg(sg == 0) = 1;
V  = V .* sg;

S       = struct();
S.n     = n;
S.wn    = sqrt(lam);
S.psit  = V.' * L.psi;
S.c     = L.ci(1);            % uniform modal damping
S.kh    = kh;
S.alpha = alpha;
S.uy    = uy;
S.beta  = 0.5/uy;
S.gamma = 0.5/uy;
S.V     = V;
S.chain = L;

Sc    = S;                    % snapshot captured by the factory
S.rhs = @(A, Om) make_bw_rhs(Sc, A, Om);
end

function f = make_bw_rhs(S, A, Om)
f = @(t,y) bw_rhs(t, y, S.n, S.wn.^2, S.c, S.psit, S.kh, S.alpha, ...
                  S.beta, S.gamma, A, Om);
end

function dy = bw_rhs(t, y, n, W2, c, pt, kh, al, be, ga, A, Om)
p  = y(1:n);
v  = y(n+1:2*n);
z  = y(2*n+1);
w  = pt.'*p;
wd = pt.'*v;
resid = (al*kh*w + (1-al)*kh*z) - kh*w;
zd    = wd - be*abs(wd)*z - ga*wd*abs(z);
acc   = -c*v - W2.*p - pt*resid + pt*(A*cos(Om*t));
dy    = [v; acc; zd];
end
