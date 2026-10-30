function mx = monodromy(L, wq, tau, T, rtol, atol)
%MONODROMY  Max Floquet multiplier of the linearisation about a carrier.
%
%   Port of  _monodromy()  in  onset_mechanism.py.
%
%   mx = MONODROMY(L, wq, tau, T, rtol, atol)
%       wq   station displacement psi.'*q sampled over one carrier period
%       tau  the matching time grid on [0, T]
%       T    carrier period
%
%   The cubic element modulates the tangent stiffness by
%   3 k3 (psi.q)^2 psi psi.', which is periodic at the carrier period, so the
%   sidebands see a parametric excitation.  Integrating the 2n x 2n variational
%   system over one period gives the monodromy matrix; mx > 1 means the
%   tracked NNM is unstable.  No identification enters this computation.
%
%   VECTORISATION ORDER.  Python ravels the matrix row-major, MATLAB column-
%   major.  Both solve the SAME matrix ODE dM/dt = G(M), M(0) = I -- vec and
%   unvec are each other's inverse within each language -- so M(T) is the same
%   matrix and its eigenvalues are identical.  The identity initial condition
%   is symmetric, so even the packed initial vector agrees.
%
%   Defaults rtol 1e-10, atol 1e-12 are the Python values.

if nargin < 5 || isempty(rtol), rtol = 1e-10;    end
if nargin < 6 || isempty(atol), atol = rtol*1e-2; end

n   = L.n;
W2  = L.wn.^2;
C   = L.ci;
PSI = L.psi;
k3  = L.k3;

% griddedInterpolant reproduces numpy.interp's piecewise-linear rule.  mod(t,T)
% never leaves [tau(1), tau(end)] = [0, T], so the extrapolation mode is never
% reached and numpy's end-clamping is irrelevant here.
F = griddedInterpolant(tau(:), wq(:), 'linear', 'linear');

solver = odesolver();
opts   = odeset('RelTol', rtol, 'AbsTol', atol);
y0     = reshape(eye(2*n), [], 1);

[~, Y] = solver(@(t,y) mono_rhs(t, y, n, W2, C, PSI, k3, F, T), ...
                [0, T/2, T], y0, opts);

Mm = reshape(Y(end,:).', 2*n, 2*n);
mx = max(abs(eig(Mm)));
end

function dy = mono_rhs(t, y, n, W2, C, PSI, k3, F, T)
M   = reshape(y, 2*n, 2*n);
dK  = 3*k3*F(mod(t,T))^2;
top = M(1:n, :);
bot = M(n+1:end, :);
dy  = reshape([ bot ; ...
               -(W2.*top) - dK*(PSI*(PSI.'*top)) - C.*bot ], [], 1);
end
