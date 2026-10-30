function L = lattice(n, j0, k3, zeta1)
%LATTICE  Fixed-fixed mass-spring chain with one grounded cubic spring.
%
%   Port of  chain_modes()  and  Lattice.__init__  in  lattice_nl.py.
%
%   L = LATTICE()                 defaults of the Paper-3 study
%   L = LATTICE(n, j0, k3, zeta1)
%
%   Modal coordinates throughout: M = I, K = diag(wn.^2), and the nonlinear
%   station enters as the rank-one direction psi = Phi.' * e_j0.
%
%       a_ddot_i + c_i a_dot_i + wn_i^2 a_i + psi_i k3 (psi.a)^3 = psi_i A cos(Om t)
%
%   INDEX BASE -- READ THIS.  The Python study uses J0 = 1 and MODE = 6, both
%   ZERO-BASED, i.e. the forcing/nonlinear station is the SECOND mass and the
%   tracked resonance is the SEVENTH mode.  Here j0 and mode are ONE-BASED, so
%   the same configuration is  j0 = 2  and  mode = 7.  Those are the defaults.
%
%   EIGENVECTOR SIGNS.  numpy.linalg.eigh and MATLAB eig each return an
%   arbitrary sign per eigenvector, and they disagree on 12 of the 24 chain
%   modes.  Both are canonicalised here to FIRST ROW POSITIVE, which for this
%   chain reproduces the analytic mode shapes
%       Phi(j,i) = sqrt(2/(n+1)) sin(j i pi/(n+1))
%   to 5e-15.  The choice is physics-neutral: flipping column i flips a_i and
%   psi_i together, leaving psi.a, every modal energy and every reported
%   quantity unchanged (verified bit-exact in Python -- 0.0 difference with 12
%   signs flipped).  The smallest first-row entry is 0.0354, so the sign is
%   never in numerical doubt.
%
%   Fields: n, j0 (1-based), k3, zeta1, wn (n x 1, ascending), Phi (n x n),
%           psi (n x 1), ci (n x 1, uniform = 2 zeta1 wn(1)).

if nargin < 1 || isempty(n),     n     = 24;    end
if nargin < 2 || isempty(j0),    j0    = 2;     end   % 1-BASED (Python j0 = 1)
if nargin < 3 || isempty(k3),    k3    = 0.0;   end
if nargin < 4 || isempty(zeta1), zeta1 = 0.010; end

K = diag(2*ones(n,1)) - diag(ones(n-1,1),1) - diag(ones(n-1,1),-1);
K = (K + K.')/2;                       % exact symmetry -> symmetric eig path
[V, D] = eig(K);
[w2, ix] = sort(diag(D), 'ascend');    % numpy eigh guarantees ascending; MATLAB
V = V(:, ix);                          % eig does not document it, so sort here
sg = sign(V(1,:)); sg(sg == 0) = 1;    % canonical: first row positive
V = V .* sg;

e = zeros(n,1); e(j0) = 1;

L = struct();
L.n     = n;
L.j0    = j0;
L.k3    = k3;
L.zeta1 = zeta1;
L.wn    = sqrt(w2);
L.Phi   = V;
L.psi   = V.' * e;                     % == V(j0,:).'
L.ci    = repmat(2*zeta1*L.wn(1), n, 1);   % UNIFORM modal damping
end
