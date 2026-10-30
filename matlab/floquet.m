function out = floquet(mu, A, mode, n, j0, zeta1, t_probe, npts)
%FLOQUET  Stability of the settled periodic orbit at similarity parameter mu.
%
%   Port of  floquet_true_orbit()  in  onset_mechanism.py.
%
%   out = FLOQUET(mu)                               study defaults
%   out = FLOQUET(mu, A, mode, n, j0, zeta1, t_probe, npts)
%
%   mu = k3 A^2 is the similarity parameter; with A fixed at 2e-2 the lattice
%   is built with k3 = mu/A^2.  mode and j0 are ONE-BASED (7 and 2 here;
%   Python writes 6 and 1 zero-based).
%
%   The settled orbit is probed over one period starting at t_probe = 2000,
%   BEFORE the sidebands saturate, and the linearisation about that carrier is
%   integrated for one period.  The onset of the damping bias is the NNM's
%   stability boundary, so it is predictable from this computation rather than
%   being an empirical threshold: damping identifiability and nonlinear-normal-
%   mode stability are the same question.
%
%   Reference values (Python, and in reference.mat):
%       mu = 3.0e-4  ->  max_multiplier 0.999480  (stable)
%       mu = 5.0e-4  ->  max_multiplier 1.011861  (unstable)
%
%   The orbit probe runs at RelTol 1e-11 / AbsTol 1e-14, the Python settings.

if nargin < 2 || isempty(A),       A       = 2e-2;   end
if nargin < 3 || isempty(mode),    mode    = 7;      end   % 1-BASED
if nargin < 4 || isempty(n),       n       = 24;     end
if nargin < 5 || isempty(j0),      j0      = 2;      end   % 1-BASED
if nargin < 6 || isempty(zeta1),   zeta1   = 0.010;  end
if nargin < 7 || isempty(t_probe), t_probe = 2000.0; end
if nargin < 8 || isempty(npts),    npts    = 801;    end

L  = lattice(n, j0, mu/A^2, zeta1);
Om = L.wn(mode);
T  = 2*pi/Om;

tev    = linspace(t_probe, t_probe+T, npts);
solver = odesolver();
opts   = odeset('RelTol', 1e-11, 'AbsTol', 1e-14);

% One continuous integration from rest, reporting only over the probe window.
% Prepending 0 to tspan is what makes it a single run rather than a restart;
% the extra first row is dropped.
[~, Y] = solver(rhs_forced(L, A, Om), [0, tev], zeros(2*L.n,1), opts);
wq  = Y(2:end, 1:L.n)*L.psi;
tau = linspace(0, T, npts).';

out.mu                = mu;
out.k3                = L.k3;
out.w_peak            = max(abs(wq));
out.k3_wpeak2         = L.k3*out.w_peak^2;
out.max_multiplier    = monodromy(L, wq, tau, T);
out.unstable          = out.max_multiplier > 1.0;
out.growth_rate_energy= 2*log(out.max_multiplier)/T;
out.crest_factor      = out.w_peak/sqrt(mean(wq.^2));
end
