function [tt, aa, vv, ac, y0] = settle_release(L, A, mode, span, dt, rtol, t_settle)
%SETTLE_RELEASE  Force to steady state, release, return the free-decay window.
%
%   Port of  bias_map._settle  +  Lattice.decay  (lattice_nl.py), i.e. the
%   protocol of  bias_map.controlled_release.
%
%   [tt, aa, vv, ac, y0] = SETTLE_RELEASE(L, A, mode, span, dt, rtol, t_settle)
%
%   mode is ONE-BASED (the study's tracked resonance is mode = 7; Python
%   writes MODE = 6 zero-based).  Resonant station forcing at Om = wn(mode)
%   builds mode `mode` up from rest over t_settle, the force is released, and
%   the first `span` of the ensuing free decay is returned on a uniform grid
%   of step dt.  This pins the state by PROTOCOL rather than inheriting it
%   from a violent decay history -- the reason bias_map prepares every cell
%   independently instead of sliding a window down one long record.
%
%   Defaults are the study's: span 400, dt 0.05, rtol 1e-9, t_settle 6000,
%   with AbsTol = rtol*1e-2 exactly as scipy is called (atol = rtol*1e-2).
%
%   tt is (span/dt) x 1 running 0, dt, ..., span-dt  -- the half-open grid of
%   numpy.arange(0, span, dt), NOT 0:dt:span.  It is built by multiplication,
%   not by the colon operator, so the last sample is exact.
%
%   ac is recomputed algebraically from the free RHS at every output sample,
%   which is what Lattice.decay does; it is not a finite difference.

if nargin < 4 || isempty(span),     span     = 400.0;  end
if nargin < 5 || isempty(dt),       dt       = 0.05;   end
if nargin < 6 || isempty(rtol),     rtol     = 1e-9;   end
if nargin < 7 || isempty(t_settle), t_settle = 6000.0; end

solver = odesolver();
opts   = odeset('RelTol', rtol, 'AbsTol', rtol*1e-2);
n      = L.n;

% --- settle under resonant station forcing, from rest --------------------
% A three-point tspan integrates continuously and reports only at those
% points, matching scipy's behaviour while bounding memory.
[~, Ys] = solver(rhs_forced(L, A, L.wn(mode)), [0, t_settle/2, t_settle], ...
                 zeros(2*n,1), opts);
y0 = Ys(end,:).';

% --- release and record the free decay -----------------------------------
tt = (0:round(span/dt)-1).' * dt;
[~, Yd] = solver(rhs_free(L), tt, y0, opts);
aa = Yd(:, 1:n);
vv = Yd(:, n+1:2*n);
ac = -(L.ci.').*vv - ((L.wn.^2).').*aa - (L.k3*(aa*L.psi).^3).*(L.psi.');
end
