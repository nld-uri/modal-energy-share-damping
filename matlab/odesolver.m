function s = odesolver()
%ODESOLVER  Handle to the integrator used throughout the package.
%
%   ode89 (Verner 8/9, base MATLAB since R2021b) is the nearest equivalent to
%   scipy's DOP853 (Dormand-Prince 8(5,3)) used by every Python script here.
%
%   THEY ARE DIFFERENT IMPLEMENTATIONS OF THE SAME ORDER.  Agreement between
%   this package and the Python reference is expected to SOLVER TOLERANCE, not
%   to machine precision.  selftest.m sets the tolerance of every integrated
%   check from a measured sensitivity study (see its header) rather than from
%   a blanket constant.
%
%   Falls back to ode45 on releases without ode89; at RelTol 1e-9 ode45 will
%   be slow and its integrated results may miss the tolerances below, which is
%   a solver limitation and not a port error.

if exist('ode89', 'file') || exist('ode89', 'builtin')
    s = @ode89;
else
    warning('odesolver:noODE89', ...
        ['ode89 not found; falling back to ode45. Integrated self-test ' ...
         'checks may exceed their tolerances -- that is a solver ' ...
         'difference, not a port error.']);
    s = @ode45;
end
end
