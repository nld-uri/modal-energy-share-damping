function f = rhs_forced(L, A, Om, ph)
%RHS_FORCED  Free equations plus station forcing at the nonlinear station.
%
%   Port of  Lattice.rhs_forced  in  lattice_nl.py, generalised to the
%   multi-harmonic band forcing of  bandwidth_axis.band_force.
%
%   f = RHS_FORCED(L, A, Om)      single tone:  u(t) = A cos(Om t)
%   f = RHS_FORCED(L, A, Om, ph)  band:         u(t) = sum_k A cos(Om_k t + ph_k)
%
%   With scalar A and vector Om/ph this reproduces bandwidth_axis.one_cell
%   exactly (there A is a scalar amplitude common to all harmonics).
%
%       a_dot_dot = -c.*a_dot - wn.^2.*a - psi*(k3*(psi.'*a)^3) + psi*u(t)

if nargin < 4 || isempty(ph), ph = zeros(size(Om)); end

n   = L.n;
w2  = L.wn.^2;
c   = L.ci;
psi = L.psi;
k3  = L.k3;
Av  = A(:).';
Omv = Om(:).';
phv = ph(:).';

f = @(t,z) [ z(n+1:2*n) ; ...
            -c.*z(n+1:2*n) - w2.*z(1:n) - psi*(k3*(psi.'*z(1:n))^3) ...
            + psi*sum(Av.*cos(Omv*t + phv)) ];
end
