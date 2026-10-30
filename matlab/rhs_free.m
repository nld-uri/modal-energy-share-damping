function f = rhs_free(L)
%RHS_FREE  Free modal equations of motion with the grounded cubic.
%
%   Port of  Lattice.rhs_free  in  lattice_nl.py.
%
%   f = RHS_FREE(L) returns @(t,z) for the 2n state z = [a; a_dot]:
%
%       a_dot_dot = -c.*a_dot - wn.^2.*a - psi*(k3*(psi.'*a)^3)
%
%   t is unused (the free system is autonomous) but kept in the signature so
%   the handle can be passed straight to ode89.

n   = L.n;
w2  = L.wn.^2;
c   = L.ci;
psi = L.psi;
k3  = L.k3;

f = @(t,z) [ z(n+1:2*n) ; ...
            -c.*z(n+1:2*n) - w2.*z(1:n) - psi*(k3*(psi.'*z(1:n))^3) ];
end
