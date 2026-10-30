function [u, mpc] = realify(v)
%REALIFY  Phase-rotate a complex mode to maximum collinearity.
%
%   Port of  realify()  in  gcvd.py.
%
%   [u, mpc] = REALIFY(v) returns the unit-norm real mode shape u and the
%   modal phase collinearity mpc, which is 1 for a purely normal (real) mode.
%
%   u carries a deterministic sign (largest-magnitude component positive), so
%   it is directly comparable between Python and MATLAB even though the
%   eigenvector that produced v is not.
%
%   max() in MATLAB and argmax() in numpy both return the FIRST maximiser, so
%   the normalisation index agrees.

v = v(:);
[~, i1] = max(abs(v));
v = v / v(i1);

R = [real(v), imag(v)];
R = R - mean(R, 1);
s = svd(R);
if numel(s) < 2, s(2) = 0; end
mpc = (s(1)^2 - s(2)^2)^2 / (s(1)^2 + s(2)^2)^2;

th = angle(sum((v/norm(v)).^2))/2;
u  = real(v*exp(-1i*th));
u  = u/norm(u);
[~, i2] = max(abs(u));
u = u*sign(u(i2));
end
