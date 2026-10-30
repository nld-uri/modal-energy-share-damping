function [lam, Phi, U] = gcvd(disp_, vel, acc, center)
%GCVD  Generalised characteristic value decomposition of one response window.
%
%   Port of  gcvd()  in  gcvd.py  (Stein & Chelidze, Nonlinear Dyn 113(1):25-45,
%   2025, Sec. 4).
%
%   [lam, V, Xi] = GCVD(disp, vel, acc, center)
%
%   disp, vel, acc are (m x n): n spatial stations over m snapshots.
%   center (default true) removes each column's window mean, so the window is
%   linearised about the LOCAL equilibrium rather than the origin.
%
%       X1 = [disp vel],  X2 = [vel acc],  Kij = Xi^H Xj
%       Q   = 1/4 (K21 K11^-1 + K22 K12^-1 + K11 K21^-1 + K12 K22^-1) = V Lam V^-1
%       Xi  = X1 V^-H  (Sigma1 = I),   Sigma2 = pinv(Xi) X2 V^-H = T Lam T^-1
%
%   lam are the EIGENVALUES of Sigma2, not its diagonal: Sigma2 is diagonal
%   only for an exactly linear, noise-free record, and its off-diagonal
%   leakage is of the order of Re(lam), which is the damping.
%
%   SHAPES are the canonical regression shapes Phi = V T^-H, paired with the
%   temporal basis U = Xi T, where Sigma2 = T Lam T^-1. This is the only choice
%   for which both X1 = U Phi^H and X2 = U Lam Phi^H hold, i.e. the only one
%   under which this really is the diagonal decomposition written above. Under
%   it lam(i) already belongs to Phi(:,i), so the one-to-one assignment that
%   the unrotated-V convention required (matchpairs on |T|.^2) is not performed
%   at all. Canonical since 2026-10-22; the superseded convention returned the
%   eigenvectors of Q unrotated.
%
%   lam are the poles at the response amplitude of the window:
%   lam = -zeta*wn +/- i*wn*sqrt(1-zeta^2).
%
%   ORDER AND SCALING.  numpy.linalg.eig and MATLAB eig may return the
%   eigenvectors of Q in a different order and with a different complex
%   scaling.  Neither matters: lam is invariant under any column scaling
%   V -> V*D (the D cancels between pinv(Xi) and X2 V^-H), and the consumers
%   below sort by |lam| and phase-normalise every shape through realify().
%   Only the ORDER of the returned lam/V can differ from Python, so
%   selftest.m compares sorted quantities.
%
%   A/B in MATLAB is mrdivide == numpy's _rdiv(A,B) = A @ inv(B); both solve
%   rather than forming the inverse.

if nargin < 4 || isempty(center), center = true; end

X1 = [disp_, vel];
X2 = [vel,  acc];
if center
    X1 = X1 - mean(X1, 1);
    X2 = X2 - mean(X2, 1);
end

K11 = X1.'*X1;  K12 = X1.'*X2;
K21 = X2.'*X1;  K22 = X2.'*X2;

Q = 0.25*( K21/K11 + K22/K12 + K11/K21 + K12/K22 );

[V, ~] = eig(Q);
VmH = inv(V)';                 % V^-H, matching np.linalg.inv(V).conj().T
Xi  = X1*VmH;                  % Sigma1 = I
S2  = pinv(Xi)*(X2*VmH);
[T, D] = eig(S2);
lam = diag(D);
Phi = V/(T');                  % Phi = V T^-H   (mrdivide: V*inv(T'))
U   = Xi*T;                    % U   = Xi T
end
