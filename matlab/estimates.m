function out = estimates(L, aa, vv, ac, mode, dt)
%ESTIMATES  The three estimates of the same modal damping, per record.
%
%   Port of  bias_map._three_estimates.
%
%   out = ESTIMATES(L, aa, vv, ac, mode, dt)     (mode is ONE-BASED)
%
%   Each record carries three estimates of c_m, so a physical bias can be
%   separated from an artefact of the decomposition.  All three are returned
%   as RATIOS to the true modal damping c_m = L.ci(mode):
%
%     r_gcvd  windowed GCVD state relation -- the method under test
%     r_env   Hilbert log-envelope fit to the modal coordinate -- needs no
%             decomposition, so it is the reference measurement
%     r_flux  the prediction c_m - <P_m>/<qdot_m^2> from the exact modal
%             energy balance, P_m = -psi_m k3 (psi.q)^3 qdot_m
%
%   Also: stiff_pct (identified frequency shift, percent), a_m (rms modal
%   coordinate), w_rms (rms station displacement), e_frac (tracked mode's
%   share of the mean modal energy), cond (of [aa vv]).
%
%   The GCVD mode is picked as the one whose shape has the largest component
%   AT the tracked mode's index, over the 8 tracked candidates -- not by
%   frequency -- exactly as in Python.
%
%   The envelope fit trims 1/20 of the window at EACH end: Python slices
%   tt[k : len-k] with k = len//20, which is MATLAB rows k+1 : m-k.

if nargin < 6 || isempty(dt), dt = 0.05; end

n   = L.n;
psi = L.psi;
wn2 = L.wn.^2;
m   = size(aa,1);

w  = aa*psi;                       % station displacement, m x 1
tt = (0:m-1).' * dt;

% --- GCVD state relation --------------------------------------------------
gc = gcvd_windowed(aa, vv, ac, 1, 8, true, eye(n));
[~, mq] = max(abs(reshape(gc.shape(1,:,mode), [], 1)));
out.r_gcvd = 2*gc.zeta(1,mq)*gc.wn(1,mq)/L.ci(mode);

% --- Hilbert log-envelope reference --------------------------------------
env = abs(hilbert_fft(aa(:,mode)));
k   = floor(m/20);
sl  = (k+1):(m-k);
p   = polyfit(tt(sl), log(env(sl)), 1);
out.r_env = -2*p(1)/L.ci(mode);

% --- energy-flux prediction ----------------------------------------------
P = -(L.k3*w.^3).*vv.*(psi.');
out.r_flux = (L.ci(mode) - mean(P(:,mode))/mean(vv(:,mode).^2))/L.ci(mode);

% --- diagnostics ----------------------------------------------------------
E  = mean(0.5*(vv.^2 + (wn2.').*aa.^2), 1);
sv = svd([aa vv]);
out.stiff_pct = 100*(gc.wn(1,mq)/L.wn(mode) - 1);
out.a_m       = sqrt(mean(aa(:,mode).^2));
out.w_rms     = sqrt(mean(w.^2));
out.e_frac    = E(mode)/sum(E);
out.cond      = sv(1)/sv(end);
end
