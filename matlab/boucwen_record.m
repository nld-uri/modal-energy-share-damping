function out = boucwen_record(A, mode, t_settle, span, dt, rtol, edge)
%BOUCWEN_RECORD  Settle, release and analyse one hysteretic record.
%
%   Port of  one_amplitude()  in  friction_boundary.py.
%
%   out = BOUCWEN_RECORD(A, mode, t_settle, span, dt, rtol, edge)
%   mode is ONE-BASED (7).  Defaults are the study's: t_settle 4000,
%   span 400, dt 0.05, rtol 1e-9, edge 20.
%
%   Returns slip_ratio (max|z|/uy -- 1 means fully slipping), w_rms, e_frac,
%   and gamma_tot_over_c / gamma_sub_over_c / r_env_over_c.
%
%   The point of the comparison: for the conservative cubic gamma_tot/c is
%   0.999 at every amplitude while the single-mode estimate is off by up to
%   0.54, whereas here gamma_tot/c rises to about 6, is strongly amplitude
%   dependent, and the single-mode estimate TRACKS it -- the dominant term is
%   a real dissipation common to all modes, not a redistribution between them.

if nargin < 1 || isempty(A),        A        = 2e-2;   end
if nargin < 2 || isempty(mode),     mode     = 7;      end
if nargin < 3 || isempty(t_settle), t_settle = 4000.0; end
if nargin < 4 || isempty(span),     span     = 400.0;  end
if nargin < 5 || isempty(dt),       dt       = 0.05;   end
if nargin < 6 || isempty(rtol),     rtol     = 1e-9;   end
if nargin < 7 || isempty(edge),     edge     = 20;     end

S      = boucwen();
solver = odesolver();
opts   = odeset('RelTol', rtol, 'AbsTol', rtol*1e-2);
n      = S.n;

[~, Ys] = solver(S.rhs(A, S.wn(mode)), [0, t_settle/2, t_settle], ...
                 zeros(2*n+1,1), opts);
y0 = Ys(end,:).';

tt = (0:round(span/dt)-1).' * dt;
[~, Yd] = solver(S.rhs(0, 0), tt, y0, opts);
pp = Yd(:, 1:n);
vv = Yd(:, n+1:2*n);
z  = Yd(:, 2*n+1);

Ei   = 0.5*(vv.^2 + ((S.wn.^2).').*pp.^2);
Etot = sum(Ei, 2);
m    = numel(tt);
k    = floor(m/edge);
sl   = (k+1):(m-k);
sub  = max(1,mode-2):min(n,mode+2);
env  = abs(hilbert_fft(pp(:,mode)));
w    = pp*S.psit;

p1 = polyfit(tt(sl), log(Etot(sl)), 1);
p2 = polyfit(tt(sl), log(sum(Ei(:,sub),2)), 1);
p3 = polyfit(tt(sl), log(env(sl)), 1);

out.A                = A;
out.slip_ratio       = max(abs(z))/S.uy;
out.w_rms            = sqrt(mean(w.^2));
out.e_frac           = mean(Ei(:,mode))/mean(Etot);
out.gamma_tot_over_c = -p1(1)/S.c;
out.gamma_sub_over_c = -p2(1)/S.c;
out.r_env_over_c     = -2*p3(1)/S.c;
out.y0               = y0;
end
