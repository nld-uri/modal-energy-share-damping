function Rout = run_all(level)
%RUN_ALL  Reproduce the Paper-3 tables in MATLAB and print them.
%
%   run_all            % 'quick' -- every table, reduced sweeps  (~5-15 min)
%   run_all full       % the complete Python sweeps             (~1-3 h)
%   Rout = run_all(...)% also returns the tables as a struct
%
%   Source scripts reproduced here:
%     Table 1  chain spectrum and station influence      lattice_nl.py
%     Table 2  three estimates over a mu sweep           bias_map.py
%     Table 3  the derived damping law                   share_law.py
%     Table 4  which modes must be observed              recovery_recipe.py
%     Table 5  Floquet stability of the tracked NNM      onset_mechanism.py
%     Table 6  dissipative boundary case                 friction_boundary.py
%     Table 7  noise and partial instrumentation         noise_sensing.py
%
%   TABLE 7 IS NOT BIT-COMPARABLE WITH PYTHON.  It draws pseudorandom noise,
%   and numpy's PCG64 stream cannot be reproduced by base MATLAB (which offers
%   Mersenne Twister, Philox and Threefry).  Only the ensemble statistics are
%   comparable; the noise-free row and the projection conditioning are exact
%   and are the parts selftest.m checks.
%
%   Runtimes are dominated by the t_settle = 6000 forced integrations at
%   RelTol 1e-9.  Progress is printed per cell.

if nargin < 1 || isempty(level), level = 'quick'; end
level = lower(char(level));
isfull = strcmp(level, 'full');

here = fileparts(mfilename('fullpath')); addpath(here);
N = 24; J0 = 2; MODE = 7; ZET = 0.010;
SPAN = 400.0; DT = 0.05; RTOL = 1e-9; TSET = 6000.0; EDGE = 20;
A_FIX = 2e-2;
Rout = struct();

fprintf('\n################ Paper 3, MATLAB port -- level: %s ################\n', level);
fprintf('N=%d, station j0=%d, tracked mode=%d (all ONE-BASED), zeta1=%.3f\n', N, J0, MODE, ZET);
tAll = tic;

%% ------------------------------------------------------------- Table 1 ----
L0 = lattice(N, J0, 0.0, ZET);
fprintf('\n=== Table 1: chain spectrum and station influence ===\n');
fprintf('  mode   omega_i     psi_i      omega_i/omega_7\n');
for i = 1:N
    fprintf('  %4d  %9.6f  %9.6f  %9.6f\n', i, L0.wn(i), L0.psi(i), L0.wn(i)/L0.wn(MODE));
end
fprintf('  uniform modal damping c_i = %.10g for all i\n', L0.ci(1));
fprintf('  3*omega_7 = %.6f, band edge = %.6f -> third harmonic in band: %d\n', ...
        3*L0.wn(MODE), L0.wn(end), 3*L0.wn(MODE) <= L0.wn(end));
Rout.wn = L0.wn; Rout.psi = L0.psi;

%% ------------------------------------------------------------- Table 2 ----
if isfull, K3S = [0 0.5 1 2 4]; AMPS = [1e-3 3e-3 5e-3 1e-2 2e-2];
else,    K3S = [1 4];          AMPS = [5e-3 2e-2];
end
fprintf('\n=== Table 2: three estimates per record (bias_map.py) ===\n');
fprintf('     k3       A        mu      r_gcvd   r_env    r_flux   e_frac  stiff%%   w_rms\n');
T2 = [];
for k3 = K3S
    for A = AMPS
        L = lattice(N, J0, k3, ZET);
        [~, aa, vv, ac] = settle_release(L, A, MODE, SPAN, DT, RTOL, TSET);
        e = estimates(L, aa, vv, ac, MODE, DT);
        fprintf('  %6.2f  %7.1e  %7.1e  %7.4f  %7.4f  %7.4f  %6.4f  %6.3f  %7.4f\n', ...
                k3, A, k3*A^2, e.r_gcvd, e.r_env, e.r_flux, e.e_frac, e.stiff_pct, e.w_rms);
        T2(end+1,:) = [k3 A k3*A^2 e.r_gcvd e.r_env e.r_flux e.e_frac e.stiff_pct e.w_rms]; %#ok<AGROW>
    end
end
Rout.T2 = T2;

%% ------------------------------------------------------------- Table 3 ----
if isfull, MUS3 = [4e-4 5e-4 6e-4 7e-4 8e-4 1e-3 1.3e-3 1.6e-3];
else,    MUS3 = [5e-4 1e-3 1.6e-3];
end
fprintf('\n=== Table 3: the derived damping law, c_ident = cbar - d(ln f)/dt ===\n');
fprintf('   With UNIFORM modal damping cbar == c_true, so the entire bias is the\n');
fprintf('   logarithmic drift of the tracked mode''s energy share.\n');
fprintf('     mu      share    dlnf/dt    cbar/c    r_env   r_derived  r_proxy   |err|\n');
T3 = [];
for mu = MUS3
    L = lattice(N, J0, mu/A_FIX^2, ZET);
    [tt, aa, vv, ~] = settle_release(L, A_FIX, MODE, SPAN, DT, RTOL, TSET);
    r = energy_rates(L, tt, aa, vv, MODE, 1:N, EDGE);
    env = abs(hilbert_fft(aa(:,MODE)));
    k = floor(numel(tt)/EDGE); sl = (k+1):(numel(tt)-k);
    p = polyfit(tt(sl), log(env(sl)), 1);
    r_env = -2*p(1)/L.ci(MODE);
    fprintf('  %7.1e  %7.4f  %9.2e  %7.4f  %7.4f  %8.4f  %7.4f  %7.4f\n', ...
            mu, r.share_mean, r.dlnf_tot, r.cbar_mean_over_c, r_env, ...
            r.r_derived, r.share_mean, abs(r_env - r.r_derived));
    T3(end+1,:) = [mu r.share_mean r.dlnf_tot r.cbar_mean_over_c r_env r.r_derived]; %#ok<AGROW>
end
fprintf('   derived law : mean |err| %.4f, worst %.4f\n', ...
        mean(abs(T3(:,5)-T3(:,6))), max(abs(T3(:,5)-T3(:,6))));
fprintf('   share proxy : mean |err| %.4f, worst %.4f\n', ...
        mean(abs(T3(:,5)-T3(:,2))), max(abs(T3(:,5)-T3(:,2))));
Rout.T3 = T3;

%% ------------------------------------------------------------- Table 4 ----
if isfull, MUS4 = [6e-4 8e-4 1e-3 1.3e-3 1.6e-3]; else, MUS4 = 1e-3; end
lo = [MODE, MODE-1, MODE-2, MODE-4, 1];
hi = [MODE, MODE+1, MODE+2, MODE+4, N];
nms = {'carrier only','carrier +/- 1','carrier +/- 2','carrier +/- 4','all modes'};
fprintf('\n=== Table 4: what must be observed (recovery_recipe.py) ===\n');
fprintf('   gamma_S / c_true, and the VACUOUS correction gamma_S - dln f_S/dt,\n');
fprintf('   which is identically the single-mode estimate for every subset S.\n');
G4 = zeros(numel(MUS4), 5); V4 = G4;
for q = 1:numel(MUS4)
    L = lattice(N, J0, MUS4(q)/A_FIX^2, ZET);
    [tt, aa, vv, ~] = settle_release(L, A_FIX, MODE, SPAN, DT, RTOL, TSET);
    fprintf('   mu = %.1e\n', MUS4(q));
    for s = 1:5
        r = energy_rates(L, tt, aa, vv, MODE, lo(s):hi(s), EDGE);
        G4(q,s) = r.gamma_over_c; V4(q,s) = r.vacuous_over_c;
        fprintf('     %-16s gamma/c = %8.4f   |err| = %7.4f   vacuous = %10.6f\n', ...
                nms{s}, r.gamma_over_c, abs(r.gamma_over_c-1), r.vacuous_over_c);
    end
    fprintf('     vacuity check, spread across subsets: %.2e\n', ...
            max(V4(q,:))-min(V4(q,:)));
end
fprintf('   summary over %d mu:\n', numel(MUS4));
for s = 1:5
    fprintf('     %-16s mean |err| %.4f, worst %.4f\n', ...
            nms{s}, mean(abs(G4(:,s)-1)), max(abs(G4(:,s)-1)));
end
Rout.T4 = G4;

%% ------------------------------------------------------------- Table 5 ----
if isfull, MUS5 = [3.0e-4 3.2e-4 3.4e-4 3.6e-4 3.8e-4 4.0e-4 5.0e-4 6.0e-4 8.0e-4];
else,    MUS5 = [3.0e-4 5.0e-4];
end
fprintf('\n=== Table 5: Floquet stability of the tracked NNM (onset_mechanism.py) ===\n');
fprintf('   The damping-bias onset IS the NNM stability boundary, so it is\n');
fprintf('   predictable before any identification is attempted.\n');
fprintf('       mu       k3     w_peak   k3*wpk^2  max|mult|  unstable  crest\n');
T5 = [];
for mu = MUS5
    f = floquet(mu, A_FIX, MODE, N, J0, ZET);
    fprintf('  %7.1e  %6.3f  %8.6f  %8.5f  %9.6f  %6d    %6.4f\n', ...
            mu, f.k3, f.w_peak, f.k3_wpeak2, f.max_multiplier, f.unstable, f.crest_factor);
    T5(end+1,:) = [mu f.k3 f.w_peak f.k3_wpeak2 f.max_multiplier f.unstable]; %#ok<AGROW>
end
Rout.T5 = T5;

%% ------------------------------------------------------------- Table 6 ----
if isfull, AMPS6 = [1e-6 1e-5 3e-5 1e-4 1e-3 5e-3 2e-2 6e-2];
else,    AMPS6 = [1e-4 2e-2];
end
fprintf('\n=== Table 6: dissipative boundary case, Bouc-Wen (friction_boundary.py) ===\n');
fprintf('   For the CONSERVATIVE cubic gamma_tot/c is 0.999 at every amplitude.\n');
fprintf('   Here it is amplitude dependent and the single-mode estimate TRACKS it:\n');
fprintf('   the identified damping is not biased, it measures a different quantity.\n');
fprintf('       A      slip    E_m frac  gamma_tot/c  gamma_sub/c  r_env/c\n');
T6 = [];
for A = AMPS6
    b = boucwen_record(A, MODE, 4000.0, SPAN, DT, RTOL, EDGE);
    fprintf('  %7.0e  %6.3f  %8.4f  %10.4f  %11.4f  %8.4f\n', ...
            A, b.slip_ratio, b.e_frac, b.gamma_tot_over_c, b.gamma_sub_over_c, b.r_env_over_c);
    T6(end+1,:) = [A b.slip_ratio b.e_frac b.gamma_tot_over_c b.gamma_sub_over_c b.r_env_over_c]; %#ok<AGROW>
end
Rout.T6 = T6;

%% ------------------------------------------------------------- Table 7 ----
fprintf('\n=== Table 7: noise and partial instrumentation (noise_sensing.py) ===\n');
fprintf('   NOT bit-comparable with Python: numpy PCG64 cannot be reproduced in\n');
fprintf('   base MATLAB, so the noise realisations differ.  Compare the ENSEMBLE\n');
fprintf('   statistics, not individual numbers.  The SNR = Inf row is exact.\n');
MU7 = 1e-3; NREP = 5;
L7 = lattice(N, J0, MU7/A_FIX^2, ZET);
[tt7, q7, qd7, ~] = settle_release(L7, A_FIX, MODE, SPAN, DT, RTOL, TSET);
x7 = q7*L7.Phi.';  xd7 = qd7*L7.Phi.';        % physical displacement, velocity
sub5 = (MODE-2):(MODE+2);
sx0 = sqrt(mean(x7(:).^2)); sv0 = sqrt(mean(xd7(:).^2));
fprintf('   SNR sweep, full sensing:\n');
fprintf('      SNR dB    mean gamma_sub/c    std      mean |err|\n');
T7a = [];
for snr = [Inf 60 50 40 30 20]
    nr = 1; if isfinite(snr), nr = NREP; end
    g = zeros(nr,1);
    for rep = 1:nr
        rng(20260928 + 1000*rep + round(min(snr,999)), 'twister');
        if isfinite(snr)
            xn = x7 + sx0*10^(-snr/20)*randn(size(x7));
            vn = xd7 + sv0*10^(-snr/20)*randn(size(xd7));
        else
            xn = x7; vn = xd7;
        end
        r = energy_rates(L7, tt7, xn*L7.Phi, vn*L7.Phi, MODE, sub5, EDGE);
        g(rep) = r.gamma_over_c;
    end
    fprintf('      %6.0f   %14.4f  %9.4f  %9.4f\n', snr, mean(g), std(g,1), mean(abs(g-1)));
    T7a(end+1,:) = [snr mean(g) std(g,1) mean(abs(g-1))]; %#ok<AGROW>
end
fprintf('   sensor sweep at 60 dB (spatial aliasing is the failure mode):\n');
fprintf('      sensors   cond(Phi_s)   mean gamma_sub/c   mean |err|\n');
T7b = [];
for ms = [24 16 12 8 6 5]
    idx = unique(round(linspace(1, N, ms)));
    Ps  = L7.Phi(idx, sub5);
    cnd = cond(Ps);  P = pinv(Ps);
    g = zeros(NREP,1);
    for rep = 1:NREP
        rng(20260928 + 7919*rep + ms, 'twister');
        xn = x7(:,idx) + sx0*10^(-60/20)*randn(numel(tt7), numel(idx));
        vn = xd7(:,idx) + sv0*10^(-60/20)*randn(numel(tt7), numel(idx));
        qh = zeros(numel(tt7), N); qdh = qh;
        qh(:,sub5)  = xn*P.';
        qdh(:,sub5) = vn*P.';
        r = energy_rates(L7, tt7, qh, qdh, MODE, sub5, EDGE);
        g(rep) = r.gamma_over_c;
    end
    fprintf('      %5d   %11.4f   %16.4f   %10.4f\n', numel(idx), cnd, mean(g), mean(abs(g-1)));
    T7b(end+1,:) = [numel(idx) cnd mean(g) mean(abs(g-1))]; %#ok<AGROW>
end
Rout.T7a = T7a; Rout.T7b = T7b;

fprintf('\n################ done in %.0f s ################\n\n', toc(tAll));
end
