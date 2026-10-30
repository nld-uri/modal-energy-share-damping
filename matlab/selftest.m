function selftest(level)
%SELFTEST  Check this MATLAB port against reference vectors exported from the
%          verbatim Python sources.  Run it from the package directory:
%
%       >> selftest            % everything  (integrates; several minutes)
%       >> selftest quick      % algebraic checks only (seconds)
%
%   THE AUTHOR OF THIS PACKAGE NEVER EXECUTED ANY OF IT.  MATLAB cannot be
%   run in the environment where it was written.  This script is how you
%   establish that it works.  Every line below prints the computed value, the
%   Python reference, the absolute and relative difference and the tolerance
%   used, so a FAIL tells you WHICH quantity is wrong and BY HOW MUCH.
%
% ---------------------------------------------------------------------------
% TOLERANCES.  There is no blanket constant.  Each tier is set from a measured
% sensitivity, and each is listed on its own check line.
%
%   T_ALG  = 1e-12 (rel)  Pure linear algebra reproduced from identical
%       formulas: chain eigenproblem, psi, the FFT analytic signal, realify,
%       RHS evaluations.  Both languages call the same LAPACK and FFT kernels;
%       only summation order differs, which is ~1e-15 relative.  1000x margin.
%
%   T_GCVD = 1e-10 (rel)  GCVD on the synthetic exact record.  Four explicit
%       matrix inverses appear in Q; a 1e-16 input perturbation grows by the
%       conditioning of K11..K22.  100x margin on the observed spread.
%
%   T_REC  = 1e-8  (rel)  estimates/energy_rates evaluated on the SUPPLIED
%       sub-record (no integration).  Adds two least-squares polyfits, an SVD
%       of an 8000x48 matrix and a GCVD on a record of condition 3.5e4.
%
%   T_INT  = 1e-5  (abs)  Anything that requires integrating the settle and
%       release.  MEASURED in Python: moving DOP853's own rtol from 1e-9 to
%       1e-12 shifts these estimates by at most 2.3e-9 and the settled state
%       by 9.3e-9 relative -- the settled orbit is attracting, so it does not
%       amplify integration error.  That measurement bounds this integrator's
%       sensitivity to its OWN error level; it does not bound the difference
%       between two different 8th-order pairs, which can be larger.  Set at
%       ~4000x the measured spread on that reasoning -- still three to four
%       decades below any real porting bug, which changes these O(1) ratios
%       by 1e-2 or more.
%
%   T_STATE= 1e-5  (rel)  The settled state vector itself, from the same
%       9.3e-9 measurement on the same reasoning.
%
%   T_FLOQ = 2e-5  (abs)  Floquet max multiplier.  TWO MEASUREMENTS, both run
%       in Python.  (i) Tightening the nominal tolerances to rtol 1e-12 moves
%       the multiplier by 9.4e-9.  (ii) Vectorising the monodromy column-major
%       instead of row-major -- the same matrix ODE, the same solver, only the
%       2304 components in a different order, so the step-size controller sums
%       its error norm differently -- moves it by 5.7e-9 at mu = 3e-4 and
%       1.6e-9 at mu = 5e-4.  The second is the closest available proxy for
%       "the same method assembled differently", and it is only a proxy: a
%       genuinely different Runge-Kutta pair can differ by more.  Set ~2000x
%       above the larger.  The stable/unstable verdict turns on a gap of
%       0.0124 (0.999480 vs 1.011861), so this is three decades below anything
%       able to flip it -- and the verdict itself is checked separately, at
%       zero tolerance.
%
%   T_BW   = 1e-3  (rel)  Integrated Bouc-Wen rates.  The abs() in the
%       hysteresis law makes the RHS non-differentiable at every velocity and
%       force reversal; a Runge-Kutta pair loses its formal order crossing
%       each kink and the loss is INTEGRATOR-SPECIFIC, so it is not bounded by
%       the 1.1e-7 DOP853-internal spread that was measured.  Set four decades
%       looser than that.  A wrong sign, a wrong beta/gamma or the wrong modal
%       basis changes these rates by O(1), which this still catches by three
%       decades.
% ---------------------------------------------------------------------------

if nargin < 1 || isempty(level), level = 'full'; end
level = lower(char(level));

here = fileparts(mfilename('fullpath'));
addpath(here);
R = load(fullfile(here, 'reference.mat'));

T_ALG = 1e-12; T_GCVD = 1e-10; T_REC = 1e-8;
T_INT = 1e-5;  T_STATE = 1e-5; T_FLOQ = 2e-5; T_BW = 1e-3;

np_ = 0; nf_ = 0; fails = {};
N    = 24;
J0   = R.cfg_j0_matlab1based;      % 2   (Python j0 = 1, zero-based)
MODE = R.cfg_mode_matlab1based;    % 7   (Python MODE = 6, zero-based)
DT   = R.cfg_dt;  SPAN = R.cfg_span;  RTOL = R.cfg_rtol;
ZET  = R.cfg_zeta1;  TSET = R.cfg_t_settle;

fprintf('\n=== Paper-3 MATLAB port self-test (level: %s) ===\n', level);
fprintf('MATLAB %s.  Reference: reference.mat exported from the Python sources.\n', version);
fprintf('Config: N=%d, station j0=%d (1-based), tracked mode=%d (1-based), ', N, J0, MODE);
fprintf('zeta1=%.3f, mu=%.1e\n\n', ZET, R.cfg_mu_ref);
t0 = tic;

%% ---------------------------------------------------- 1. chain structure ---
sect('1. Chain structure (lattice.m)');
L0 = lattice(N, J0, 0.0, ZET);
chk('chain_wn',            L0.wn,  R.chain_wn,  T_ALG, 'rel');
chk('chain_Phi',           L0.Phi, R.chain_Phi, T_ALG, 'rel');
chk('chain_psi (j0=2)',    L0.psi, R.chain_psi, T_ALG, 'rel');
chk('chain_ci (uniform)',  L0.ci,  R.chain_ci,  T_ALG, 'rel');
% psi must be the j0-th ROW of Phi, transposed -- catches a transpose slip
chk('psi == Phi(j0,:).''',  L0.psi, L0.Phi(J0,:).', 0, 'abs');

%% ------------------------------------------------- 2. analytic signal ------
sect('2. Analytic signal (hilbert_fft.m) vs scipy.signal.hilbert');
ze = hilbert_fft(R.hilb_even_x);
zo = hilbert_fft(R.hilb_odd_x);
chk('hilbert even, real part', real(ze), R.hilb_even_re,  T_ALG, 'rel');
chk('hilbert even, imag part', imag(ze), R.hilb_even_im,  T_ALG, 'rel');
chk('hilbert even, envelope',  abs(ze),  R.hilb_even_env, T_ALG, 'rel');
chk('hilbert odd, real part',  real(zo), R.hilb_odd_re,   T_ALG, 'rel');
chk('hilbert odd, imag part',  imag(zo), R.hilb_odd_im,   T_ALG, 'rel');
chk('hilbert odd, envelope',   abs(zo),  R.hilb_odd_env,  T_ALG, 'rel');

%% ------------------------------------------------- 3. GCVD, synthetic ------
sect('3. GCVD on an exact 6-mode record (gcvd.m, realify.m, gcvd_windowed.m)');
[lam, ~, ~] = gcvd(R.gcvd_disp, R.gcvd_vel, R.gcvd_acc, true);
chk('gcvd |lam| (sorted)', sort(abs(lam)), sort(R.gcvd_lam_abs), T_GCVD, 'rel');
g1 = gcvd_windowed(R.gcvd_disp, R.gcvd_vel, R.gcvd_acc, 1, 6, true, eye(6));
chk('tracked wn',     g1.wn(1,:).',   R.gcvd_track_wn,     T_GCVD, 'rel');
chk('tracked zeta',   g1.zeta(1,:).', R.gcvd_track_zeta,   T_GCVD, 'rel');
chk('tracked mpc',    g1.mpc(1,:).',  R.gcvd_track_mpc,    T_GCVD, 'rel');
chk('tracked energy', g1.energy(1,:).', R.gcvd_track_energy, T_GCVD, 'rel');
chk('tracked amp',    g1.amp(1,:).',  R.gcvd_track_amp,    T_GCVD, 'rel');
chk('tracked shapes', reshape(g1.shape(1,:,:), 6, 6), R.gcvd_track_shape, T_GCVD, 'rel');
% independent of Python: the synthetic record has KNOWN poles
chk('recovers true wn',   g1.wn(1,:).',   R.gcvd_true_wn,   1e-8, 'rel');
chk('recovers true zeta', g1.zeta(1,:).', R.gcvd_true_zeta, 1e-8, 'rel');
% the tracking/assignment branch (matchpairs vs linear_sum_assignment)
g3 = gcvd_windowed(R.gcvd_disp, R.gcvd_vel, R.gcvd_acc, 3, 3, true, eye(6));
chk('3-window tracked wn',   g3.wn,   R.gcvd_w3_wn,   T_GCVD, 'rel');
chk('3-window tracked zeta', g3.zeta, R.gcvd_w3_zeta, T_GCVD, 'rel');
chk('window edges', g3.edges(:), R.gcvd_w3_edges(:), 0, 'abs');
[ur, mr] = realify(complex(R.realify_v_re, R.realify_v_im));
chk('realify shape', ur, R.realify_u,   T_ALG, 'rel');
chk('realify mpc',   mr, R.realify_mpc, T_ALG, 'rel');

%% ------------------------------------------------- 4. RHS evaluations ------
sect('4. Equations of motion, evaluated pointwise (rhs_free.m, rhs_forced.m)');
Lr = lattice(N, J0, R.cfg_k3_ref, ZET);
ff = rhs_free(Lr);
fo = rhs_forced(Lr, R.rhs_forced_A, R.rhs_forced_Om);
chk('rhs_free(0, z)',    ff(0.0, R.rhs_state),            R.rhs_free_out,   T_ALG, 'rel');
chk('rhs_forced(t, z)',  fo(R.rhs_forced_t, R.rhs_state), R.rhs_forced_out, T_ALG, 'rel');

sect('5. Bouc-Wen element and stuck-basis structure (boucwen.m)');
S = boucwen();
chk('bw basis frequencies', S.wn,   R.bw_wn,   T_ALG, 'rel');
chk('bw basis psi-tilde',   S.psit, R.bw_psit, T_ALG, 'rel');
chk('bw beta == gamma == 0.5/uy', [S.beta; S.gamma], [R.bw_beta; R.bw_gamma], T_ALG, 'rel');
bf = S.rhs(0, 0);
bF = S.rhs(2e-2, S.wn(MODE));
chk('bw rhs, free',   bf(0.0,  R.bw_rhs_state), R.bw_rhs_free_out,   T_ALG, 'rel');
chk('bw rhs, forced', bF(3.25, R.bw_rhs_state), R.bw_rhs_forced_out, T_ALG, 'rel');

%% -------------------------------- 6. estimators on a SUPPLIED record -------
sect('6. Estimators on a supplied 2000-sample record (no integration)');
es = estimates(Lr, R.sub_disp, R.sub_vel, R.sub_acc, MODE, DT);
chk('sub r_gcvd',    es.r_gcvd,    R.subest_r_gcvd,    T_REC, 'rel');
chk('sub r_env',     es.r_env,     R.subest_r_env,     T_REC, 'rel');
chk('sub r_flux',    es.r_flux,    R.subest_r_flux,    T_REC, 'rel');
chk('sub stiff_pct', es.stiff_pct, R.subest_stiff_pct, T_REC, 'rel');
chk('sub a_m',       es.a_m,       R.subest_a_m,       T_REC, 'rel');
chk('sub w_rms',     es.w_rms,     R.subest_w_rms,     T_REC, 'rel');
chk('sub e_frac',    es.e_frac,    R.subest_e_frac,    T_REC, 'rel');
chk('sub cond',      es.cond,      R.subest_cond,      T_REC, 'rel');

sect('7. Sensor placement and projection conditioning (deterministic)');
% ROUNDING HAZARD: MATLAB round() is half-AWAY-FROM-ZERO, numpy's is
% half-to-EVEN.  linspace(1,24,5) contains the exact tie 12.5, so the two
% rules take different branches -- they happen to agree here only because
% Python rounds the zero-based grid and adds one.  The index SETS are
% therefore compared against the exported Python placement explicitly, so a
% divergence at some other sensor count is caught rather than silently
% changing which masses are instrumented.
sub5 = (MODE-2):(MODE+2);
cnd = zeros(numel(R.sensor_counts),1);
idx_match = 1;
for q = 1:numel(R.sensor_counts)
    mS  = R.sensor_counts(q);
    idx = unique(round(linspace(1, N, mS)));
    ref = R.sensor_idx_matlab(q,:); ref = ref(ref > 0);
    if numel(idx) ~= numel(ref) || any(idx(:).' ~= ref(:).')
        idx_match = 0;
        fprintf('      sensor placement differs at m=%d: MATLAB [%s] vs Python [%s]\n', ...
                mS, num2str(idx), num2str(ref));
    end
    cnd(q) = cond(L0.Phi(idx, sub5));
end
chk('sensor index sets match Python', idx_match, 1, 0, 'abs');
chk('cond(Phi_sensed) x6', cnd, R.sensor_cond, 1e-10, 'rel');

if strcmp(level, 'quick')
    finish();
    return
end

%% -------------------------------------- 8. settle + release (integrated) ---
sect('8. Settle and release at mu = 1e-3 (INTEGRATED -- ode89 vs DOP853)');
fprintf('    integrating t_settle=%g then span=%g ... ', TSET, SPAN);
tI = tic;
[tt, aa, vv, ac, y0] = settle_release(Lr, R.cfg_A_ref, MODE, SPAN, DT, RTOL, TSET);
fprintf('%.0f s\n', toc(tI));
chk('settled state y0 (48)', y0, R.settle_y0, T_STATE, 'rel');
ef = estimates(Lr, aa, vv, ac, MODE, DT);
chk('r_gcvd',    ef.r_gcvd,    R.est_r_gcvd,    T_INT, 'abs');
chk('r_env',     ef.r_env,     R.est_r_env,     T_INT, 'abs');
chk('r_flux',    ef.r_flux,    R.est_r_flux,    T_INT, 'abs');
chk('stiff_pct', ef.stiff_pct, R.est_stiff_pct, T_INT, 'abs');
chk('e_frac',    ef.e_frac,    R.est_e_frac,    T_INT, 'abs');
chk('w_rms',     ef.w_rms,     R.est_w_rms,     T_INT, 'abs');

sect('9. Share law on that record (energy_rates.m)');
er = energy_rates(Lr, tt, aa, vv, MODE, 1:N, R.cfg_edge);
chk('share mean',      er.share_mean,       R.share_mean,           T_INT, 'abs');
chk('share start',     er.share_start,      R.share_start,          T_INT, 'abs');
chk('share end',       er.share_end,        R.share_end,            T_INT, 'abs');
chk('d(ln f)/dt',      er.dlnf_tot,         R.share_dlnf_dt,        1e-8,  'abs');
chk('cbar/c_true',     er.cbar_mean_over_c, R.share_cbar_over_ctrue,T_INT, 'abs');
chk('r_derived',       er.r_derived,        R.share_r_derived,      T_INT, 'abs');

sect('10. Recovery-recipe subset table (carrier, +/-1, +/-2, +/-4, all)');
gsub = zeros(5,1); vsub = zeros(5,1);
for q = 1:5
    ss = R.subset_lo_matlab(q):R.subset_hi_matlab(q);
    e2 = energy_rates(Lr, tt, aa, vv, MODE, ss, R.cfg_edge);
    gsub(q) = e2.gamma_over_c;  vsub(q) = e2.vacuous_over_c;
    fprintf('      subset %2d:%-2d  gamma/c = %.6f   (Python %.6f)\n', ...
            ss(1), ss(end), gsub(q), R.subset_gamma_over_c(q));
end
chk('subset gamma/c x5', gsub, R.subset_gamma_over_c, T_INT, 'abs');
chk('vacuous correction x5', vsub, R.subset_vacuous, T_INT, 'abs');
chk('vacuity: spread across subsets', max(vsub)-min(vsub), 0, 1e-6, 'abs');

%% ------------------------------------------------------- 11. Floquet -------
sect('11. Floquet stability of the tracked NNM (floquet.m, monodromy.m)');
for q = 1:2
    if q == 1, tag = 'stable';   mu = R.floq_stable_mu;   rf = R.floq_stable_max_mult;
                                 wp = R.floq_stable_wpeak;
    else,      tag = 'unstable'; mu = R.floq_unstable_mu; rf = R.floq_unstable_max_mult;
                                 wp = R.floq_unstable_wpeak;
    end
    fprintf('    mu = %.1e (%s) ... ', mu, tag); tI = tic;
    fl = floquet(mu, R.floq_A, MODE, N, J0, ZET, R.floq_t_probe, R.floq_npts);
    fprintf('%.0f s\n', toc(tI));
    chk(sprintf('floquet %s: w_peak', tag),     fl.w_peak,         wp, T_INT,  'abs');
    chk(sprintf('floquet %s: max |mult|', tag), fl.max_multiplier, rf, T_FLOQ, 'abs');
    chk(sprintf('floquet %s: verdict', tag), double(fl.unstable), double(rf > 1), 0, 'abs');
end

%% ------------------------------------------------------- 12. Bouc-Wen ------
sect('12. Bouc-Wen record at A = 2e-2 (INTEGRATED, non-smooth RHS)');
fprintf('    integrating ... '); tI = tic;
bw = boucwen_record(R.bw_A, MODE, R.bw_t_settle, R.bw_span, R.bw_dt, R.bw_rtol, R.cfg_edge);
fprintf('%.0f s\n', toc(tI));
chk('bw slip ratio',       bw.slip_ratio,       R.bw_slip_ratio,       T_BW, 'rel');
chk('bw w_rms',            bw.w_rms,            R.bw_w_rms,            T_BW, 'rel');
chk('bw e_frac',           bw.e_frac,           R.bw_e_frac,           T_BW, 'rel');
chk('bw gamma_tot/c',      bw.gamma_tot_over_c, R.bw_gamma_tot_over_c, T_BW, 'rel');
chk('bw gamma_sub/c',      bw.gamma_sub_over_c, R.bw_gamma_sub_over_c, T_BW, 'rel');
chk('bw r_env/c',          bw.r_env_over_c,     R.bw_r_env_over_c,     T_BW, 'rel');
chk('bw settled state y0', bw.y0,               R.bw_y0,               T_BW, 'rel');

%% ------------------------------------------------- 13. bandwidth cell ------
sect('13. Band-forced cell, beta = 0.25 (rhs_forced.m multi-harmonic branch)');
fprintf('    integrating ... '); tI = tic;
Lb   = lattice(N, J0, R.band_k3, ZET);
solv = odesolver();
op9  = odeset('RelTol', 1e-9, 'AbsTol', 1e-11);
tev  = R.band_t_settle + R.band_releases(:).';
[~, Yb] = solv(rhs_forced(Lb, R.band_a, R.band_wk, R.band_ph), ...
               [0, tev], zeros(2*N,1), op9);
y0b  = Yb(2,:).';                               % release = 0 is tev(1)
ttb  = (0:round(SPAN/DT)-1).' * DT;
opR  = odeset('RelTol', RTOL, 'AbsTol', RTOL*1e-2);
[~, Yv] = solv(rhs_free(Lb), ttb, y0b, opR);
ab = Yv(:,1:N); vb = Yv(:,N+1:2*N);
cb = -(Lb.ci.').*vb - ((Lb.wn.^2).').*ab - (Lb.k3*(ab*Lb.psi).^3).*(Lb.psi.');
fprintf('%.0f s\n', toc(tI));
eb = estimates(Lb, ab, vb, cb, MODE, DT);
rb = energy_rates(Lb, ttb, ab, vb, MODE, 1:N, R.cfg_edge);
chk('band release state', y0b,          R.band_y0_release0, T_STATE, 'rel');
chk('band r_env',         eb.r_env,     R.band_r_env,       T_INT,   'abs');
chk('band r_gcvd',        eb.r_gcvd,    R.band_r_gcvd,      T_INT,   'abs');
chk('band share mean',    rb.share_mean,R.band_share_mean,  T_INT,   'abs');
chk('band d(ln f)/dt',    rb.dlnf_tot,  R.band_dlnf_dt,     1e-8,    'abs');

finish();

% ======================================================================= %
    function sect(txt)
        fprintf('\n-- %s\n', txt);
    end

    function chk(name, got, ref, tol, kind)
        g = double(got(:));  r = double(ref(:));
        if numel(g) ~= numel(r)
            ad = Inf; rd = Inf;
        else
            ad  = max(abs(g - r));
            den = max(abs(r));
            if den == 0 || ~isfinite(den), den = 1; end
            rd  = ad/den;
        end
        if strcmpi(kind, 'rel'), val = rd; else, val = ad; end
        ok = isfinite(val) && val <= tol;
        if numel(g) > 1, nm = sprintf('%s (n=%d)', name, numel(g)); else, nm = name; end
        if ok, tag = 'PASS'; np_ = np_ + 1;
        else,  tag = 'FAIL'; nf_ = nf_ + 1; fails{end+1} = nm; %#ok<AGROW>
        end
        if isempty(g), g = NaN; end
        if isempty(r), r = NaN; end
        fprintf('  [%s] %-36s got %- 15.8g ref %- 15.8g  |d| %8.2e  rel %8.2e  tol %8.2e (%s)\n', ...
                tag, nm, real(g(1)), real(r(1)), ad, rd, tol, kind);
    end

    function finish()
        fprintf('\n=====================================================================\n');
        if nf_ == 0
            fprintf('SUMMARY: %d checks, ALL PASS.  (%.0f s)\n', np_, toc(t0));
            fprintf('This port reproduces the Python reference within the stated tolerances.\n');
        else
            fprintf('SUMMARY: %d checks, %d PASS, %d FAIL.  (%.0f s)\n', ...
                    np_+nf_, np_, nf_, toc(t0));
            fprintf('FAILED: %s\n', strjoin(fails, ', '));
            fprintf(['Read the failures top-down: the earliest section to fail localises\n' ...
                     'the bug, because later sections are built on earlier ones.\n']);
        end
        fprintf('=====================================================================\n\n');
    end
end
