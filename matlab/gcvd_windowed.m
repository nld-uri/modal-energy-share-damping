function out = gcvd_windowed(disp_, vel, acc, n_windows, n_track, center, M)
%GCVD_WINDOWED  Windowed GCVD with resonance tracking across windows.
%
%   Port of  windowed_gcvd()  in  gcvd.py.
%
%   out = GCVD_WINDOWED(disp, vel, acc, n_windows, n_track, center, M)
%
%   Splits the record into n_windows equal windows, decomposes each, and
%   tracks n_track resonances across them by combined MAC and log-frequency
%   proximity, solved as a linear assignment problem.  Pass M = eye(n) to get
%   modal-amplitude and modal-energy estimates; M = [] to skip them.
%
%   Fields (n_windows x n_track unless noted): wn, zeta, mpc, amp, energy;
%   shape is (n_windows x n_track x n); edges is the 1 x (n_windows+1) vector
%   of window boundaries as ZERO-BASED sample offsets (as in Python), so
%   window w covers MATLAB rows edges(w)+1 : edges(w+1).
%
%   ASSIGNMENT.  scipy.optimize.linear_sum_assignment is replaced by
%   matchpairs (base MATLAB since R2019a, no toolbox) with a large
%   unmatched-cost so every tracked resonance is forced to take a partner.
%   Both solve the same problem; with distinct costs the optimum is unique,
%   so they agree.  estimates.m calls this with n_windows = 1, where the
%   assignment never runs at all -- selftest.m exercises the tracking branch
%   separately (n_windows = 3) so a matchpairs discrepancy is visible on its
%   own line rather than hiding inside an estimate.
%
%   std(...,1) is deliberate: numpy's .std() is the POPULATION standard
%   deviation (ddof = 0), MATLAB's default std() is the sample one (ddof = 1).

if nargin < 4 || isempty(n_windows), n_windows = 48;    end
if nargin < 5 || isempty(n_track),   n_track   = 3;     end
if nargin < 6 || isempty(center),    center    = true;  end
if nargin < 7,                       M         = [];    end

[m, n] = size(disp_);
ed = floor(linspace(0, m, n_windows+1));     % numpy .astype(int) truncates

out = struct();
out.wn     = nan(n_windows, n_track);
out.zeta   = nan(n_windows, n_track);
out.mpc    = nan(n_windows, n_track);
out.amp    = nan(n_windows, n_track);
out.energy = nan(n_windows, n_track);
out.shape  = nan(n_windows, n_track, n);
out.edges  = ed;

prev_shape = [];
prev_wn    = [];

for w = 1:n_windows
    rows = (ed(w)+1):ed(w+1);
    [lam, V, ~] = gcvd(disp_(rows,:), vel(rows,:), acc(rows,:), center);

    ii = find(imag(lam) > 1e-6);
    [~, o] = sort(abs(lam(ii)));             % stable in both languages
    idx = ii(o);
    nc  = numel(idx);

    c_wn = zeros(nc,1); c_ze = zeros(nc,1);
    c_sh = zeros(n,nc); c_mp = zeros(nc,1);
    for j = 1:nc
        i = idx(j);
        c_wn(j) = abs(lam(i));
        c_ze(j) = -real(lam(i))/abs(lam(i));
        [uu, mm] = realify(V(1:n, i));
        c_sh(:,j) = uu;
        c_mp(j)   = mm;
    end

    if isempty(prev_wn)
        sel = 1:min(n_track, nc);
    else
        np_  = numel(prev_wn);
        cost = zeros(np_, nc);
        for k = 1:np_
            for j = 1:nc
                cost(k,j) = (1 - (c_sh(:,j).'*prev_shape(:,k))^2) ...
                            + abs(log(c_wn(j)/prev_wn(k)));
            end
        end
        Mp  = matchpairs(cost, 1e6);         % forces every row to be matched
        sel = zeros(1, np_);
        for r = 1:size(Mp,1)
            sel(Mp(r,1)) = Mp(r,2);
        end
    end

    dX = disp_(rows,:) - mean(disp_(rows,:), 1);
    for k = 1:numel(sel)
        j = sel(k);
        if j == 0, continue; end
        out.wn(w,k)      = c_wn(j);
        out.zeta(w,k)    = c_ze(j);
        out.shape(w,k,:) = c_sh(:,j);
        out.mpc(w,k)     = c_mp(j);
        if ~isempty(M)
            vm = c_sh(:,j)/sqrt(c_sh(:,j).'*M*c_sh(:,j));   % mass-normalised
            a  = sqrt(2)*std(dX*(M*vm), 1);                 % ddof = 0
            out.amp(w,k)    = a*abs(vm(end));
            out.energy(w,k) = 0.5*(c_wn(j)*a)^2;
        end
    end

    keep       = sel(sel > 0);
    prev_shape = c_sh(:, keep);
    prev_wn    = c_wn(keep);
end
end
