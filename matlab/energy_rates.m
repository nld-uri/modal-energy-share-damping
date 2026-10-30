function out = energy_rates(L, tt, aa, vv, mode, subset, edge)
%ENERGY_RATES  Modal energies, share f, d(ln f)/dt and gamma over a subset.
%
%   Port of the shared machinery of  share_law.one_record  and
%   recovery_recipe.one_mu.
%
%   out = ENERGY_RATES(L, tt, aa, vv, mode, subset, edge)
%       mode   ONE-BASED tracked mode (7 for the study)
%       subset ONE-BASED modal indices summed for the subset energy
%              (default 1:n, i.e. "all modes"); the study's recipe is
%              subset = mode-2 : mode+2
%       edge   trim 1/edge of the window at each end before fitting (20)
%
%   THE LAW.  Write the tracked mode's energy as E_m = f E_tot.  The total
%   decays at the energy-weighted mean damping cbar, and an envelope fit to
%   mode m reports whatever exponential rate E_m shows, so
%
%       c_ident = cbar - d(ln f)/dt .
%
%   Here the modal damping is UNIFORM, so cbar == c_true identically and the
%   ENTIRE bias is the logarithmic drift of the energy share.  Mixing cannot
%   produce bias in this system, which is what makes the second term visible.
%
%   THE VACUOUS CORRECTION.  vacuous_over_c is gamma_S - d(ln f_S)/dt, which
%   by the product rule equals -d(ln E_m)/dt for ANY subset S -- i.e. it
%   returns the single-mode estimate it was meant to correct.  It is computed
%   here so run_all can DEMONSTRATE the identity across subsets rather than
%   assert it; it is not a usable correction.
%
%   Two shares are returned because the two source scripts use different
%   denominators: share_tot (vs total energy, share_law) and share_sub
%   (vs the subset, recovery_recipe).

if nargin < 6 || isempty(subset), subset = 1:L.n; end
if nargin < 7 || isempty(edge),   edge   = 20;    end

tt = tt(:);
m  = numel(tt);

Ei   = 0.5*(vv.^2 + ((L.wn.^2).').*aa.^2);
Etot = sum(Ei, 2);
Esub = sum(Ei(:,subset), 2);

k  = floor(m/edge);
sl = (k+1):(m-k);

f_sub = Ei(:,mode)./Esub;
f_tot = Ei(:,mode)./Etot;
cbar  = (Ei*L.ci)./Etot;              % energy-weighted mean modal damping

p = polyfit(tt(sl), log(f_sub(sl)), 1);  out.dlnf_sub = p(1);
p = polyfit(tt(sl), log(f_tot(sl)), 1);  out.dlnf_tot = p(1);
p = polyfit(tt(sl), log(Esub(sl)),  1);  out.gamma    = -p(1);

out.gamma_over_c      = out.gamma/L.ci(mode);
out.vacuous_over_c    = (out.gamma - out.dlnf_sub)/L.ci(mode);
out.cbar_mean_over_c  = mean(cbar(sl))/L.ci(mode);
out.r_derived         = (mean(cbar(sl)) - out.dlnf_tot)/L.ci(mode);
out.share_mean        = mean(f_tot);
out.share_start       = f_tot(1);
out.share_end         = f_tot(end);
out.share_tot         = f_tot;
out.share_sub         = f_sub;
out.Ei                = Ei;
out.Etot              = Etot;
out.Esub              = Esub;
end
