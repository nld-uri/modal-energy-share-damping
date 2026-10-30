# M track: pre-declaration addendum 4 — M4 and M8 (written 25 Oct 2026, before any M4 or M8 run)

## Why this addendum exists

The original pre-declaration gave M2, M4, M7 and M8 a **cost gate only**: "estimate
the wall time ... run only if the estimate is <= 30 min, else report the cost and
stop". M4 was estimated at 35-45 min and M8 at hours, so both were correctly stopped
and neither has any scientific criterion on record. David has now authorised the run
directly, which overrides the budget rule but not the discipline: criteria are fixed
here, before results are seen, as was done for M3b (addendum 1), M2 (addendum 2) and
M7 (addendum 3).

## Provenance rule for this addendum

M4 is re-run here from Alt B's code on this machine; its inputs
(`protocols_release_records.json`, the PR family) are regenerated, not inherited.

M8's raw inputs are ~17 000 releases archived by Alt B as
`checkpoints/growth_ext/growth_ext_raw_records.tar.gz`. If those raw records are
used rather than regenerated, M8 is a **re-analysis of Alt B's data, not an
independent replication**, and every M8 number entering the manuscript must say so.
This is declared now so the distinction cannot be made after seeing the result.

## M4: does the linear growth budget predict the post-hold sideband share?

Computed (Alt B's `protocols_growth_budget.py`, unmodified apart from its location):
for each non-`PR-ramp` release record, the seed pair-energy share of the
perturbation at release, the linear growth factor `max|lam|^(2N)` with
`N = T_SETTLE / T_orbit`, the predicted post-hold pair share
`seed * max|lam|^(2N)`, and the observed share (`share_6_8` for PR-hold with pair
(5,7); `share_5_9` for LIN-hold with pair (4,8)). Also the kappa at which the
predicted share reaches 0.1, for the PR and LIN families separately.

- **P4.1, no missed failure.** At every record whose predicted pair share is
  below 0.1, the observed pair share is also below 0.1.
- **P4.2, linear growth is an over-estimate, not an under-estimate.** At every
  record, `observed <= 10 * predicted`. Saturation may make the observed share
  much smaller than the linear budget; the budget falling an order of magnitude
  short of what actually grew would mean the mechanism is not this one.
- **P4.3, ordering.** The two predicted failure thresholds `kappa_fail_PR` and
  `kappa_fail_LIN` are both finite and inside the sampled kappa range
  [0.010, 0.120].

Refuted if P4.1 or P4.2 fails at any record. No direction is predicted for which
of the two protocols fails first: that comparison is **reported, not tested**,
because nothing in the port or in M1-M7 fixes its sign in advance.

## M8: do frozen forced-PR-orbit rates explain the release-to-window gain?

Computed (Alt B's `growth_ext_gain_cause.py`): for each tested record, the
surrogate `r_pred(t) = prod_blocks |lam_pair(kappa_blk)|^(2 dt / T_PR(kappa_blk))`
using pair-resolved Floquet rates of the forced phase-resonant family at the
current amplitude, against the observed release-to-window pair-energy gain, for
the PR pair (5,7)/(6,8) and the LIN pair (4,8)/(5,9).

Alt B's docstring asserts the surrogate "describes the PR gain in the tested
records and fails for the LIN pair". That assertion is the hypothesis under test:

- **P8.1, PR pair.** The median over tested records of `|log10(r_pred / r_obs)|`
  is <= 0.5 (agreement within a factor of about three).
- **P8.2, LIN pair.** The same median is > 0.5, i.e. the surrogate does measurably
  worse on the pair for which Alt B says it fails.

Refuted if P8.1 fails: the surrogate would then not describe the gain it was
introduced to explain. If P8.1 holds and P8.2 fails, the surrogate is better than
Alt B claims and the asymmetry it rests on is not reproduced; that outcome is
reported as such and no "fails for the LIN pair" statement may enter the paper.

## Reporting

As for M1-M7: every number proposed for the manuscript is produced by a check in
`verify_m_track.py` from a stored JSON, printed at the precision the check uses,
and the check searches the proposed text for the printed string.
