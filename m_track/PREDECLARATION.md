# M track: pre-declaration (written 25 Oct 2026, before any M-track run)

Model, protocol and variable: the port's (see `MAPPING.md`). Reference environment: `/Users/chelo/.claude-science/conda/envs/python` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1), one BLAS thread per process. Port values used as targets come from `checkpoints/onset_mechanism_results.json` and `checkpoints/bias_map_results.json` (the stored results the port's verifier reads).

The theory code is a copy of Alt B's `code/experiments/theory_averaged_threshold.py`, saved as `m_track/lib/averaged.py` and unmodified apart from its location. It is evaluated on the port's `Lattice` object.

## M1: averaged combination-resonance condition, ω_lin protocol

Computed: for ζ₁ ∈ {0.005, 0.010, 0.020} at Ω = ω₇, the unstable κ-intervals of every sideband pair (carrier excluded). Predicted µ_c is the lower edge of the interval that contains the port's Floquet boundary, mapped to µ by log–log interpolation of the port's own converged κ(µ) records (spurious record excluded).

- **P1.1, boundary.** The predicted κ_c of the boundary the port reports lies inside the port's Floquet bracket in κ at all three damping values. The bracket runs from the last stable to the first unstable converged record, within the declared interval. The predicted µ_c is also within 5 % of the port's interpolated µ_c (2.955e-4, 3.849e-4, 8.704e-4).
- **P1.2, pair.** The pair whose interval gives that boundary is (5,9) at all three damping values. This is the port's leading-eigenvector pair.
- **P1.3, multipliers.** At every valid converged record (47 of 48), |λ_pred − λ_Floquet| ≤ 1.0e-3, where λ_pred is the averaged prediction from the record's own κ (max over pairs and over the passive exp(−ηT)). The two also never disagree on which side of 1 a record lies, except within 2e-4 of 1.
- **P1.4, why (6,8) loses.** At the ζ₁ = 0.010 boundary, the (6,8) margin is negative while (5,9) is positive. Both raw detuning and raw ψ-product favour (6,8), as the port states, so the averaged condition supplies the "third" quantity the port says it does not offer.

Refuted if any of P1.1–P1.3 fails at any damping value.

## M3: Floquet across damping, below the port's sampled range

Computed: the theory's instability intervals with lower edges below the port's lowest sampled µ (1e-4). For each damping value where an interval is predicted, plus ζ₁ = 0.010 as a control, run new Newton-shot Floquet orbits at µ ∈ {5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5}, A = 0.02.

- **P3.1.** Every new record agrees with the theory about which side of |λ| = 1 it lies on, except records within one grid step of a predicted edge.
- **P3.2.** The pair dominating the leading Floquet eigenvector of each unstable new record is the predicted pair.

Refuted if a record more than one grid step inside a predicted interval is stable, or a record outside every predicted interval is unstable.

**Interpretation rule, declared now.** Let r_κ = κ_c(0.020)/κ_c(0.005) for the (5,9) boundary, computed from the port's records. If r_κ < 1.5 while the µ ratio is 2.95, the port's sentence that a fourfold change in damping moves the threshold by 2.95 has to be qualified: most of that dependence is the A → w_peak forced-response map, not the stability problem. If any lower interval is confirmed, the port's "boundary" at that damping value is not the first loss of stability of the ω_lin orbit.

## M6: model-informed recovery of c and k₃

Computed: the port protocol (A = 0.02, ζ₁ = 0.010, 6000 settle, first 400 units of free decay, Δt = 0.05, DOP853 rtol 1e-9) at µ ∈ {0, 3e-4, 1.6e-3}, with k₃ = µ/A² and k₃ = 0 at µ = 0. Joint least squares for the common c and k₃ in −q̈_i − ω_i²q_i = c q̇_i + k₃ψ_i w³, stacked over all 24 modes and all samples. Variant (a) uses the model acceleration; variant (b) uses a fourth-order centred difference of the sampled velocity, with 2 samples trimmed at each end. The same records also give the port's envelope ratio, for context.

- **P6.1.** Variant (a): |ĉ/c − 1| ≤ 1e-4 and |k̂₃/k₃ − 1| ≤ 1e-4 at µ = 3e-4 and 1.6e-3; |k̂₃| ≤ 1e-6 at µ = 0.
- **P6.2.** Variant (b): the same at ≤ 1e-3.
- **P6.3.** At µ = 1.6e-3 the envelope ratio recomputed on this record departs from 1 by more than 10 %, so the record is "strongly biased" for the response-only estimator.

Refuted if P6.1 or P6.2 fails at µ = 1.6e-3. That would mean the bias is not merely a readout of an equivalent rate.

## M5: share threshold against damping error, on the port's own map

Computed from stored results only: the port's 35 cells (25 grid plus 10 threshold sweep), using share = `e_frac` and errors |r_env − 1| and |r_gcvd − 1|, with tolerance τ = 0.005 and share threshold 0.99.

- Report N₁ = cells with share ≥ 0.99 and error > τ, and N₂ = cells with share < 0.99 and error ≤ τ, for each estimator.
- No pass/fail on generality. If N₁ = 0, the port may say the share rule holds on its 35 cells, and must say that seeded preparations (M4) are untested. If N₁ > 0, the port's "below the onset there is no bias" must be narrowed.

## M2, M4, M7, M8: cost first

For each, estimate the wall time from Alt B's code and stage costs (`code/run_paper3.py` STAGES), with at most one timed probe. Run only if the estimate is ≤ 30 min, else report the cost and stop. M8 depends on M2 and M4. M9 follows from M1–M4.

## Rounding and reporting

Every number proposed for the manuscript is produced by `verify_m_track.py` from a JSON in `results/`. It is printed at the precision the check uses, and the check searches the proposed text for the printed string. Nothing is taken from Alt B text.
