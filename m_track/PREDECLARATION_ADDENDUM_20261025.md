# Addendum 1 to PREDECLARATION.md (25 Oct 2026, written after M1/M3/M5/M6 finished, before any M3b run)

The frozen pre-declaration is unchanged (sha256 6959eefe…). This addendum adds one run that the M3 result calls for. It changes no existing rule.

## Why

M3 confirmed the theory's lower (6,8) instability window at ζ₁ = 0.005: the ω_lin orbit is unstable from µ ≈ 1e-5 to ≈ 6e-5. That is below the port's reported ζ₁ = 0.005 onset, µ_c = 3.0e-4, which is the (5,9) boundary. The port's stability statement is therefore not "the first loss of stability" at that damping. Whether the window matters for identification is a separate question. The largest multiplier there is about 1.002 per forcing period, and the deterministic protocol has no seed beyond round-off.

## M3b, declared now

Run the port protocol (`bias_map._settle` + `Lattice.decay`, A = 0.02, 6000-unit build-up, first 400 units of decay, Δt = 0.05, DOP853 rtol 1e-9) at ζ₁ = 0.005 and µ ∈ {1e-5, 2.5e-5, 4e-5, 6e-5}, with µ = 1.5e-4 as an in-range stable control. Compute the port's three estimates (envelope, GCVD, share).

- **Q1.** If every record in the window has share ≥ 0.99 and |r_env − 1| ≤ 0.005: the window is "unstable but identification-silent under this deterministic protocol". The port then qualifies its onset sentence (first loss of stability *that biases the estimate*) and points to seeded preparation (M4) as untested.
- **Q2.** If any record in the window has share < 0.99 or |r_env − 1| > 0.005: the port's ζ₁ = 0.005 onset value is wrong as an identification boundary and must be revised.

No other outcome is declared.
