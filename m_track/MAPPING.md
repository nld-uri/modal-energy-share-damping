# M track: mapping between the port (paper3.tex) and Alt B

Written 25 Oct 2026, before any run. Sources: `paper3.tex` §§2–7, `code/` of the paper-3 folder, and Alt B's working tree (`Alternative Papers/paper_b/working/paper3_nd_code`).

## Same system, same ω_lin protocol

| item | port | Alt B | same? |
|---|---|---|---|
| chain | n = 24 unit masses, unit springs, fixed–fixed | same | yes: `models/lattice_nl.py` byte-identical |
| element and station | grounded cubic at the second mass (`J0 = 1`, 0-based) | same; force at the same station | yes |
| tracked mode | m = 7 (`MODE = 6`, 0-based), ψ₇ = 0.2778 | same | yes |
| damping | uniform modal c_i = 2ζ₁ω₁, ζ₁ = 0.010 nominal; 0.005 and 0.020 in the damping sweep | same | yes |
| forcing | single tone at Ω = ω₇, from rest, 6000 time units, A = 0.02 in the onset sweep, k₃ = µ/A² | "ω_lin from rest", A = 0.02, T_settle = 6000 | yes: `experiments/bias_map.py` byte-identical |
| window | first 400 time units of the free decay, Δt = 0.05, DOP853 rtol 1e-9 | same | yes |
| Floquet | Newton-shot orbit, 48-dimensional monodromy (`onset_mechanism.py`) | same script with a 2-line difference; the stored `onset_mechanism_results.json` is byte-identical | yes |
| GCVD readout | `identification/gcvd.py` (eigenvalue readout) | different file (105 differing lines) | not needed for M1/M3; M6 uses no GCVD |

**Consequence:** for the ω_lin protocol there is one framework. Alt B's κ results for ω_lin describe the same orbits the port parametrises by µ. Per the M-track rule, every number proposed for the port is recomputed here from the port's code and model. None is copied from Alt B's text.

## The map between κ and µ

- Port: µ = k₃A², with A the force amplitude (§4, Eq. `eq:mu`). Every dimensionless response is a function of µ alone, at fixed ζ₁, Ω, station and mode.
- Alt B: κ = k₃ w²_peak, with w = ψ·q the station displacement and w_peak its peak over the forced periodic orbit. The port uses the same quantity once, in its single-harmonic cross-check (§6), as `k3_wpeak2`.
- The map κ(µ; ζ₁) is the forced-response curve of the converged orbit. It is **not** a constant: by the scaling, κ = µ·(w_peak/A)², with w_peak/A the orbit's station gain. The gain falls with amplitude as the mode stiffens off resonance, and it depends on ζ₁ because the force sits at the linear resonance.
- The port's stored Newton-shot records (`checkpoints/onset_mechanism_results.json`, `converged`) carry κ for 16 values of µ at each of ζ₁ = 0.005, 0.010, 0.020. For example, at ζ₁ = 0.010: µ = 3.8e-4 → κ = 0.08328 and µ = 4.0e-4 → κ = 0.08490. One record (ζ₁ = 0.005, µ = 5e-4) is a spurious Newton solution (κ = 46.6, residual 6e-3) and is excluded, as in the port.
- **Lowest µ sampled is 1e-4** at every damping value, i.e. κ = 0.0545, 0.0485 and 0.0287. Nothing below that has been computed in the port.

## What each M item needs in the port's framework

| item | port variable | new computation needed |
|---|---|---|
| M1 averaged condition | κ from the theory, mapped to µ through the port's own κ(µ) | theory evaluation (seconds); κ→µ by interpolating the port's converged orbits, plus new orbits where the theory points outside the sampled range |
| M3 Floquet across damping | µ | new Newton-shot orbits below µ = 1e-4, where the theory may predict a lower instability interval |
| M2 phase resonance | µ = k₃A² with A solved per Ω on the phase-resonant family | a new protocol, as the port has none |
| M4 growth budget | µ, seeds, hold time | new hold protocol and seeded scans |
| M5 share vs damping error | the port's 35-cell map | stored results |
| M6 model-informed recovery | µ | re-integrate the port protocol at chosen µ; regression |
| M7 further configurations | µ at other stations and modes | new configuration screens |
| M8 PR frozen-orbit gain | needs M2 | — |
| M9 supplement | follows M1–M4 | — |
