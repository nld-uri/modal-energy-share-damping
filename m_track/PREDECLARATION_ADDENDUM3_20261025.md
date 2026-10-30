# Addendum 3 to PREDECLARATION.md (25 Oct 2026; written before any M7 run)

The frozen pre-declaration is unchanged. M7 meets the "≤ 30 min" cost rule (screen in seconds; 2 × 16 Newton-shot orbits; about 15 release records), so it is run as declared. This addendum fixes how it is read.

## M7, further station/mode configurations, port's ω_lin protocol (ζ₁ = 0.010, A = 0.02)

The configurations are those Alt B's screen (`theory_second_config_screen.py`) selects, re-run on the port's model: the screen itself is re-run here, and its choice is not taken from Alt B's stored file.

- **P7.1.** For each configuration, the averaged condition's predicted boundary (lower edge of the first window, mapped to µ through that configuration's own converged orbits) lies inside the Floquet bracket of adjacent converged orbits.
- **P7.2.** The pair dominating the leading Floquet eigenvector at the first unstable orbit equals the predicted pair.
- **P7.3.** In the release sweep, the envelope ratio departs from 1 by more than 0.5 % only at µ at or above the Floquet boundary.
- **Reading for the port.** If every configuration passes P7.1–P7.3, the port's generality caveat can cite them as further configurations. Any failure is reported as a limit of the isolated-pair condition, naming the configuration.
