# M track: results (25 Oct 2026)

Every number below is checked by `verify_m_track.py` (80/80) against `results/*.json` and `m2/`, `m7/` checkpoints. The draft text is in `proposed_sections.tex`, with insertion points; nothing has been inserted into `paper3_nd.tex`. Rules: `PREDECLARATION.md` (frozen, hash unchanged) plus addenda 1–3, each hashed before its run.

| item | status | pre-declared rule | key result |
|---|---|---|---|
| M1 averaged condition, ω_lin | done | P1.1–P1.4 **pass** | (5,9) at κ_c = 0.0801 / 0.0837 / 0.101, inside every Floquet bracket; µ_c within 0.9 % of the port; 47 orbits, max multiplier gap 7.5e-5. The mean shift is what demotes (6,8). Damping rule triggered: κ_c ratio 1.26 against µ ratio 2.95 |
| M3 Floquet below the sampled range | done | P3.1–P3.2 **pass** | ζ₁ = 0.005 has a lower (6,8) window, unstable from µ = 9.6e-6 to between 6e-5 and 7e-5, max multiplier 1.0023; ζ₁ = 0.010 none |
| M3b (addendum 1) | done | **Q1** (silent) | share ≥ 0.998 and envelope within 0.17 % inside that window |
| M2 phase resonance, Floquet only (addendum 2) | done, ~1 min | P2.1 **fail as worded** (averaged edge 0.04–0.41 % above bisection brackets about 2e-5 wide); P2.2 **pass** | onset pair (6,8) at κ = 0.0297 / 0.0404 / 0.0827; averaged 0.0297 / 0.0405 / 0.0831. No release records |
| M5 share vs error, port map | done | n/a (descriptive) | 25 high-share cells all within 0.5 % (worst: envelope 0.48 %, GCVD 0.35 %); 10 low-share cells all outside |
| M6 model-informed recovery | done | P6.1–P6.3 **pass** | at µ = 1.6e-3 (envelope 0.451): c to 2.0e-14 and k₃ to 1.0e-14 (model acceleration); 4.6e-7 and 2.3e-6 (differenced) |
| M7 further configurations (addendum 3) | done, ~5 min | primary: P7.1 **fail** (−0.7 %), P7.2–P7.3 pass; secondary: P7.1–P7.2 **fail**, P7.3 pass | secondary: predicted (1,14) at 0.0823, Floquet (14,19) at 0.0959–0.0963. Envelope within 0.12 % below the boundary in both |
| M4 growth budget | run 25 Oct 2026 | P4.1 pass, P4.2 REFUTED (16/28), P4.3 pass | independent re-run; reproduces Alt B exactly. Budget is an upper bound once it predicts >= 1e-3 (11 records, ratio <= 0.95); below that the observed share sits on a floor <= 5e-4 the budget does not model. kappa_fail 0.0695 (PR), 0.0951 (LIN). In the manuscript in restricted form. |
| M8 PR frozen-orbit gain | run 25 Oct 2026 | P8.1 pass (PR median 0.2475), P8.2 FAIL (LIN median 0.0395) | RE-ANALYSIS OF ALT B's ARCHIVED RECORDS, not an independent replication; reproduces Alt B exactly. Alt B's 'fails for the LIN pair' is NOT reproduced - the surrogate is 6x better on LIN at the median; the asymmetry is tail-only (LIN 0.65 % of records beyond 1.0, PR none). Not placed in the manuscript. |
| M9 supplement | not started | — | follows from what is accepted |

Process notes: the M7 screen ran with Alt B's default pool of 6 workers (199 s); every other run used ≤ 4. The M2 and M7 trees use Alt B's scripts on the port's model files. M7's only edit is the pool size.


## Addendum 4 (M4, M8) — 25 Oct 2026

Both were stopped originally by the cost gate and had no scientific criteria on record. David authorised the run; criteria were fixed in `PREDECLARATION_ADDENDUM4_20261025.md` (sha256 ac95d5697c00b6a07029f53845f42a20af975809b63ef57f2737375d8a8d906a) before either ran. The 35-45 min M4 estimate was wrong: the chain took 56 s on this machine.
