# Paper 3 — damping identifiability on a nonlinear lattice: MATLAB port

## Read this first

**The MATLAB in this package was never executed by its author.** It was written in an
environment where MATLAB cannot run (the sandbox denies the `sysctl` syscall, so MATLAB's
Qt layer misreads the CPU feature probe and aborts with *"Incompatible processor … requires
neon"*; Octave segfaults on `--eval`). Every line of it is therefore **untested code**.

`selftest.m` is how you establish that it works. It compares this port against reference
vectors exported from the verbatim Python sources and prints, for each of 76 checks, the
computed value, the reference value, the absolute and relative difference, and the
tolerance used. Run it before you trust any number this package produces:

```matlab
cd matlab
selftest          % all 76 checks, integrates (estimated 10–30 min, see note)
selftest quick    % the 42 algebraic checks only (seconds)
```

Section [What was verified without MATLAB](#what-was-verified-without-matlab) states exactly
which failure modes were ruled out before shipping and which could not be.

## The study, in one paragraph

A fixed–fixed chain of 24 unit masses carries one grounded cubic spring at the second mass.
The chain is chosen for its *spectrum*: a beam's `ω²` grows as `i⁴` and filters high modes
out of any amplitude-dependent correction, whereas a lattice dispersion relation saturates
at the band edge, so the corrections keep their high-mode content. Modal damping is
**uniform** (`c_i = 2ζ₁ω₁` for every `i`), which is the experiment's key design choice: any
energy-weighted mixture of modal damping values is then exactly the true value, so mixing
can produce no bias at all and whatever bias remains must come from somewhere else. Each
record is prepared by a fixed protocol — resonant station forcing builds mode 7 up from
rest, the force is released, and the first window of the ensuing free decay is analysed —
and carries three estimates of the same modal damping: a Hilbert log-envelope fit (the
reference measurement, needing no decomposition), the windowed GCVD state relation (the
method under test), and the exact modal energy-flux prediction. The result is that the
identification error is the **logarithmic drift of the tracked mode's share of the record
energy**, `c_ident = c̄ − d(ln f)/dt`, with an onset set by a sideband instability: past
`μ = k₃A² ≈ 4×10⁻⁴` the tracked NNM loses stability to a symmetric `m±2` sideband pair, so
the onset is predictable from a Floquet analysis rather than being an empirical threshold —
damping identifiability and nonlinear-normal-mode stability are the same question.
Recovering the estimand needs energy summed over the carrier **and** that sideband pair
(0.4 % error, against 54 % for the single mode and 50 % for the naive nearest-neighbour
choice). A Bouc-Wen element at the same station marks the boundary: a *dissipative*
nonlinearity does net work over a cycle, so the estimand of the conservative theory no
longer exists and the identified damping is measuring a different, physically meaningful
quantity.

## Contents

```
matlab/
  lattice.m           chain modes, station influence vector psi, uniform modal damping
  rhs_free.m          free modal EOM with the grounded cubic
  rhs_forced.m        the same with station forcing A cos(Om t)  (also multi-harmonic)
  settle_release.m    force to steady state, release, return the free-decay window
  estimates.m         the three estimates per record
  gcvd.m              generalised characteristic value decomposition
  gcvd_windowed.m     windowed GCVD with cross-window resonance tracking
  realify.m           phase-rotate a complex mode to maximum collinearity, + MPC
  hilbert_fft.m       analytic signal via FFT (replaces the toolbox hilbert)
  energy_rates.m      modal energies, share f, d(ln f)/dt, gamma over a modal subset
  floquet.m           stability of the settled orbit at a given mu
  monodromy.m         monodromy of the linearisation about a carrier, max |multiplier|
  boucwen.m           hysteretic element + stuck-basis reference structure
  boucwen_record.m    settle / release / analyse one hysteretic record
  odesolver.m         picks ode89, falls back to ode45
  run_all.m           reproduces the paper's tables and prints them
  selftest.m          the 76 checks
  reference.mat       reference vectors (MATLAB v5+zlib, i.e. a "v7" file)
  reference_scalars.csv   the scalar references in plain text, for eyeballing

python/
  models/lattice_nl.py           |  the nine source scripts, VERBATIM
  identification/gcvd.py         |
  experiments/bias_map.py        |
  analysis/onset_mechanism.py    |
  analysis/share_law.py          |
  analysis/recovery_recipe.py    |
  analysis/friction_boundary.py  |
  analysis/noise_sensing.py      |
  analysis/bandwidth_axis.py     |
  export_reference.py            regenerates matlab/reference.mat
  verification/mlmirror.py       line-by-line back-translation of the MATLAB
  verification/check_mirror.py   runs that back-translation against reference.mat
```

## Toolbox-free

**Base MATLAB only. No toolboxes are used or required.** In particular the Signal
Processing Toolbox is not used: `hilbert_fft.m` builds the analytic signal from `fft`/`ifft`
directly. The only functions called outside the core language are `ode89`/`ode45`, `odeset`,
`eig`, `svd`, `pinv`, `cond`, `polyfit`, `matchpairs`, `griddedInterpolant`, `fft`, `ifft`,
`randn` and `rng` — all in base MATLAB. `matchpairs` requires R2019a or later and `ode89`
requires R2021b or later; you are on R2026a. `odesolver.m` degrades to `ode45` with a
warning if `ode89` is absent.

## How to run the Python

The nine scripts under `python/` are unmodified. They expect to sit in sibling directories
(`models`, `identification`, `experiments`) and write JSON into a `checkpoints/` directory
two levels up, which is how they are laid out in the paper repository. Requires numpy and
scipy only.

```bash
cd python/experiments && python bias_map.py          # writes bias_map_results.json
cd python/analysis    && python onset_mechanism.py   # Floquet, sidebands
                         python share_law.py         # the derived law
                         python recovery_recipe.py   # which modes must be observed
                         python friction_boundary.py # the dissipative boundary case
                         python noise_sensing.py     # noise + partial instrumentation
                         python bandwidth_axis.py    # the second axis
```

To regenerate the reference file after changing any of them:

```bash
cd python && python export_reference.py     # ~25 s, writes ../matlab/reference.mat
```

## How to run the MATLAB

```matlab
cd matlab
selftest              % verify the port first
run_all               % 'quick' — every table, reduced sweeps
run_all full          % the complete Python sweeps
T = run_all('full');  % also returns the tables as a struct
```

Individual pieces:

```matlab
L = lattice(24, 2, 2.5, 0.010);                    % n, j0, k3, zeta1
[tt, aa, vv, ac] = settle_release(L, 2e-2, 7);     % A, mode
e = estimates(L, aa, vv, ac, 7)                    % r_gcvd, r_env, r_flux, ...
r = energy_rates(L, tt, aa, vv, 7, 5:9)            % carrier +/- 2 subset
f = floquet(5.0e-4)                                % max_multiplier 1.011861 (unstable)
b = boucwen_record(2e-2)                           % the dissipative boundary case
```

### Indexing — the one thing to get right

The Python uses `J0 = 1` and `MODE = 6`, both **zero-based**: the nonlinear/forcing station
is the **second mass** and the tracked resonance is the **seventh mode**. Everything in this
package is **one-based**, so the same configuration is `j0 = 2` and `mode = 7`, and those
are the defaults. `selftest.m` checks `psi == Phi(j0,:).'` at zero tolerance, so an
off-by-one here fails on the first line of section 1 rather than silently shifting the
station.

### Integration

Python integrates with scipy's DOP853 at `rtol` 1e-9 (1e-11 for the Floquet orbit probe);
this package uses `ode89` with `RelTol`/`AbsTol` set to the same values. **DOP853 and ode89
are different implementations of the same order**, so agreement is expected to *solver
tolerance* and not to machine precision. That is why the integrated checks in `selftest.m`
carry looser tolerances than the algebraic ones, and why each of those tolerances is
justified from a measured sensitivity in the file header rather than picked as a round
number.

## What `selftest.m` checks, and what a FAIL means

Checks are tiered so that **the earliest failing section localises the bug**: later sections
are built on earlier ones, so read the output top-down and fix the first failure first.

| § | What | Tolerance | A FAIL here means |
|---|------|-----------|-------------------|
| 1 | chain frequencies, `Phi`, `psi`, uniform `ci` | 1e-12 rel | The eigenproblem, the sign convention, or the 1-based station index is wrong. Nothing downstream can be trusted. |
| 2 | `hilbert_fft` vs scipy, **even and odd** length | 1e-12 rel | The analytic signal is wrong — almost certainly the Nyquist bin or the even/odd branch. Every envelope-based damping estimate in the package is then quietly wrong. |
| 3 | GCVD on an exact 6-mode record; tracking over 3 windows; `realify` | 1e-10 rel | The decomposition. A failure on *`recovers true wn/zeta`* but not on the Python comparison means the reference itself is suspect; a failure only on *3-window* lines isolates the `matchpairs` ↔ `linear_sum_assignment` substitution. |
| 4–5 | `rhs_free`, `rhs_forced`, Bouc-Wen basis and RHS, evaluated pointwise | 1e-12 rel | An equation of motion is mistyped. These are pure arithmetic at a supplied state, so no integrator is involved. |
| 6 | the three estimates on a **supplied** 2000-sample record | 1e-8 rel | The estimator, not the integrator — the record is read from `reference.mat`, so this isolates `estimates.m` from `settle_release.m`. |
| 7 | sensor placement indices and projection conditioning | 0 / 1e-10 rel | The sensor index set differs from Python (see the rounding hazard below) or the pseudoinverse projection is wrong. |
| 8 | settle + release, then the three estimates | 1e-5 | The integration. If §6 passed and §8 fails, the estimator is fine and the discrepancy is in `settle_release.m` or in ode89 itself. |
| 9–10 | share law; the five-subset recovery table; the vacuity identity | 1e-5 / 1e-8 | Energy bookkeeping. The *vacuity* line checks that `gamma_S − d(ln f_S)/dt` is independent of the subset, which is an algebraic identity — a failure there is a real bug, not a tolerance issue. |
| 11 | Floquet multipliers at μ = 3e-4 (stable) and 5e-4 (unstable) | 2e-5 | The variational system or the monodromy assembly. The stable/unstable **verdict** is checked separately at zero tolerance; if the verdict flips, the instability onset — the paper's central mechanism — is not being reproduced. |
| 12 | one Bouc-Wen record | 1e-3 rel | The hysteretic element. Loosest tolerance in the package, because the `abs()` makes the RHS non-differentiable and Runge-Kutta pairs lose their formal order at each kink in an implementation-specific way. A real bug here (wrong sign, wrong β/γ, wrong basis) moves these O(6) rates by O(1), three decades above the tolerance. |
| 13 | a band-forced cell, β = 0.25 | 1e-5 | The multi-harmonic branch of `rhs_forced.m`. Reproducible only because the 25 random phases are exported in `reference.mat`. |

A FAIL is not automatically a bug in this package. Distinguish three causes:

1. **A port error** — by far the most likely, and what the tiering is designed to localise.
2. **A solver difference** — plausible *only* in sections 8–13, only near the tolerance, and
   never in sections 1–7 (which do no integration at all). If an integrated check misses by
   a factor of a few, re-run it with tighter `RelTol` and see whether it converges toward
   the reference; if it does, it is the solver.
3. **A stale reference** — if you edited the Python, re-run `export_reference.py`.

## What was verified without MATLAB

Since the MATLAB could not be executed, two substitutes were used, and it is worth being
precise about what each does and does not cover.

**A line-by-line back-translation** (`python/verification/`). Every `.m` file was
transliterated back into Python with its one-based indexing kept explicit (`rng1(a,b)` is
MATLAB's `a:b`), and the result was run against the same `reference.mat` that `selftest.m`
loads. All 72 back-translation checks pass. This rules out the failure modes that would
otherwise be silent: off-by-one window slices, transposed `psi`, the wrong even/odd branch
in the analytic signal, `ddof` on a standard deviation, column- versus row-major reshaping
of the monodromy, and the MATLAB-vs-numpy rounding rule in the sensor placement. It found
one real defect before shipping (see *Findings*, item 1). **It does not cover MATLAB syntax,
and it does not cover the DOP853 → ode89 substitution** — the back-translation integrates
with scipy, so the integrated checks there agree to ~1e-16 and say nothing about ode89.

**A lexer-based static check** of block balance, bracket balance and `fprintf` format/argument
counts across all 17 `.m` files. Clean. This is not a parser: it will not catch a misspelled
function name, a wrong number of output arguments, or a shape error.

**Two solver-sensitivity measurements**, used to set the tolerances in sections 8–13 rather
than guessing. First, varying DOP853's own `rtol` from 1e-9 to 1e-12 moves the estimates by
≤2.3e-9, the settled state by 9.3e-9 relative and the Floquet multiplier by 9.4e-9 — the
settled orbit is attracting, so it does not amplify integration error. Second, vectorising
the monodromy column-major instead of row-major — the same matrix ODE and the same solver,
with only the 2304 components in a different order, so the step-size controller accumulates
its error norm differently — moves the Floquet multiplier by 5.7e-9 at μ = 3e-4 and 1.6e-9
at μ = 5e-4. Both are measurements of how much *this* integrator moves under perturbations
that leave the mathematics unchanged; **neither bounds the DOP853 → ode89 substitution**,
which is a genuinely different Runge-Kutta pair and can differ by more. The integrated
tolerances were set 2000–4000× above these figures on that reasoning, which still leaves
them three to four decades below the size of any real porting error.

**Runtime estimate, not a measurement.** `selftest` full is estimated at 10–30 minutes and
`run_all full` at one to three hours, dominated by the `t_settle = 6000` forced integrations
at `RelTol` 1e-9 and by the 2304-state monodromy. These are extrapolations from the Python
timings; they were not measured in MATLAB and could be off by a large factor.

## Findings in the Python, noted while porting

1. **Eigenvector signs are never canonicalised** — `lattice_nl.chain_modes` and
   `friction_boundary.system` both return whatever sign `numpy.linalg.eigh` happens to pick,
   and MATLAB's `eig` disagrees on 12 of the 24 chain modes. Every *scalar* result is exactly
   invariant to this (verified: flipping 12 signs changes the three estimates, the energy
   share and the Bouc-Wen rates by precisely 0.0, because the flip is an exact negation in
   IEEE arithmetic), but every *vector* result — `psi`, a state vector, a modal record — is
   basis-dependent and so is not reproducible across LAPACK builds, numpy versions or
   languages. This is worth two lines upstream: canonicalising to first-row-positive
   reproduces the analytic chain modes `sqrt(2/(n+1)) sin(j i π/(n+1))` to 5e-15, and the
   smallest first-row entry is 0.0354, so the sign is never in numerical doubt. This package
   canonicalises in `lattice.m` and `boucwen.m` and in `export_reference.py`. It is the one
   defect the back-translation caught — before the fix, six of the self-test's vector checks
   failed while every scalar passed.

2. **Latent `IndexError` in `gcvd.windowed_gcvd`.** In the first window
   `sel = range(min(n_track, len(cand)))`, so `prev` has `min(n_track, len(cand))` entries;
   but in every later window the cost matrix is built with `for k in range(n_track)` and
   indexes `prev[k]`. If a window ever yields fewer than `n_track` candidate poles, the
   second window raises `IndexError`. No current call can trigger it (all pass `n_track ≪ 2n`
   with full-rank records), but it will fire on a rank-deficient or heavily noise-filtered
   record. `gcvd_windowed.m` deliberately diverges here: it sizes the cost matrix by the
   number of entries actually carried forward, so it degrades instead of erroring. That is
   the only intentional behavioural difference in the package.

3. **`bias_map._three_estimates` hard-codes the module-global `DT`** when it builds its own
   time vector, so it silently assumes any record handed to it was sampled at 0.05. All
   current callers satisfy that; a differently sampled record would return a wrong decay rate
   with no warning. `estimates.m` takes `dt` as an argument (defaulting to 0.05) to make the
   assumption explicit.

4. **Log-argument guards are inconsistent across the scripts.** `noise_sensing._rates`
   protects its fits with `np.maximum(env, 1e-300)` and a `Es > 0` mask; `bias_map`,
   `share_law`, `recovery_recipe` and `bandwidth_axis` do not. Nothing currently reaches a
   non-positive argument (envelopes and energies are positive by construction), but the
   failure mode is *quieter in MATLAB*, where `log` of a negative number silently returns a
   complex value instead of warning. `selftest.m` compares with `abs()` of a complex
   difference, so a complex result shows up as a large FAIL rather than as a plausible number.

5. **numpy's PCG64 cannot be reproduced in base MATLAB.** This affects two experiments.
   `bandwidth_axis.band_force` draws 25 random phases — those are exported into
   `reference.mat`, so that experiment *is* exactly reproducible here. `noise_sensing` draws
   full noise fields on an 8000×24 record and cannot be handled that way, so `run_all`'s
   Table 7 uses MATLAB's Mersenne Twister and is **not** bit-comparable with Python; only its
   ensemble statistics are. The deterministic parts of that experiment — the noise-free row
   and the sensor-placement conditioning — are exact and are what `selftest.m` checks.

6. **`Lattice.decay` recomputes acceleration in a Python loop**, one RHS call per output
   sample. Algebraically identical to the vectorised expression used in `settle_release.m`,
   just `O(m)` interpreter round-trips; a performance note, not a defect.

## Known limitations of this port

- `noise_sensing.py` has no bit-comparable MATLAB counterpart (finding 5); `run_all` Table 7
  reproduces the *experiment*, not the *numbers*.
- The tracking branch of `gcvd_windowed.m` substitutes `matchpairs` for
  `scipy.optimize.linear_sum_assignment`. Both solve the same linear assignment problem and
  the optimum is unique when costs are distinct, but the substitution is checked only by the
  3-window lines in §3 of the self-test — that is the line to read if they disagree.
- `estimates.m` is only exercised on records sampled at `dt = 0.05`.
- Everything in `matlab/` is untested code until `selftest.m` says otherwise.

## Readout change (2026-10-01)

`gcvd.m` previously read the characteristic values off the DIAGONAL of
`Sigma2 = pinv(Xi)*X2*V^-H`. It now takes the EIGENVALUES of `Sigma2`, keeps the
modes `V` as the eigenvectors of `Q` (unrotated), and assigns each eigenvalue
one-to-one to the mode carrying most of its eigenvector's weight
(`matchpairs`, R2019a or later). This matches the Python source that produced
the paper. `reference.mat` / `reference_scalars.csv` were regenerated from the
updated Python sources; only the decomposition damping entries changed
(`est_r_gcvd`, `subest_r_gcvd`, `band_r_gcvd` and the matching `stiff_pct`).
The line-by-line Python mirror of this port passes 72 of 72 checks against
the new references. The GCVD checks in `selftest.m` section 3 use an exactly
linear record, on which the two readouts coincide, so they cannot tell the
readouts apart; the `est_*` checks can.

## Status of `selftest` (2026-10-06)

`selftest.m` has not been run by anyone on this bundle. An attempt from the assistant's sandbox (MATLAB R2026a, `-batch`)
stopped at start-up: MATLAB exits with "Incompatible processor. This Qt build requires the following features: neon" because
the sandbox denies the `sysctl` query that reports the CPU features (the processor is arm64). Nothing of the selftest ran, so
no pass or fail is claimed. The Python mirror of the port (`python/verification/check_mirror.py`) does run; see the line below.
To run the selftest on your machine, in a terminal:

    cd <this folder>/matlab && /Applications/MATLAB_R2026a.app/bin/matlab -batch "selftest quick"

`selftest quick` is the 42 algebraic checks (seconds); drop `quick` for all 76 (estimated 10-30 minutes). A line starting
with FAIL names the check; send the printed output.
