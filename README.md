# Modal energy share and sideband instability: code and results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23818514.svg)](https://doi.org/10.5281/zenodo.23818514)

Code, stored results and verification scripts for

> D. Chelidze, *Modal energy share and sideband instability: when is an
> identified damping value meaningful?* (submitted to *Nonlinear
> Dynamics*, 30 October 2026; under review).

Every number, table and figure in the paper is produced by this code from a
simulated model. There are no measured data. The model is a fixed–fixed chain
of 24 unit masses carrying one grounded cubic spring, with uniform modal
damping. The decays are identified output-only with the canonical
characteristic value decomposition (NCVD), and the eigensystem realization
algorithm (ERA) serves as an independent check.

## Quick start

You need Python 3.10 or later, NumPy and SciPy. Regenerating the figures also
needs Matplotlib and a LaTeX installation (the figure text is typeset by LaTeX):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

There are three levels of reproduction, from seconds to about 20 minutes.

| level | command (from the repository root) | time | what it shows |
|---|---|---|---|
| 1. Figures | `.venv/bin/python code/figures/make_figures.py` | about a minute | the figures, drawn into `figures/` from the stored results in `checkpoints/`, with nothing recomputed |
| 2. Text against results | `.venv/bin/python code/verification/verify_paper3.py --tex PATH/TO/paper.tex` | seconds | recomputes 161 quoted quantities from the stored results and checks the manuscript prints each one |
| 3. Everything | `.venv/bin/python code/run_paper3.py` | about 17 min on 10 cores | re-integrates every stage and rewrites `checkpoints/` |

The manuscript source is not part of this archive. Level 2 needs the LaTeX
source of the paper, given with `--tex` or the `PAPER3_TEX` environment
variable. The record of the last full verification is
`code/verification/results/`.

`python code/run_paper3.py --list` lists the stages and the section each one
supports. Each stage also runs alone, e.g. `python code/run_paper3.py share_law`.

## Layout

| path | contents |
|---|---|
| `code/models/lattice_nl.py` | the 24-mass chain and its nonlinear station |
| `code/identification/` | `ncvd.py` (the canonical decomposition), `gcvd.py` (an adapter that exposes it to the scripts), `era.py` (the independent realization check) |
| `code/experiments/` | one script per stage; each writes one JSON into `checkpoints/` |
| `code/figures/` | one script per figure; each reads stored JSONs only |
| `code/verification/` | `verify_paper3.py` (161 checks), `verify_m_track.py` (88 checks on the pre-declared analyses), `discipline.py` (provenance audit), `make_cell_table.py` |
| `checkpoints/` | the stored results the paper quotes, one JSON per stage |
| `m_track/` | the pre-declared analyses M1–M7 and the growth budget M4, with their pre-declarations (see below) |
| `matlab/` | a toolbox-free MATLAB port with `selftest.m`. **Never executed by the author; see `README_matlab.md`** |

## The pre-declared analyses (`m_track/`)

Several sections rest on analyses whose acceptance criteria were written and
hashed before they ran. The pre-declarations and their SHA-256 records are in
`m_track/`. Check them with

```bash
cd m_track && for h in *.sha256; do grep -E '^[0-9a-f]{64}  ' "$h" | shasum -a 256 -c -; done
```

`m_track/RESULTS.md` lists every criterion, including those that failed, and
the manuscript reports the failures. The run scripts (`run_m1.py`,
`run_m3.py`, `run_m3b.py`, `run_m5.py`, `run_m6.py`, `run_m7_summary.py`)
write into `checkpoints/m_track/`. `m2/` and `m7/` are self-contained trees
for the phase-resonance sweep and the further configurations; their READMEs
give the run order. `m4/` reproduces the growth budget of the seeded
preparations:

```bash
cd m_track/m4/code/experiments
python protocols_release_records.py && python protocols_release_records_v2.py && python protocols_growth_budget.py
```

This takes about a minute and writes `m_track/m4/checkpoints/protocols_growth_budget.json`.
It equals the stored `checkpoints/m_track/m4_growth_budget.json` bit for bit.

Several scripts in `m2/`, `m4/` and `m7/` were written for an independent
re-analysis of the same model, and are used here against this paper's own
model and identification modules, as `m_track/MAPPING.md` records.

## Environment and reproducibility

The stored results were produced with Python 3.11.16, NumPy 2.4.6 and
SciPy 1.17.1 on macOS (arm64). The results are not version-sensitive, but
eigenvalue solvers can differ in the last digits between BLAS builds. For a
bit-for-bit comparison, run single-threaded:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

`verify_paper3.py` compares printed values with the precision the manuscript
prints. It does not compare files byte for byte.

## Related software

The identification is the NCVD toolbox (D. Chelidze, H. Li and D. L. Stein,
*NCVD toolbox: Nonlinear Characteristic Value Decomposition for Python and
MATLAB*, version 1.0.0, Zenodo, 2026,
[doi:10.5281/zenodo.23549069](https://doi.org/10.5281/zenodo.23549069)).
`code/identification/ncvd.py` is a copy of its decomposition, included so that
this archive runs on its own.

## Licence

Code: MIT (`LICENSE`). Stored results in `checkpoints/` and `m_track/`:
CC BY 4.0 (`LICENSE-DATA`).

## Citation

Please cite the paper and this archive:

> D. Chelidze, *Modal energy share and sideband instability: code and stored
> results*, version 1.0.0, Zenodo, 2026,
> [doi:10.5281/zenodo.23818514](https://doi.org/10.5281/zenodo.23818514).

Machine-readable metadata: `CITATION.cff`.

## Contact

David Chelidze, Nonlinear Dynamics Laboratory, University of Rhode Island
(chelidze@uri.edu).
