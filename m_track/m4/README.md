# M4: growth budget of the seeded preparations

The scripts in `code/experiments/` (`protocols_release_records.py`,
`protocols_release_records_v2.py`, `protocols_growth_budget.py`,
`protocols_core.py`) come from an independent re-analysis of the same model.
Here they run against this paper's own `lattice_nl.py`, `bias_map.py`,
`onset_mechanism.py` and identification modules, copied into `code/`. The
inputs in `checkpoints/` are the paper's onset results and the phase-resonance
family of M2.

Run, single-threaded, from `code/experiments/`:

    python protocols_release_records.py
    python protocols_release_records_v2.py
    python protocols_growth_budget.py

The output `checkpoints/protocols_growth_budget.json` equals the stored
`../../checkpoints/m_track/m4_growth_budget.json` bit for bit (checked on
30 Oct 2026; about one minute on 10 cores).
