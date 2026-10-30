# M2: phase-resonance Floquet sweep on the port's model

Alt B's scripts `nnm_backbone.py`, `nnm_continuation.py`, `protocols_core.py` and `protocols_phase_resonance.py` are copied unmodified. They run against the **port's** `models/`, `identification/`, `bias_map.py` and `onset_mechanism.py` (copied from `paper-3-damping-identifiability/code/`). Outputs go to `m2/checkpoints/`. Run with the reference env, threads = 1:
`nnm_backbone.py`, then `protocols_phase_resonance.py checkpoints/nnm.json`.
