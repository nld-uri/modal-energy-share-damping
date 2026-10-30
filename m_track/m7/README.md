# M7: further station/mode configurations on the port's model

Alt B's `theory_second_config_screen.py`, `theory_second_config.py`, `theory_averaged_threshold.py`, `theory_floquet_runs.py` and `protocols_core.py` are copied. They run against the port's `models/`, `identification/`, `bias_map.py` and `onset_mechanism.py`. One edit: the pool in `theory_second_config.py` goes from 10 to 4 workers. The screen re-run here is byte-identical to Alt B's stored screen, so its configuration choice (primary j0 = 4 / mode 6; secondary j0 = 9 / mode 7, 0-based station) is reproduced, not copied.
