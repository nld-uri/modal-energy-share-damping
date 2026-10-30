"""Reproduce every number in paper 3, in order, from a cold start.

    python run_paper3.py            run everything (about 17 minutes; timed 2026-10-06,
                                    see checkpoints/timing_run_paper3.json)
    python run_paper3.py --list     show the stages and what each produces
    python run_paper3.py bias_map onset_mechanism     run named stages only

Each stage writes a JSON into ../checkpoints/ and is independently runnable
(`python experiments/<stage>.py`).  Nothing here depends on state left by a
previous stage, so a failed stage can be re-run alone.

Stages, and the paper section each supports:

  bias_map          Sec 4-6   the controlled-release protocol, the similarity
                              collapse, the 25-cell grid and 10-point onset sweep
  era_benchmark     Sec 5-6   the realization algorithm on the same records,
                              below and above onset, with a stabilization table
  onset_mechanism   Sec 6     sideband destinations, growth histories, and the
                              Floquet stability boundary (the expensive stage)
  share_law         Sec 7     the share-drift account and the uniform-damping
                              check that rules out a mixing explanation
  bandwidth_axis    Sec 7.1   96 band-limited records; separates the law from
                              the energy-share proxy
  recovery_recipe   Sec 7.3   which modes must be measured, and the vacuity
                              check on the "correction"
  friction_boundary Sec 8     Bouc-Wen element; where the estimand stops existing
  noise_sensing     Sec 9     SNR, window length and sensor count

WHAT TO CHECK FIRST if you are auditing this: `recovery_recipe.py` prints a
vacuity check whose value must be at machine precision (~1e-15).  It confirms
that gamma_S - dln(f_S)/dt is independent of the modal subset S, which is the
algebraic fact that stops the "correction" reading of Eq. (5) from being
meaningful.  If that number is not tiny, something is wrong with the install.
"""
import argparse
import importlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _preflight():
    """Check numpy/scipy before running anything.

    Without this the ten stages each fail with the same ModuleNotFoundError
    and the real problem -- the interpreter, not the code -- is buried in
    ten tracebacks.
    """
    missing = []
    for pkg in ("numpy", "scipy"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    exe = sys.executable
    print("=" * 70)
    print(f"Missing required package(s): {', '.join(missing)}")
    print("=" * 70)
    print(f"\nThe interpreter running this script is:\n    {exe}\n    "
          f"Python {sys.version.split()[0]}\n")
    print("This study needs only numpy and scipy. Binary wheels exist for\n"
          "CPython 3.12-3.15 on macOS arm64, so nothing has to be compiled.\n")
    print("Recommended -- a dedicated virtual environment, which leaves your\n"
          "system and Homebrew Pythons untouched:\n")
    print("    python3 -m venv ~/venvs/paper3")
    print("    ~/venvs/paper3/bin/pip install -U pip")
    print("    ~/venvs/paper3/bin/pip install numpy scipy")
    print(f"    ~/venvs/paper3/bin/python {os.path.relpath(__file__)}\n")
    print("Or, to install into the interpreter you just used:\n")
    print(f"    {exe} -m pip install numpy scipy\n")
    print("(On a Homebrew Python this may report 'externally-managed-environment'.\n"
          " That is Homebrew protecting its own install -- use the venv above\n"
          " rather than --break-system-packages.)\n")
    print("Then check with:\n")
    print("    <your python> -c \"import numpy, scipy; print(numpy.__version__, "
          "scipy.__version__)\"")
    sys.exit(2)


_preflight()
sys.path.insert(0, os.path.join(HERE, "experiments"))
sys.path.insert(0, os.path.join(HERE, "models"))
sys.path.insert(0, os.path.join(HERE, "identification"))

STAGES = [
    ("bias_map", "Sec 4-6", "bias_map_results.json"),
    ("era_benchmark", "Sec 5-6", "era_benchmark_results.json"),
    ("onset_mechanism", "Sec 6", "onset_mechanism_results.json"),
    ("share_law", "Sec 7", "share_law_results.json"),
    ("bandwidth_axis", "Sec 7.1", "bandwidth_axis_results.json"),
    ("recovery_recipe", "Sec 7.3", "recovery_recipe_results.json"),
    ("nonuniform_damping", "Sec 7.4", "nonuniform_damping_results.json"),
    ("friction_boundary", "Sec 8", "friction_boundary_results.json"),
    ("noise_sensing", "Sec 9", "noise_sensing_results.json"),
    ("second_configuration", "Sec 10", "second_configuration_results.json"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help="stage names; default is all")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args()

    if args.list:
        for name, sec, out in STAGES:
            print(f"  {name:<18} {sec:<9} -> checkpoints/{out}")
        return 0

    todo = args.stages or [s[0] for s in STAGES]
    unknown = [t for t in todo if t not in {s[0] for s in STAGES}]
    if unknown:
        print(f"unknown stage(s): {unknown}\nrun with --list to see valid names")
        return 2

    failures, t_all = [], time.time()
    for name, sec, out in STAGES:
        if name not in todo:
            continue
        print(f"\n{'=' * 70}\n{name}  ({sec})\n{'=' * 70}", flush=True)
        t0 = time.time()
        try:
            mod = importlib.import_module(name)
            mod.main()
            print(f"[{name}] ok in {time.time() - t0:.0f} s", flush=True)
        except Exception as exc:                      # keep going; report at the end
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[{name}] FAILED after {time.time() - t0:.0f} s: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # Both audits run here, not by hand: verify_paper3 recomputes every
    # headline number and requires the manuscript to print it; discipline
    # checks that no result is older than the script that made it, that no
    # manuscript literal is unsourced, and that the threshold is resolved by
    # the procedure that located it. A regenerated pipeline whose prose was
    # not revisited is exactly the state these catch.
    if not failures:
        for audit in ("verify_paper3", "discipline"):
            print(f"\n{'=' * 70}\n{audit}\n{'=' * 70}", flush=True)
            rc = subprocess.call([sys.executable,
                                  os.path.join(HERE, "verification",
                                               f"{audit}.py")])
            if rc:
                failures.append((audit, f"exit {rc}"))

    print(f"\n{'=' * 70}")
    print(f"{len(todo) - len(failures)} of {len(todo)} stages completed "
          f"in {time.time() - t_all:.0f} s")
    for name, err in failures:
        print(f"  FAILED  {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
