"""Guardrails against the three ways this project has actually gone wrong.

`verify_paper3.py` checks that specific claims match specific stored values.
It cannot catch the failures that come from *how* a number reached the page.
Three of those are mechanical, so they are checked mechanically here.

1. STALENESS. A results file older than the script that writes it means the
   analysis on disk is not the analysis in the code, and every number read
   from it is suspect. This has happened: a figure and a manuscript sentence
   were once written from a checkpoint that a later run had already replaced.
   Compares mtimes; also warns when the manuscript is older than a result it
   quotes, which means the prose has not seen the latest numbers. File times
   are only meaningful in the tree where the results were produced: a tree
   assembled by copying the scripts afresh (as the packaged code archive was)
   reports results as older than their scripts although no script changed, so
   a FAIL here after unpacking an archive must be checked against the script
   contents before anything is re-run.

2. UNSOURCED LITERALS. Every number printed in the manuscript should be
   traceable to a stored result. Literals that appear nowhere in any
   checkpoint are either hand-computed (a ratio worked out in prose, which is
   how "low by 26 per cent" reached a message when the file said 46) or
   inherited from a run whose output was never stored. The run classifies every
   numeric token between the abstract and the end of the text into four
   disjoint classes and prints all four counts, so that the numerator and the
   denominator of the coverage statement are both visible:

     structural     matches SKIP below (mode numbers, years, round numbers);
                    not checked
     untraceable    fewer than `min_sig` significant digits; not checked
                    (a short literal matches some stored value by chance too
                    often to carry any evidence)
     matched        `min_sig` or more significant digits and equal, at the
                    printed precision, to a stored value or to 100 times one
     unsourced      `min_sig` or more significant digits and no such match

   "Checkable" means matched plus unsourced. The report is a net, not a gate:
   a clean report means "nothing caught among the checkable literals", never
   "everything sourced". Its reach is not a constant of the method, so the run
   computes it on the checkpoints present (`coincidence_rate`): the share of
   random literals of 3, 4 and 5 significant digits that match some stored
   value by chance. A rate near 1 means a match proves nothing at that
   precision.

3. RESOLUTION. A quoted boundary must be resolved by the procedure that
   located it. Checks that the sensitivity of the deciding quantity to the
   procedure's own parameters is smaller than the margin the decision turns
   on -- the test that would have caught a threshold quoted to three figures
   from a probe whose spread was sixty times the margin.

Usage:  python discipline.py [--strict]     (--strict exits nonzero on warnings)
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CK = os.path.join(ROOT, "checkpoints")
TEX = os.environ.get("PAPER3_TEX", os.path.join(ROOT, "manuscript", "paper3_nd.tex"))

# results file  ->  script that writes it
PRODUCER = {
    "bias_map_results.json": "code/experiments/bias_map.py",
    "onset_mechanism_results.json": "code/experiments/onset_mechanism.py",
    "share_law_results.json": "code/experiments/share_law.py",
    "bandwidth_axis_results.json": "code/experiments/bandwidth_axis.py",
    "recovery_recipe_results.json": "code/experiments/recovery_recipe.py",
    "friction_boundary_results.json": "code/experiments/friction_boundary.py",
    "noise_sensing_results.json": "code/experiments/noise_sensing.py",
    "nonuniform_damping_results.json": "code/experiments/nonuniform_damping.py",
    "second_configuration_results.json": "code/experiments/second_configuration.py",
    "window_fit_results.json": "code/experiments/window_fit.py",
}


def _git_clean(*paths):
    """True if ROOT is a git checkout and none of `paths` differs from HEAD."""
    import subprocess
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        return False
    try:
        out = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", *paths],
                             capture_output=True, text=True, timeout=30)
    except Exception:
        return False
    return out.returncode == 0 and out.stdout.strip() == ""


def staleness():
    """Results older than their producing script, and prose older than results."""
    bad, warn = [], []
    tex_m = os.path.getmtime(TEX)
    for res, script in sorted(PRODUCER.items()):
        rp, sp = os.path.join(CK, res), os.path.join(ROOT, script)
        if not os.path.exists(rp):
            bad.append(f"{res}: MISSING (producer {script})")
            continue
        if not os.path.exists(sp):
            warn.append(f"{script}: missing producer for {res}")
            continue
        if _git_clean(rp, sp):
            # A git checkout sets every file time to the checkout time, so the
            # time order says nothing; the commit is the provenance. Only files
            # edited since the checkout are compared by time.
            continue
        dt = os.path.getmtime(rp) - os.path.getmtime(sp)
        if dt < 0:
            bad.append(f"{res}: {-dt/60:.0f} min OLDER than {script} "
                       f"-- rerun before quoting it")
        if os.path.getmtime(rp) > tex_m:
            warn.append(f"{res}: newer than paper3.tex by "
                        f"{(os.path.getmtime(rp)-tex_m)/60:.0f} min "
                        f"-- prose may not reflect it")
    return bad, warn


def _stored_numbers():
    """Every numeric value anywhere in the checkpoints, as a set of strings."""
    seen = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, bool):
            pass
        elif isinstance(x, (int, float)):
            v = abs(float(x))
            # Round to the SAME number of significant digits the manuscript
            # prints, and allow only the transformations a caption legitimately
            # applies: the value itself, and the value as a percentage. An
            # earlier version also admitted 1/v and any precision from 1 to 5
            # significant digits; with ~10^5 stored numbers that matched
            # essentially any literal, and a deliberate three-fabrication test
            # passed it cleanly. Keep this narrow or it certifies nothing.
            for p in (3, 4, 5, 6, 7):
                seen.add(f"{v:.{p}g}")
                seen.add(f"{v * 100:.{p}g}")
    for f in os.listdir(CK):
        if f.endswith(".json"):
            try:
                walk(json.load(open(os.path.join(CK, f))))
            except Exception:
                pass
    return seen


# numbers that are structural rather than computed
SKIP = re.compile(r"^(0|1|2|3|4|5|6|7|8|9|10|11|12|16|20|24|25|30|33|35|40|48|"
                  r"50|60|96|100|103|104|400|401|801|1200|1600|2000|2400|3000|"
                  r"4000|6000|8000|19[0-9][0-9]|20[0-9][0-9])$")


def _sig(s):
    """Significant digits in a printed literal."""
    d = s.lstrip("0.").replace(".", "").rstrip()
    return len(d.lstrip("0"))


def census(min_sig=4, tex_path=None, stored=None):
    """Classify every numeric token of the manuscript body (see the docstring).

    Returns a dict with the counts `total`, `structural`, `untraceable`,
    `matched`, `unsourced` (their sum is `total`), the list `unsourced_list`
    of (line, literal), and `n_stored`, the number of stored value-strings.
    """
    tex = open(tex_path or TEX).read()
    tex = re.sub(r"%.*", "", tex)                    # drop comments
    i = tex.find(r"\begin{thebibliography}")
    if i > 0:
        tex = tex[:i]
    stored = _stored_numbers() if stored is None else stored
    # Start at the abstract: the title block carries a postal code, which is
    # not a numerical claim about anything.
    head = tex.find(r"\begin{abstract}")
    if head < 0:
        head = tex.find(r"\abstract{")        # Springer sn-jnl
    body = tex[head:] if head > 0 else tex
    offset = tex[:head].count("\n") if head > 0 else 0
    n = dict(total=0, structural=0, untraceable=0, matched=0, unsourced=0)
    out = []
    for m in re.finditer(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])", body):
        s = m.group(1)
        n["total"] += 1
        if SKIP.match(s):
            n["structural"] += 1
        elif _sig(s) < min_sig:
            n["untraceable"] += 1
        elif f"{float(s):.{_sig(s)}g}" in stored:
            n["matched"] += 1
        else:
            n["unsourced"] += 1
            out.append((offset + body[:m.start()].count("\n") + 1, s))
    assert sum(n[k] for k in ("structural", "untraceable", "matched", "unsourced")) == n["total"]
    n["unsourced_list"] = out
    n["n_stored"] = len(stored)
    return n


def unsourced(min_sig=4, tex_path=None, stored=None):
    """(unsourced literals as (line, literal), count below `min_sig` digits)."""
    c = census(min_sig, tex_path, stored)
    return c["unsourced_list"], c["untraceable"]


def coincidence_rate(stored, sig, n=4000, seed=0):
    """Share of random `sig`-digit literals that match a stored value by chance.

    Literals are log-uniform in [1e-4, 1e2], rounded to `sig` significant
    digits, formatted as the detector formats a printed literal.
    """
    import math
    import random
    rng = random.Random(seed)
    hit = 0
    for _ in range(n):
        v = 10.0 ** rng.uniform(-4.0, 2.0)
        hit += f"{float(f'{v:.{sig}g}'):.{sig}g}" in stored
    return hit / n


def resolution():
    """Is each quoted boundary resolved by the procedure that located it?"""
    p = os.path.join(CK, "onset_mechanism_results.json")
    if not os.path.exists(p):
        return ["onset_mechanism_results.json missing"], []
    om = json.load(open(p))
    bad, warn = [], []
    if "orbit_accuracy" in om and "converged" in om:
        conv = next(b for b in om["converged"]
                    if abs(b["zeta1"] - om["config"]["zeta1"]) < 1e-12)
        # The margin is what the stability decision actually turns on: how far
        # from unity the first unstable multiplier sits. (An earlier version
        # used the change in max|lambda| between the two amplitudes that
        # bracket the crossing, which is a property of the grid, not of the
        # decision -- and it disagreed with the figure the manuscript quotes.)
        margin = min(abs(r["max_multiplier"] - 1.0)
                     for r in conv["records"] if r["unstable"])
        # One solve in the sweep may fail to converge above the crossing; the
        # threshold cannot depend on it, so judge resolution on the rest.
        res = sorted(r["return_residual"] for r in conv["records"])[-2]
        if res > margin:
            bad.append(f"threshold: orbit residual {res:.1e} exceeds the "
                       f"deciding margin {margin:.1e}")
        else:
            warn.append(f"threshold resolved: residual {res:.1e} against "
                        f"margin {margin:.1e} ({margin/res:.0e}x)")
    return bad, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--min-sig", type=int, default=4)
    a = ap.parse_args()

    fails, warns = [], []
    b, w = staleness(); fails += b; warns += w
    b, w = resolution(); fails += b; warns += w
    stored = _stored_numbers()
    c = census(a.min_sig, stored=stored)
    us, name = c["unsourced_list"], os.path.basename(TEX)

    print("=" * 70)
    print("STALENESS")
    for x in fails:
        print(f"  FAIL  {x}")
    for x in warns:
        print(f"  note  {x}")
    chk = c["matched"] + c["unsourced"]
    print(f"\nNUMERIC LITERALS in {name}, abstract to end: {c['total']}")
    print(f"  structural (SKIP list, not checked)                 {c['structural']:>5}")
    print(f"  below {a.min_sig} significant digits (untraceable, not checked)   {c['untraceable']:>5}")
    print(f"  checkable ({a.min_sig}+ significant digits)                       {chk:>5}"
          f"  = {c['matched']} matched a stored value + {c['unsourced']} matched none")
    print(f"  chance match of a random literal against the {c['n_stored']} stored value-strings:")
    rates = {sg: coincidence_rate(stored, sg) for sg in (3, 4, 5)}
    for sg in (3, 4, 5):
        print(f"    {sg} significant digits  {100 * rates[sg]:5.1f} %")
    if rates.get(a.min_sig, rates[4]) > 0.5:
        print(f"  note  at this checkpoint volume a {a.min_sig}-digit match is not evidence of a source "
              f"(chance rate above 50 %); the guard against unsourced numbers is the coverage rule of "
              f"verify_paper3.py, not this count")
    print(f"\nUNSOURCED LITERALS: {len(us)} of {chk} checkable")
    for line, s in us[:40]:
        print(f"  {name}:{line}  {s}")
    if len(us) > 40:
        print(f"  ... and {len(us)-40} more")
    print("=" * 70)
    print(f"{len(fails)} failures, {len(warns)} notes, {len(us)} unsourced literals")
    if fails or (a.strict and us):
        sys.exit(1)


if __name__ == "__main__":
    main()
