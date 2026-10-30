"""Check every headline number in paper3.tex against the saved result JSONs.

Motivated by experience: over this study's drafting, five separate numerical
claims in the prose or the manuscript were found to disagree with the artifacts
they came from -- a maximum quoted from the wrong array, a ratio stated as 4x
when it was 5.2x, a table row silently dropped, a bias described as monotone
when it is not, and a round-up of 2.90 decades to "three". Every one was a
transcription or summarisation error, not a computational one. This script
removes hand transcription from the loop: it recomputes each quantity from the
JSON and checks that the value PRINTED IN THE MANUSCRIPT matches.

    python verify_paper3.py            # check, print a ledger, exit 1 on failure
    python verify_paper3.py --json      # also write verification_paper3.json

Each check states what it verifies, recomputes it, and searches paper3.tex for
the printed form. A check FAILS if the recomputed value disagrees, or if the
string is absent from the manuscript (which catches a number that was silently
edited away or never written).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CK = os.path.join(ROOT, "checkpoints")
# The manuscript is not part of this archive. Pass its LaTeX source with --tex,
# or set PAPER3_TEX; a file manuscript/paper3_nd.tex in the repository root is used if present.
TEX = os.environ.get("PAPER3_TEX", os.path.join(ROOT, "manuscript", "paper3_nd.tex"))


TOL_REL = 0.02      # a printed value may be rounded, not wrong


def load(name):
    with open(os.path.join(CK, f"{name}_results.json")) as fh:
        return json.load(fh)


def _numbers(s):
    """Every number in a string, with LaTeX scientific notation understood.

    `$8.2\\times10^{-3}$` reads as 0.0082, `$10^{-1}$` as 0.1, and a bare
    `$354$` as 354, so that a computed float can be compared against what the
    manuscript actually prints rather than merely found inside it.
    """
    t = re.sub(r"([0-9.]+)\s*\\times\s*10\^\{?(-?[0-9]+)\}?", r"\1e\2", s)
    t = re.sub(r"(?<![0-9.])10\^\{?(-?[0-9]+)\}?", r"1e\1", t)
    t = t.replace("\\,", "").replace("{", " ").replace("}", " ")
    t = re.sub(r"(?<=\d)\s*-{1,2}\s*(?=\d)", " ", t)   # 0.64--0.89 is a range
    out = []
    for m in re.finditer(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", t):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def _decimals(s):
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tex", default=TEX, help="LaTeX source of the manuscript to check against")
    args = ap.parse_args()

    if not os.path.exists(args.tex):
        sys.exit("manuscript source not found: %s\nPass it with --tex PATH (see README)." % args.tex)
    tex = open(args.tex).read()
    # Compare against a whitespace-collapsed copy. A caption rewrap is not a
    # numerical change, and an audit that fails on one trains its reader to
    # ignore failures.
    flat = re.sub(r"\s+", " ", tex)
    bm, om = load("bias_map"), load("onset_mechanism")
    sl, ba = load("share_law"), load("bandwidth_axis")
    rr, fb = load("recovery_recipe"), load("friction_boundary")
    ns = load("noise_sensing")
    sc2 = load("second_configuration")
    nu = load("nonuniform_damping")
    eb = load("era_benchmark")
    rows = []

    def check(label, computed, printed, source, numeric=True):
        """printed: the string the manuscript should contain, compared with
        runs of whitespace collapsed so that line breaks do not matter.

        Presence is necessary but not sufficient. Every number in `computed`
        must also appear among the numbers in `printed`, to within a relative
        tolerance. Testing presence alone let a sentence quoting one statistic
        pass a check computing a different one -- the manuscript said 0.44 per
        cent, meaning |mean - 1|, while the check computed mean|. - 1| = 0.88
        per cent, and the check passed because the string 0.44 was in the file.
        """
        ok = re.sub(r"\s+", " ", printed) in flat
        want, have = _numbers(str(computed)), _numbers(printed)
        if ok and numeric and want:
            for v in want:
                tol = max(TOL_REL * abs(v), 10 ** -(_decimals(str(computed)) + 0.3))
                if not any(abs(v - u) <= tol for u in have):
                    ok = False
                    break
        rows.append(dict(label=label, computed=computed, printed=printed,
                         source=source, pass_=bool(ok)))

    # --- the readout itself, re-run on the same records (readout_final_check) --
    with open(os.path.join(CK, "readout_final_check.json")) as fh:
        rf = json.load(fh)
    check("diagonal readout, worst error on the single-mode records (pct)",
          f"{100 * rf['diagonal_single_mode_worst']:.0f}",
          "the same records give errors up to $128\\,\\%$", "readout_final_check")
    check("eigenvalue readout, worst error on the single-mode records (pct)",
          f"{100 * rf['final_single_mode_worst']:.2f}", "worst case of\n$0.35\\,\\%$",
          "readout_final_check")
    dmax = max(abs(a["r_gcvd"] - b["r_gcvd"]) for a, b in zip(rf["rotated"], rf["final"]))
    check("shape convention leaves tracked damping unchanged", f"{dmax:.1e}",
          "identical under either convention", "readout_final_check", numeric=False)
    rows[-1]["pass_"] = rows[-1]["pass_"] and dmax < 1e-12

    # --- Sec 5-6: the realization algorithm, below and above onset (era_benchmark)
    # Recomputed from records regenerated by era_benchmark.py; its r_gcvd and
    # r_env must equal bias_map's for the same cells, or the two files describe
    # different records.
    er = eb["records"]
    bkey = {(g, round(c["k3"], 9), round(c["A"], 9)): c
            for g in ("grid", "threshold_sweep") for c in bm[g]}
    dsame = max(max(abs(r["r_gcvd"] - bkey[(r["group"], round(r["k3"], 9), round(r["A"], 9))]["r_gcvd"]),
                    abs(r["r_env"] - bkey[(r["group"], round(r["k3"], 9), round(r["A"], 9))]["r_env"]))
                for r in er)
    check("ERA records are bias_map's records", f"{dsame:.1e}",
          "given to an independent realization", "era_benchmark", numeric=False)
    rows[-1]["pass_"] = rows[-1]["pass_"] and dsame < 1e-12
    eg = [r for r in er if r["group"] == "grid" and r["e_frac"] > 0.99]
    e_era = np.array([abs(r["r_era"] - 1) for r in eg])
    e_gc = np.array([abs(r["r_gcvd"] - 1) for r in eg])
    check("ERA single-mode grid cell count", len(eg), "over the $23$ single-mode cells",
          "era_benchmark")
    check("ERA median error (pct)", f"{100 * np.median(e_era):.3f}",
          "returns a median error of $0.005\\,\\%$", "era_benchmark")
    check("ERA worst error (pct)", f"{100 * e_era.max():.2f}",
          "and a worst case of $0.35\\,\\%$ on them", "era_benchmark")
    check("ERA vs GCVD, max error difference", f"{np.abs(e_era - e_gc).max():.1e}",
          "agree cell by cell to $9.1\\times10^{-4}$ in error", "era_benchmark")
    check("ERA vs GCVD error correlation", f"{np.corrcoef(e_era, e_gc)[0, 1]:.3f}",
          "with a correlation of $0.963$", "era_benchmark")
    check("ERA on the linear column", f"{max(abs(r['r_era'] - 1) for r in eg if r['k3'] == 0):.1e}",
          "the realization algorithm to $2.9\\times10^{-6}$", "era_benchmark")
    st = lambda r, k: r["stabilization"][k]["r"]
    low = [f"{o}_200" for o in (6, 7, 8, 9)]
    high = [f"{o}_200" for o in (10, 12, 14, 16)]
    check("ERA low-order (6-9) worst error (pct)",
          f"{100 * max(abs(st(r, k) - 1) for r in eg for k in low):.2f}",
          "at orders $6$ to $9$ the worst error on the same cells is $0.54\\,\\%$",
          "era_benchmark")
    nl = sorted([r for r in eg if r["k3"] > 0], key=lambda r: r["mu"])
    two = [r["mu"] for r in nl if r["o12_significant"] >= 2]
    check("order 12: second significant pole from mu", f"{min(two):.1e}",
          "every cell from $\\mu=1.8\\times10^{-5}$ up", "era_benchmark")
    rows[-1]["pass_"] = rows[-1]["pass_"] and all(r["o12_significant"] >= 2 for r in nl if r["mu"] >= min(two))
    c4 = [r for r in nl if abs(r["mu"] - 4e-4) < 1e-12][0]
    check("order-12 pole pair at mu 4e-4", " ".join(f"{w:.4f}" for w in c4["o12_poles_near"]),
          "$0.8524$ and $0.8553$ at $\\mu=4\\times10^{-4}$", "era_benchmark")
    dep = [r["mu"] for r in nl if min(st(r, k) for k in high) < 0.99]
    check("orders 10-16 depart from mu", f"{min(dep):.0e}",
          "cell from $\\mu=9\\times10^{-6}$ up", "era_benchmark")
    rows[-1]["pass_"] = rows[-1]["pass_"] and dep == [r["mu"] for r in nl if r["mu"] >= min(dep)]
    check("orders 10-16 lowest value", f"{min(st(r, k) for r in nl for k in high):.2f}",
          "falling with $\\mu$ to $0.46$", "era_benchmark")
    allk = list(eg[0]["stabilization"])
    check("linear column, all sixteen settings",
          f"{max(abs(st(r, k) - 1) for r in eg if r['k3'] == 0 for k in allk):.1e}",
          "all sixteen settings return the true value to $2.9\\times10^{-5}$", "era_benchmark")
    rows[-1]["pass_"] = rows[-1]["pass_"] and len(allk) == 16
    check("300-row Hankel, cell nearest onset", f"{st(c4, '8_300'):.3f}",
          "moves the cell nearest the onset to $1.041$", "era_benchmark")
    rows[-1]["pass_"] = rows[-1]["pass_"] and max(eg, key=lambda r: abs(st(r, "8_300") - 1))["mu"] == 4e-4
    ab = sorted([r for r in er if r["group"] == "threshold_sweep" and r["e_frac"] <= 0.99],
                key=lambda r: r["mu"])
    band = [r for r in ab if r["mu"] <= 1.6e-3]
    span = [max(st(r, k) for k in allk) - min(st(r, k) for k in allk) for r in band]
    check("above onset: smallest spread across settings", f"{min(span):.2f}",
          "spans more than $2.3$ at every $\\mu$ from $5\\times10^{-4}$ to $1.6\\times10^{-3}$",
          "era_benchmark", numeric=False)
    rows[-1]["pass_"] = rows[-1]["pass_"] and min(span) > 2.3 and band[0]["mu"] == 5e-4
    r8 = [r for r in ab if abs(r["mu"] - 8e-4) < 1e-12][0]
    check("above onset: range at mu 8e-4",
          f"{min(st(r8, k) for k in allk):.2f} {max(st(r8, k) for k in allk):.2f}",
          "from $-0.58$ to $2.93$ of the true value", "era_benchmark")
    r5, r16, r20 = (next(r for r in ab if abs(r["mu"] - m) < 1e-12) for m in (5e-4, 1.6e-3, 2e-3))
    check("mu 5e-4: low-order ERA, GCVD, envelope",
          f"{min(st(r5, k) for k in low):.3f} {max(st(r5, k) for k in low):.3f} "
          f"{r5['r_gcvd']:.3f} {r5['r_env']:.3f}",
          "$0.997$ to $1.000$ at $\\mu=5\\times10^{-4}$ against the decomposition's $0.994$ and the envelope's $0.947$",
          "era_benchmark")
    check("mu 1.6e-3: low-order ERA, GCVD, envelope",
          f"{min(st(r16, k) for k in low):.3f} {max(st(r16, k) for k in low):.3f} "
          f"{r16['r_gcvd']:.3f} {r16['r_env']:.3f}",
          "where it reads $0.494$ to $0.517$ against $0.866$ and $0.451$", "era_benchmark")
    check("mu 2e-3: stable across all settings",
          f"{min(st(r20, k) for k in allk):.2f} {max(st(r20, k) for k in allk):.2f}",
          "settings ($1.66$ to $1.96$)", "era_benchmark")
    rows[-1]["pass_"] = rows[-1]["pass_"] and min(r20["r_env"], r20["r_gcvd"], min(st(r20, k) for k in allk)) > 1
    check("ERA readout: dominant and nearest selection agree at order 8",
          f"{max(abs(r['r_era'] - r['r_era_nearest']) for r in er):.1e}",
          "These figures are for a realization of order $8$", "era_benchmark", numeric=False)
    rows[-1]["pass_"] = rows[-1]["pass_"] and max(abs(r["r_era"] - r["r_era_nearest"]) for r in er) < 1e-12

    # --- Sec 5-6: single-mode recovery and the decomposition artefact ---------
    sm = [c for c in bm["grid"] + bm["threshold_sweep"] if c["e_frac"] > 0.99]
    env = np.array([c["r_env"] for c in sm])
    gc = np.array([c["r_gcvd"] for c in sm])
    mus = np.array([c["k3"] * c["A"] ** 2 for c in sm])
    nz = mus[mus > 0]
    check("single-mode envelope lower bound", f"{env.min():.4f}",
          f"[0.9952,\\ 1.0006]", "bias_map")
    check("single-mode cell count", len(sm), "$25$ single-mode cells", "bias_map")
    check("max GCVD error (pct)", f"{np.max(np.abs(gc - 1)) * 100:.2f}",
          "worst case of\n$0.35\\,\\%$", "bias_map")
    check("GCVD upper bound", f"{gc.max():.4f}", "$[1.0000,\\ 1.0035]$", "bias_map")
    check("GCVD vs envelope, below onset",
          f"{max(abs(c['r_gcvd'] - c['r_env']) for c in sm):.1e}",
          "$8.2\\times10^{-3}$", "bias_map")
    check("mu span (fold)", f"{nz.max() / nz.min():.0f}", "$800$-fold", "bias_map")
    check("k3 upper", f"{max(c['k3'] for c in sm):.0f}", "$k_{3}$ from $0$ to $4$",
          "bias_map")

    # --- Sec 6: sideband destinations and Floquet ----------------------------
    d5 = {round(d["mu"], 10): d for d in om["destinations"]}
    for mu in (4.0e-4, 1.6e-3):
        sh = {t["mode"]: t["share"] for t in d5[round(mu, 10)]["top"]}
        check(f"carrier share at mu={mu:.1e}", f"{sh[7]:.5f}",
              f"{sh[7]:.5f}", "onset_mechanism")
    sb = {p["k"]: p for p in om["spectrum"]["sideband_pairs"]}
    check("k=1 detuning", f"{sb[1]['detuning']:.4f}", "$-0.0020$", "onset_mechanism")
    check("k=2 detuning", f"{sb[2]['detuning']:.4f}", "$-0.0079$", "onset_mechanism")
    check("third harmonic", f"{om['spectrum']['third_harmonic']:.3f}",
          "$3\\omega_{7}=2.555$", "onset_mechanism")
    check("band edge", f"{om['spectrum']['band_edge']:.3f}",
          "$\\omega_{24}=1.996$", "onset_mechanism")
    _nom = next(c for c in om["converged"]
                if abs(c["zeta1"] - om["config"]["zeta1"]) < 1e-12)
    ft = {round(f["mu"], 10): f for f in _nom["records"]}
    for mu in (3.0e-4, 5.0e-4):
        v = ft[round(mu, 10)]["max_multiplier"]
        check(f"Floquet multiplier at mu={mu:.1e}", f"{v:.6f}", f"${v:.6f}$",
              "onset_mechanism")
    # The bracket comes from the CONVERGED orbits: a probe that has not settled
    # misclassifies amplitudes near the crossing, which is the whole point of
    # the orbit-accuracy study below.
    st = [f for f in _nom["records"] if not f["unstable"]]
    un = [f for f in _nom["records"] if f["unstable"]]
    b_lo = max(f["k3_wpeak2"] for f in st)
    b_hi = min(f["k3_wpeak2"] for f in un)
    check("Floquet boundary bracket (k3 wpeak^2)", f"{b_lo:.5f} {b_hi:.5f}",
          f"crossing between ${b_lo:.5f}$ and ${b_hi:.5f}$", "onset_mechanism")
    sh_ = om["floquet_single_harmonic"]
    s_lo = max(f["k3_wpeak2"] for f in sh_ if f["max_multiplier"] <= 1.0)
    s_hi = min(f["k3_wpeak2"] for f in sh_ if f["max_multiplier"] > 1.0)
    check("single-harmonic crossing bracket", f"{s_lo:.4f} {s_hi:.4f}",
          f"between ${s_lo:.4f}$ and ${s_hi:.4f}$", "onset_mechanism")
    # the brackets must actually overlap, which is what the text claims
    check("brackets overlap", f"{max(b_lo, s_lo) <= min(b_hi, s_hi)}",
          "the two brackets overlap", "onset_mechanism", numeric=False)
    rows[-1]["pass_"] = (rows[-1]["pass_"] and max(b_lo, s_lo) <= min(b_hi, s_hi))
    cf = [f["crest_factor"] for f in _nom["records"]]
    check("crest factor range", f"{min(cf):.4f} {max(cf):.4f}",
          f"stays between ${min(cf):.4f}$ and ${max(cf):.4f}$", "onset_mechanism")

    # --- Sec 7: the drift account -------------------------------------------
    check("share-law mean abs err", f"{sl['derived_law']['mean_abs_err']:.4f}",
          "$0.0057$", "share_law")
    check("share-law worst", f"{sl['derived_law']['worst_abs_err']:.4f}",
          "$0.0092$", "share_law")
    check("share proxy mean abs err", f"{sl['share_proxy']['mean_abs_err']:.4f}",
          "$0.0359$", "share_law")
    # The printed claim is the VALUE, so compare the value. Its deviation from
    # unity is guarded separately, because this equality is exact by
    # construction under uniform damping and a drift away from it would mean
    # the damping vector had been changed without the text being revisited.
    cb_max = max(r["cbar_over_ctrue"] for r in sl["records"])
    cb_min = min(r["cbar_over_ctrue"] for r in sl["records"])
    check("cbar over ctrue is unity", f"{cb_max:.6f}",
          "$\\bar c/\\ctr = 1.000000$", "share_law")
    # Unity holds identically under uniform damping; a different BLAS can leave
    # it one or two units in the last place away from 1.0 (seen: 1.0000000000000002).
    rows[-1]["pass_"] = (rows[-1]["pass_"] and abs(cb_max - 1.0) <= 1e-12 and abs(cb_min - 1.0) <= 1e-12)

    # --- Sec 6: threshold against damping, and the selection eigenvector ----
    # Everything here is sourced from the CONVERGED (Newton-shot) sweep; the
    # probed sweep survives in the JSON only to support the paragraph that
    # explains why it was dropped (orbit-accuracy checks below).
    #
    # The `printed` key is FORMATTED FROM THE COMPUTED VALUE plus fixed context
    # words, never typed. Typing it is the transcription step this script
    # exists to remove: a hand-typed key drifts silently when the data moves,
    # and then agrees with nothing. Built this way, a check fails the moment
    # the manuscript sentence stops matching the file.
    cz = om["converged"]
    zz = np.array([c["zeta1"] for c in cz])
    mc = np.array([c["threshold_mu"] for c in cz])
    for c, m in zip(cz, mc):
        check(f"converged threshold at zeta1={c['zeta1']:.3f}",
              f"{m:.2e}", f"{m * 1e4:.2f}\\times10^{{-4}}", "onset_mechanism")
    ratio = mc[-1] / mc[0]
    check("threshold range over 4x damping", f"{ratio:.2f}",
          f"moves the threshold by a factor of ${ratio:.2f}$", "onset_mechanism")
    expo = np.polyfit(np.log(zz), np.log(mc), 1)[0]
    check("fitted exponent in zeta", f"{expo:.3f}",
          f"a fitted exponent of ${expo:.3f}$", "onset_mechanism")
    loc = np.diff(np.log(mc)) / np.diff(np.log(zz))
    check("local exponent, lower interval", f"{loc[0]:.3f}",
          f"the local exponent is ${loc[0]:.3f}$ between", "onset_mechanism")
    check("local exponent, upper interval", f"{loc[1]:.3f}",
          f"${loc[1]:.3f}$ between the second two", "onset_mechanism")

    nom = next(c for c in cz if abs(c["zeta1"] - om["config"]["zeta1"]) < 1e-12)
    unst = [r for r in nom["records"] if r["unstable"]]
    got = {tuple(sorted(d["mode"] for d in r["content"]["content"][:2]))
           for r in unst}
    check("unstable eigenvector pair", f"{sorted(got)[0]}", "on mode $9$ and",
          "onset_mechanism", numeric=False)
    rows[-1]["pass_"] = (got == {(5, 9)} and rows[-1]["pass_"])
    sh = {}
    for r in unst:
        for d in r["content"]["content"]:
            sh.setdefault(d["mode"], []).append(d["share"])
    check("mode 9 share range", f"{min(sh[9]):.3f} {max(sh[9]):.3f}",
          f"${min(sh[9]):.3f}$ to ${max(sh[9]):.3f}$ on mode $9$", "onset_mechanism")
    check("mode 5 share range", f"{min(sh[5]):.3f} {max(sh[5]):.3f}",
          f"${min(sh[5]):.3f}$ to ${max(sh[5]):.3f}$ on mode $5$", "onset_mechanism")
    pair = [a + b for a, b in zip(sorted(sh[5]), sorted(sh[9], reverse=True))]
    check("pair total range", f"{min(pair):.3f} {max(pair):.3f}",
          f"accounts for ${min(pair):.3f}$ to ${max(pair):.3f}$ of the eigenvector",
          "onset_mechanism")
    pr = [r["content"]["participation"] for r in unst]
    check("participation ratio range", f"{min(pr):.3f} {max(pr):.3f}",
          f"participation ratio of ${min(pr):.3f}$ to ${max(pr):.3f}$", "onset_mechanism")
    third = max((max(v), k) for k, v in sh.items() if k not in (5, 9))
    check("largest third-mode share", f"{third[0]:.4f}",
          f"any third mode is ${third[0]:.4f}$, on mode ${third[1]}$", "onset_mechanism")
    sh_all = {}
    for c in cz:
        for r in c["records"]:
            if r["unstable"]:
                for d in r["content"]["content"]:
                    sh_all.setdefault(d["mode"], []).append(d["share"])
    check("mode 9 share across all damping",
          f"{min(sh_all[9]):.3f} {max(sh_all[9]):.3f}",
          f"mode $9$ holds ${min(sh_all[9]):.3f}$ to ${max(sh_all[9]):.3f}$ at all three",
          "onset_mechanism")
    # below the onset the leading direction wanders: assert every case the
    # paragraph names, in the order it names them
    st = [r for r in nom["records"] if not r["unstable"]]
    check("stable amplitudes below onset", len(st),
          f"over the {['zero','one','two','three','four','five','six','seven','eight','nine'][len(st)]} stable amplitudes",
          "onset_mechanism", numeric=False)
    lead_pair = {tuple(sorted(d["mode"] for d in r["content"]["content"][:2])): r["mu"]
                 for r in st}
    check("below-onset (6,8) case", f"{min(r['mu'] for r in st):.0e}",
          f"the $(6,8)$ pair at $\\mu={min(r['mu'] for r in st) * 1e4:g}\\times10^{{-4}}$",
          "onset_mechanism")
    rows[-1]["pass_"] = (rows[-1]["pass_"] and (6, 8) in lead_pair)

    # --- Sec 6: the growth panel straddles the boundary ---------------------
    gr = {r["mu"]: r for r in om["growth"]}
    g_hi, g_lo = gr[max(gr)], gr[min(gr)]
    thr_nom = nom["threshold_mu"]
    check("growth cases straddle the boundary",
          f"{min(gr) < thr_nom < max(gr)}",
          f"above the boundary (${max(gr) * 1e4:.0f}\\times10^{{-4}}$) it grows "
          f"and saturates, below it (${min(gr) * 1e4:.0f}\\times10^{{-4}}$) decays",
          "onset_mechanism", numeric=False)
    rows[-1]["pass_"] = (rows[-1]["pass_"] and min(gr) < thr_nom < max(gr))
    _t = np.array(g_hi["history"]["t"])
    _e = (np.array(g_hi["history"]["E_upper"])
          / np.array(g_hi["history"]["E_carrier"]))[_t >= 900.0]
    fold = _e[-1] / _e[0]
    check("sideband growth factor above onset", f"{fold:.0f}",
          f"grows by a factor of ${fold:.0f}$", "onset_mechanism")
    for r, scale, printed in ((g_hi, 1e3, "slopes are $%+.2f\\times10^{-3}$"),
                              (g_lo, 1e4, "against $%+.2f\\times10^{-4}$")):
        sl_ = r["log_slope_energy"][f"mode_{r['sideband_modes'][1]}"]
        # computed carries the same scale as the printed literal: comparing
        # +1.34 against a printed 1.34e-3 fails for the wrong reason
        check(f"sideband slope at mu={r['mu']:.0e}", f"{sl_:.2e}",
              printed % (sl_ * scale), "onset_mechanism")
    # below the boundary the sideband must DECAY, which is what the text says
    lo_mode = g_lo["sideband_modes"][1]
    lo_slope = g_lo["log_slope_energy"][f"mode_{lo_mode}"]
    check("below-onset sideband decays", f"{lo_slope:.2e}", "\\emph{decays}",
          "onset_mechanism", numeric=False)
    rows[-1]["pass_"] = (rows[-1]["pass_"] and lo_slope < 0.0)

    # --- Sec 6: the orbit must be solved, not probed (minor 11) -------------
    res = sorted(r["return_residual"] for c in cz for r in c["records"])
    check("Newton residual, converged solves", f"{max(res[:-1]):.0e}",
          f"is at most ${max(res[:-1]):.0e}$".replace("e-", "\\times10^{-").replace("$}", "}$")
          .replace("\\times10^{-12", "\\times10^{-12}") if False else
          f"is at most ${max(res[:-1]) * 10 ** 12:.0f}\\times10^{{-12}}$", "onset_mechanism")
    check("Newton solves attempted", len(res),
          f"One Newton solve of the ${len(res)}$", "onset_mechanism")
    check("non-converged residual", f"{res[-1]:.0e}",
          f"(residual ${res[-1] * 1e3:.0f}\\times10^{{-3}}$)", "onset_mechanism")
    oa = min(om["orbit_accuracy"], key=lambda r: r["mu"])
    byt = {}
    for v in oa["variants"]:
        byt.setdefault(v["t_probe"], []).append(v)
    tres = {t: max(v["return_residual"] for v in vs) for t, vs in byt.items()}
    check("probe return residuals",
          " ".join(f"{tres[t]:.1e}" for t in (1000.0, 2000.0, 3000.0)),
          "residuals of $%.1f\\times10^{-2}$, $%.1f\\times10^{-2}$ and $%.1f\\times10^{-3}$"
          % (tres[1000.0] * 1e2, tres[2000.0] * 1e2, tres[3000.0] * 1e3),
          "onset_mechanism")
    probe_spread = max(r["multiplier_spread"] for r in om["orbit_accuracy"])
    num_spread = max(max(v["max_multiplier"] for v in vs)
                     - min(v["max_multiplier"] for v in vs) for vs in byt.values())
    check("probe-time multiplier spread", f"{probe_spread:.1e}",
          f"largest multiplier by ${probe_spread * 1e2:.1f}\\times10^{{-2}}$",
          "onset_mechanism")
    check("numerical multiplier spread", f"{num_spread:.1e}",
          f"moves it by ${num_spread * 1e6:.1f}\\times10^{{-6}}$", "onset_mechanism")
    margin = min(abs(r["max_multiplier"] - 1.0) for r in unst)
    check("crossing margin", f"{margin:.1e}",
          f"unstable amplitude is\n${margin * 1e3:.1f}\\times10^{{-3}}$", "onset_mechanism")
    check("transient error over margin", f"{probe_spread / margin:.0f}",
          f"transient error is ${probe_spread / margin:.0f}$ times that margin",
          "onset_mechanism")
    check("margin over numerical error", f"{margin / num_spread:.0f}",
          f"smaller than it by a factor of\n${margin / num_spread:.0f}$", "onset_mechanism")
    probed = np.array([t["threshold_mu"] for t in om["threshold_vs_zeta_probed"]])
    short = 100 * (mc - probed) / mc
    check("probed sweep thresholds",
          " ".join(f"{p:.2e}" for p in probed),
          "thresholds at $%.2f$, $%.2f$ and $%.2f\\times10^{-4}$" % tuple(probed * 1e4),
          "onset_mechanism")
    check("probed sweep shortfall", " ".join(f"{v:.0f}" for v in short),
          "low by $%.0f\\,\\%%$, $%.0f\\,\\%%$ and $%.0f\\,\\%%$" % tuple(short),
          "onset_mechanism")

    # --- Sec 7.1: bandwidth pooled ------------------------------------------
    ok_b = [r for r in ba["records"] if abs(r["r_env_trimmed"]) < 5]
    e = np.concatenate([[r["r_env"] for r in sl["records"]],
                        [r["r_env_trimmed"] for r in ok_b]])
    dd = np.concatenate([[r["r_derived"] for r in sl["records"]],
                         [r["r_derived"] for r in ok_b]])
    check("pooled record count", len(e), "$103$ records", "share_law+bandwidth")
    check("pooled law mean abs err", f"{np.mean(np.abs(e - dd)):.4f}",
          "$0.0017$", "share_law+bandwidth")
    check("pooled law worst", f"{np.max(np.abs(e - dd)):.4f}", "$0.0161$",
          "share_law+bandwidth")

    # --- Sec 7.3: the instrumentation recipe --------------------------------
    gs = rr["gamma_summary"]
    for key, printed in (("carrier only", "0.304"), ("carrier +/- 1", "0.287"),
                         ("carrier +/- 2", "0.0039"), ("carrier +/- 4", "0.0012")):
        check(f"recipe {key}", f"{gs[key]['mean_abs_err']:.4f}", printed,
              "recovery_recipe")
    check("vacuity check", f"{rr['vacuity_check']['max_spread_across_subsets']:.1e}",
          "$4\\times10^{-15}$", "recovery_recipe")

    # --- Sec 8: friction boundary -------------------------------------------
    tbl = tex[tex.index("forcing $A$ & slip ratio"):]
    tbl = tbl[:tbl.index("\\bottomrule")]
    check("friction table row count", len(fb["records"]),
          "\\\\\n" * 0 + f"{len(fb['records'])} rows", "friction_boundary")
    rows[-1]["pass_"] = tbl.count("\\\\") - 1 == len(fb["records"])
    for r in fb["records"]:
        check(f"friction gamma at A={r['A']:.0e}", f"{r['gamma_tot_over_c']:.3f}",
              f"{r['gamma_tot_over_c']:.3f}", "friction_boundary")
    check("friction max gamma", f"{fb['summary']['gamma_tot_max']:.2f}",
          f"${fb['summary']['gamma_tot_max']:.2f}$", "friction_boundary")

    # --- Sec 9: noise and sensing -------------------------------------------
    W = ns["window"]
    for sp in (400.0, 4000.0):
        v = np.mean([r["gamma_sub_over_c"] for r in W
                     if r["span"] == sp and r["snr_db"] == 20.0])
        check(f"noise 20 dB at span {sp:.0f}", f"{v:.4f}", f"{v:.4f}", "noise_sensing")
    ss = ns["sensing_summary"]
    check("8 vs 6 sensor ratio",
          f"{ss['8']['mean_abs_err'] / ss['6']['mean_abs_err']:.2f}",
          "eight equally\nspaced sensors are worse than six", "noise_sensing",
          numeric=False)

    # --- numbers that appear only in figures and their captions --------------
    # A caption is prose like any other and drifts from the data the same way,
    # so every value quoted in one is checked here too.
    nsk = om["neimark_sacker"]
    check("unstable multipliers", nsk["n_unstable_max"], "\\emph{exactly two}",
          "onset_mechanism")
    rows[-1]["pass_"] = (nsk["n_unstable_max"] == 2
                         and "\\emph{exactly two}" in tex)
    arg = np.mean([r["arg_over_pi"] for r in nsk["records"]])
    check("crossing argument", f"{arg:.3f}", f"\\pm{arg:.3f}\\pi",
          "onset_mechanism")
    check("rotation frequency", f"{nsk['mean_rotation']:.4f}",
          f"{nsk['mean_rotation']:.4f}", "onset_mechanism")
    check("sideband mean offset", f"{nsk['pair_mean_offset']:.4f}",
          f"{nsk['pair_mean_offset']:.4f}", "onset_mechanism")
    check("rotation vs offset", f"{100 * nsk['rel_error']:.1f}",
          f"{100 * nsk['rel_error']:.1f}\\,\\%", "onset_mechanism")
    mod = [abs(complex(re_, im_)) for e in om["floquet_true_orbit"]
           for re_, im_ in zip(e["multipliers_re"], e["multipliers_im"])]
    med = float(np.median(mod))
    check("multipliers off the circle", f"{med:.3f}",
          f"|\\lambda|\\approx{med:.3f}", "onset_mechanism")
    # Fig: below the onset the two independent estimates agree, and the
    # caption says by how much. The guard is on the claim, not just the string:
    # every single-mode cell must have both estimates within 1 per cent of the
    # truth and within 8.2e-3 of each other.
    below = [c for c in bm["grid"] + bm["threshold_sweep"] if c["e_frac"] > 0.99]
    gap = max(abs(c["r_gcvd"] - c["r_env"]) for c in below)
    check("decomposition vs reference below onset", f"{gap:.1e}",
          "agree to\n$8.2\\times10^{-3}$ over an $800$-fold range of $\\mu$",
          "bias_map")
    rows[-1]["pass_"] = (gap < 8.3e-3
                         and max(abs(c["r_gcvd"] - 1) for c in below) < 0.01
                         and max(abs(c["r_env"] - 1) for c in below) < 0.01
                         and re.sub(r"\s+", " ", rows[-1]["printed"]) in flat)
    # Fig: sideband share and the recipe gain
    dest = max(om["destinations"], key=lambda d: d["mu"])
    sh = {t["mode"]: t["share"] for t in dest["top"]}
    check("sideband share at largest mu", f"{sh[5] + sh[9]:.2f}",
          "more than half of the", "onset_mechanism")
    rows[-1]["pass_"] = (sh[5] + sh[9] > 0.5
                         and "more than half of the" in tex)
    check("recipe gain, carrier to pair",
          f"{gs['carrier only']['mean_abs_err'] / gs['carrier +/- 2']['mean_abs_err']:.0f}",
          "factor of $78$", "recovery_recipe")
    # Fig: friction contrast, single-mode fit on the conservative system
    check("conservative single-mode floor",
          f"{min(r['r_env'] for r in sl['records']):.2f}",
          "falls away to\n$0.45$", "share_law")
    # Fig: the two estimators over the window x SNR grid
    err, errn = {}, {}
    for r in W:
        err.setdefault((r["snr_db"], r["span"]), []).append(
            r["gamma_sub_over_c"] - 1.0)
        errn.setdefault((r["snr_db"], r["span"]), []).append(
            r["gamma_sub_naive_over_c"] - 1.0)
    spans = sorted({r["span"] for r in W})
    # |mean - 1| per cell, which is what a table of means reports. The other
    # statistic, mean|. - 1|, is larger wherever realizations straddle the
    # true value and answers a different question.
    mf = max(abs(np.mean(v)) for v in err.values())
    mn = max(abs(np.mean(v)) for v in errn.values())
    check("floor-aware worst over grid (pct)", f"{mf * 100:.2f}",
          "is $0.44\\,\\%$ in the mean", "noise_sensing")
    check("log-linear worst over grid (pct)", f"{mn * 100:.1f}",
          "runs to $10.8\\,\\%$", "noise_sensing")
    check("estimator improvement factor", f"{mn / mf:.0f}",
          "factor of $24$", "noise_sensing")
    at20 = [abs(np.mean(errn[(20.0, s_)])) for s_ in spans]
    check("log-linear rebound at 20 dB", f"{at20[-1] / min(at20):.0f}",
          "$34$ between the best window", "noise_sensing")
    at20f = [abs(np.mean(err[(20.0, s_)])) for s_ in spans]
    check("floor-aware bound at 20 dB (pct)", f"{max(at20f) * 100:.2f}",
          "bounded by $0.34\\,\\%$", "noise_sensing")
    wr = max(abs(r["gamma_sub_over_c"] - 1.0) for r in W)
    check("floor-aware worst single realization (pct)", f"{wr * 100:.1f}",
          "$1.5\\,\\%$ in a", "noise_sensing")
    # the fitted floor as the aliasing diagnostic
    fl = {}
    for r in ns["sensing"]:
        fl.setdefault(r["n_sensors"], []).append(r["floor_rel"])
    check("fitted floor at m=8", f"{np.mean(fl[8]):.1e}",
          "$1.0\\times10^{-1}$", "noise_sensing")
    check("fitted floor at m=12", f"{np.mean(fl[12]):.1e}",
          "$8.0\\times10^{-13}$", "noise_sensing")

    # Fig: the similarity collapse, from the lambda-invariant pairs
    bymu = {}
    for c in bm["grid"]:
        bymu.setdefault(round(c["k3"] * c["A"] ** 2, 12), []).append(c)
    inv = {m: v for m, v in bymu.items() if len(v) > 1 and m > 0}
    check("invariant pairs", len(inv), "the four\n$\\lambda$-invariant pairs",
          "bias_map")
    rows[-1]["pass_"] = (len(inv) == 4
                         and re.sub(r"\s+", " ", rows[-1]["printed"]) in flat)
    for key, lab in (("r_gcvd", "decomposition"), ("r_env", "envelope")):
        sp = max(max(c[key] for c in v) - min(c[key] for c in v)
                 for v in inv.values())
        mant, ex = f"{sp:.1e}".split("e")
        check(f"invariant-pair spread, {lab}", f"{sp:.1e}",
              f"${mant}\\times10^{{{int(ex)}}}$", "bias_map")
    # Fig: the bandwidth protocol
    at25 = [r for r in ba["records"] if r["beta"] == 0.25]
    check("share at beta=0.25",
          f"{min(r['share_mean'] for r in at25):.2f}"
          f"-{max(r['share_mean'] for r in at25):.2f}",
          "$0.64$--$0.89$", "bandwidth_axis")
    lo25, hi25 = (min(r["r_env"] for r in at25), max(r["r_env"] for r in at25))
    check("damping at beta=0.25", f"{lo25:.3f}-{hi25:.3f}",
          f"damping of ${lo25:.2f}$--${hi25:.2f}$", "bandwidth_axis")
    # The bandwidth-only share scatter was removed as a duplicate of the
    # damping-law panel, so the pooled statistics are what the text now prints.
    xs = ([r["share_mean"] for r in ba["records"]]
          + [r["share_mean"] for r in sl["records"]])
    ys = ([r["r_env_trimmed"] for r in ba["records"]]
          + [r["r_env"] for r in sl["records"]])
    es = np.abs(np.array(xs) - np.array(ys))
    check("pooled share proxy mean error", f"{es.mean():.3f}",
          "mean absolute error $0.395$", "pooled")
    check("pooled share proxy worst", f"{es.max():.2f}", "worst case $5.09$",
          "pooled")
    check("pooled share proxy correlation",
          f"{np.corrcoef(xs, ys)[0, 1]:.3f}", "correlation\n$-0.004$", "pooled")

    # --- Sec 6: the single-harmonic cross-check ------------------------------
    # The manuscript now compares two BRACKETS (checked above) rather than an
    # interpolated crossing against a probed edge. Interpolating a crossing
    # from a grid coarser than the difference being claimed was how the old
    # "agreement to 0.7 per cent" arose: the 0.7 per cent was smaller than the
    # 0.001 grid step it was read off. Kept as retired claims so the text
    # cannot quietly regain them.
    for gone in ("at $0.0831$", "$0.0837$--$0.0881$", "residual $0.7\\,\\%$"):
        check(f"retired: {gone}", "absent", gone, "manuscript", numeric=False)
        rows[-1]["pass_"] = re.sub(r"\s+", " ", gone) not in flat

    # --- Sec 7.3: the weighting test on a Rayleigh-damped chain -------------
    for w, lab in (("c_share", "energy share"), ("c_amp", "modal amplitude"),
                   ("c_mode", "tracked mode"), ("c_flat", "unweighted")):
        st_ = nu["summary"][w]
        check(f"weighting {lab}: median", f"{st_['median']:.4f}",
              f"${st_['median']:.4f}$", "nonuniform_damping")
        check(f"weighting {lab}: worst", f"{st_['worst']:.4f}",
              f"${st_['worst']:.4f}$", "nonuniform_damping")
        check(f"weighting {lab}: above 10pct", f"{st_['n_above_10pct']}",
              f"${st_['n_above_10pct']}$", "nonuniform_damping")
    ci = nu["config"]["c_i"]
    check("Rayleigh spread", f"{max(ci) / min(ci):.0f}", "a spread of $16$",
          "nonuniform_damping")
    check("Rayleigh record count", f"{len(nu['records'])}", "$33$ records",
          "nonuniform_damping")
    cb = [r["c_share_bar"] / r["c_true_mode"] for r in nu["records"]]
    check("cbar over c_m range, low", f"{min(cb):.3f}", "$0.844$",
          "nonuniform_damping")
    check("cbar over c_m range, high", f"{max(cb):.3f}", "$1.066$",
          "nonuniform_damping")

    # --- Sec 11: the second configuration -----------------------------------
    check("second configuration psi ratio", f"{sc2['config']['psi_ratio']:.2f}",
          "a factor of $0.37$ in coupling", "second_configuration")
    check("second configuration worst envelope",
          f"{sc2['below_onset']['worst_env'] * 100:.2f}",
          "returns the true coefficient to\n$0.01\\,\\%$", "second_configuration")
    check("second configuration worst decomposition",
          f"{sc2['below_onset']['worst_gcvd'] * 100:.2f}",
          "agrees with it to $0.01\\,\\%$", "second_configuration")
    check("second configuration share",
          f"{min(c['e_frac'] for c in sc2['sweep']):.4f}",
          "holds $0.9988$ of the energy", "second_configuration")
    check("second configuration max multiplier",
          f"{max(r['max_multiplier'] for r in sc2['floquet']):.3f}",
          "the largest is\n$0.998$", "second_configuration")
    rows[-1]["pass_"] = (rows[-1]["pass_"]
                         and sc2["floquet_threshold_mu"] is None)
    check("second configuration mu range top",
          f"{max(c['mu'] for c in sc2['sweep']):.1e}", "$3.2\\times10^{-3}$",
          "second_configuration")

    # --- values that were corrected once and must not reappear ---------------
    # Every check above asks whether a correct string is PRESENT, which passes
    # happily while a superseded value survives somewhere else in the file --
    # exactly how the section 4 sentence came to contradict the Figure 2
    # caption about the same four invariant pairs. These ask the opposite.
    for bad, why in (
            ("9.7\\times10^{-9}", "superseded invariant-pair spread"),
            ("five pairs", "there are four invariant pairs, not five"),
            ("$9.5\\times10^{-8}$", "superseded decomposition invariance spread"),
            ("three decades", "800-fold is 2.90 decades; the shift spans 2.55"),
            ("two orders of magnitude", "the degradation figure was 31x"),
            ("four times worse", "the 8-vs-6 sensor ratio is not four"),
            ("five times worse", "the 8-vs-6 sensor ratio is 1.94"),
            ("rank-deficient", "the rank-deficiency mechanism is withdrawn"),
            ("$111\\,\\%$", "superseded maximum decomposition error (grid-only maximum; the single-mode records give 128)"),
            ("optimum shortens", "the window optimum was an estimator artifact"),
            ("best window shortens", "the window optimum was an estimator artifact")):
        check(f"retired: {why}", "absent", bad, "manuscript")
        rows[-1]["pass_"] = re.sub(r"\s+", " ", bad) not in flat
        rows[-1]["printed"] = f"{bad!r} must NOT appear"

    # --- report --------------------------------------------------------------
    npass = sum(r["pass_"] for r in rows)
    w = max(len(r["label"]) for r in rows)
    for r in rows:
        print(f"{'PASS' if r['pass_'] else 'FAIL'}  {r['label']:<{w}}  "
              f"computed {str(r['computed']):>12}   [{r['source']}]")
        if not r["pass_"]:
            print(f"      manuscript does not contain: {r['printed']!r}")
    print(f"\n{npass} of {len(rows)} checks passed")
    if args.json:
        out = os.path.join(CK, "verification_paper3.json")
        with open(out, "w") as fh:
            json.dump(dict(n_pass=npass, n_total=len(rows), checks=rows), fh, indent=1)
        print(f"wrote {out}")
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
