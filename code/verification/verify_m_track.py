"""Check every number of m_track/proposed_sections.tex against the M-track result JSONs.

Same discipline as code/verification/verify_paper3.py: each check recomputes a
quantity from a stored result, and the printed form must (i) occur in the text
(whitespace collapsed) and (ii) equal the recomputed value rounded to the digits
printed. A check also fails if its string is absent.

    python verify_m_track.py [TEX]      # default: proposed_sections.tex next to this file
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Imported into paper 3 on 2026-10-25: the results now live under
# checkpoints/m_track/ and the default manuscript is paper3.tex.
R = os.path.join(HERE, "..", "..", "checkpoints", "m_track")
TEX = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER3_TEX", os.path.join(HERE, "..", "..", "manuscript", "paper3_nd.tex"))


def load(p):
    with open(p) as fh:
        return json.load(fh)


def rounds_to(value, printed):
    """True if value rounds to the printed decimal (mantissa digits as printed)."""
    s = printed.replace("\\times10^{", "e").replace("}", "")
    m = re.fullmatch(r"(-?)(\d+)(?:\.(\d+))?(?:e(-?\d+))?", s)
    assert m, printed
    dec = len(m.group(3) or "")
    exp = int(m.group(4) or 0)
    p = float(s)
    half = 0.5 * 10 ** (exp - dec)
    return abs(value - p) <= half * (1 + 1e-9)


def main():
    flat = re.sub(r"\s+", " ", open(TEX).read())
    m1 = load(os.path.join(R, "m1_averaged.json"))
    m3 = load(os.path.join(R, "m3_lowmu_floquet.json"))
    m3b = load(os.path.join(R, "m3b_lowwindow_identification.json"))
    m5 = load(os.path.join(R, "m5_share_vs_error.json"))
    m6 = load(os.path.join(R, "m6_coefficients.json"))
    m2 = load(os.path.join(R, "m2", "protocols_phase_resonance.json"))
    Z = {z["zeta1"]: z for z in m1["zetas"]}
    rows = []

    def check(label, value, printed, snippet=None, scale=1.0):
        snip = snippet if snippet is not None else printed
        present = re.sub(r"\s+", " ", snip) in flat
        ok = present and rounds_to(value * scale, printed)
        rows.append((ok, label, value * scale, printed))

    def same(label, ok, snippet):
        rows.append((bool(ok) and re.sub(r"\s+", " ", snippet) in flat, label, ok, snippet))

    # ---- M1: averaged condition, omega_lin protocol ----
    for z, p in ((0.005, "0.0801"), (0.010, "0.0837"), (0.020, "0.101")):
        check(f"M1 kappa_c (5,9) zeta={z}", Z[z]["kappa_c_59"], p, f"$\\kappa_{{\\mathrm{{c}}}}={p}$" if z == 0.005 else p)
        same(f"M1 kappa_c inside Floquet bracket zeta={z}", Z[z]["kappa_c_59_in_bracket"], "Each lies inside the bracket of converged orbits")
        same(f"M1 first pair is (5,9) zeta={z}", all(m["floquet_leading_two"] == [5, 9] for m in Z[z]["multipliers"] if m["floquet"] > 1), "the first pair to satisfy it is $(5,9)$")
    for z, p in ((0.005, "2.98"), (0.010, "3.85"), (0.020, "8.63")):
        check(f"M1 mu_c predicted zeta={z} (x1e-4)", Z[z]["mu_c_pred"], p, f"$\\mu_{{\\mathrm{{c}}}}=2.98$, $3.85$ and $8.63\\times10^{{-4}}$", scale=1e4)
    check("M1 worst |mu_c pred / port - 1| (%)", max(abs(Z[z]["mu_c_rel_err"]) for z in Z), "0.9", "within $0.9\\,\\%$ of the monodromy values", scale=100)
    nrec = sum(len(Z[z]["multipliers"]) for z in Z)
    same("M1 number of converged orbits = 47", nrec == 47, "$47$ converged orbits")
    check("M1 max |lambda_pred - lambda_Floquet|", max(Z[z]["max_abs_mult_diff"] for z in Z), "7.5\\times10^{-5}", "by at most $7.5\\times10^{-5}$")
    same("M1 no side disagreement", all(not Z[z]["side_disagreements"] for z in Z), "never disagree about which side")
    P = m1["p14"]["pairs"]
    check("P1.4 raw detuning (6,8)", P["6-8"]["sigma_raw"], "0.00336", "its detuning is $0.00336$")
    check("P1.4 raw detuning (5,9)", P["5-9"]["sigma_raw"], "0.0134", "against $0.0134$ for $(5,9)$")
    check("P1.4 psi product (6,8)", P["6-8"]["psi_product_raw"], "0.0722", "$|\\psi_{6}\\psi_{8}|=0.0722$")
    check("P1.4 psi product (5,9)", P["5-9"]["psi_product_raw"], "0.0586", "$|\\psi_{5}\\psi_{9}|=0.0586$")
    check("P1.4 (6,8) exact tuning kappa", m1["p14"]["kappa_exact_tuning"]["6-8"], "0.0254", "exact tuning already at $\\kappa=0.0254$")
    check("P1.4 shifted detuning (6,8) at kappa_c", P["6-8"]["sigma_shifted"], "-0.00770", "$-0.00770$")
    check("P1.4 shifted detuning (5,9) at kappa_c", P["5-9"]["sigma_shifted"], "0.00364", "$(5,9)$ to $0.00364$")
    same("P1.4 (6,8) margin negative at kappa_c", P["6-8"]["margin"] < 0, "the pair the stiffening tunes \\emph{in}")
    check("M1 kappa_c ratio 0.020/0.005", m1["damping_dependence"]["kappa_ratio_020_005"], "1.26", "a factor of only $1.26$")
    check("M1 mu ratio 0.020/0.005 (port)", m1["damping_dependence"]["mu_ratio_020_005"], "2.95", "by a factor of $2.95$ in $\\mu$")

    # ---- M3: lower window at zeta1 = 0.005 ----
    lo = m1["lower_intervals"]["0.005"]["6-8"][0]
    check("M3 predicted window lower kappa", lo[0], "0.0212", "from $\\kappa=0.0212$ to $0.0457$")
    check("M3 predicted window upper kappa", lo[1], "0.0457", "from $\\kappa=0.0212$ to $0.0457$")
    same("M3 no window predicted at 0.010 / 0.020", m1["lower_intervals"]["0.01"] == {} and m1["lower_intervals"]["0.02"] == {}, "at $\\zeta_{1}=0.005$ only")
    S = {s["zeta1"]: s for s in m3["sweep"]}
    check("M3 Floquet lower threshold zeta=0.005 (x1e-6)", S[0.005]["threshold_mu"], "9.6", "from $\\mu=9.6\\times10^{-6}$", scale=1e6)
    r005 = {r["mu"]: r for r in S[0.005]["records"]}
    same("M3 unstable at 6e-5, stable at 7e-5", r005[6e-5]["unstable"] and not r005[7e-5]["unstable"], "between $6\\times10^{-5}$ and $7\\times10^{-5}$")
    check("M3 largest multiplier zeta=0.005", max(r["max_multiplier"] for r in S[0.005]["records"]), "1.0023", "largest multiplier of $1.0023$")
    un = [r for r in S[0.005]["records"] if r["unstable"]]
    same("M3 leading pair (6,8) at every unstable record", all(sorted(c["mode"] for c in r["content"]["content"][:2]) == [6, 8] for r in un), "between modes $8$ and $6$")
    check("M3 eigenvector share mode 8", un[0]["content"]["content"][0]["share"], "0.567", "$0.567$ and $0.433$")
    check("M3 eigenvector share mode 6", un[0]["content"]["content"][1]["share"], "0.433", "$0.567$ and $0.433$")
    same("M3 all stable at zeta=0.010", not any(r["unstable"] for r in S[0.01]["records"]), "stay stable throughout")
    check("M3 largest multiplier zeta=0.010", max(r["max_multiplier"] for r in S[0.01]["records"]), "0.9977", "reaching $0.9977$")
    win = [r for r in m3b["records"] if r["mu"] <= 6e-5]
    check("M3b min share in window", min(r["share"] for r in win), "0.998", "share of at least $0.998$")
    same("M3b min share >= printed bound", min(r["share"] for r in win) >= 0.998, "share of at least $0.998$")
    check("M3b max |r_env - 1| in window (%)", max(abs(r["r_env"] - 1) for r in win), "0.17", "within $0.17\\,\\%$", scale=100)
    same("M3b Q1 holds", m3b["Q1_identification_silent"], "The window is silent in identification")

    # ---- M2: phase resonance ----
    W = {s["zeta1"]: s for s in m2["sweeps"]}
    for z, pk, pa in ((0.005, "0.0297", "0.0297"), (0.010, "0.0404", "0.0405"), (0.020, "0.0827", "0.0831")):
        check(f"M2 Floquet onset kappa zeta={z}", W[z]["floquet_boundaries"][0]["kappa"], pk, "$\\kappa=0.0297$, $0.0404$ and $0.0827$")
        check(f"M2 averaged edge kappa zeta={z}", W[z]["averaged_windows_at_protocol_Omega"]["6-8"][0], pa, "places at $0.0297$, $0.0405$ and $0.0831$")
        same(f"M2 onset pair (6,8) zeta={z}", sorted(W[z]["floquet_boundaries"][0]["pair_unstable_side"]) == [6, 8], "is the $(6,8)$ pair")

    # ---- M5: share vs damping error on the port's map ----
    E = m5["estimators"]
    same("M5 35 cells", m5["n_cells"] == 35, "of the $35$ cells")
    same("M5 25 cells with share >= 0.99", E["r_env"]["n_share_hi"] == 25 == E["r_gcvd"]["n_share_hi"], "the $25$ with a share of at least $0.99$")
    same("M5 no high-share cell outside 0.5 %", E["r_env"]["N1_hi_share_but_error"] == 0 == E["r_gcvd"]["N1_hi_share_but_error"], "all lie within $0.5\\,\\%$")
    same("M5 no low-share cell inside 0.5 %", E["r_env"]["N2_lo_share_but_accurate"] == 0 == E["r_gcvd"]["N2_lo_share_but_accurate"], "the $10$ below it all lie outside")
    check("M5 worst envelope error, high share (%)", E["r_env"]["max_err_hi_share"], "0.48", "worst cases $0.48\\,\\%$ for the envelope", scale=100)
    check("M5 worst GCVD error, high share (%)", E["r_gcvd"]["max_err_hi_share"], "0.35", "$0.35\\,\\%$ for the decomposition", scale=100)

    # ---- M6: model-informed recovery ----
    r = next(x for x in m6["records"] if x["mu"] == 1.6e-3)
    check("M6 envelope ratio at mu=1.6e-3", r["r_env"], "0.451", "returns $0.451$")
    check("M6 |c err| model acceleration", abs(r["a_model"]["c_rel_err"]), "2.0\\times10^{-14}", "returns $c$ to $2.0\\times10^{-14}$")
    check("M6 |k3 err| model acceleration", abs(r["a_model"]["k3_rel_err"]), "1.0\\times10^{-14}", "$k_{3}$ to $1.0\\times10^{-14}$")
    check("M6 |c err| differenced acceleration", abs(r["b_fd4"]["c_rel_err"]), "4.6\\times10^{-7}", "to $4.6\\times10^{-7}$")
    check("M6 |k3 err| differenced acceleration", abs(r["b_fd4"]["k3_rel_err"]), "2.3\\times10^{-6}", "and $2.3\\times10^{-6}$")

    # ---- M7: further configurations ----
    m7 = load(os.path.join(R, "m7_second_config.json"))
    a, b = m7["primary"], m7["secondary"]
    same("M7 primary: station 5, mode 6", (a["station_1based"], a["mode_1based"]) == (5, 6), "Tracking mode $6$ with the spring at mass $5$")
    same("M7 primary: predicted pair (4,8) = Floquet pair", a["P7_2"] and a["floquet_pair"] == [4, 8], "it names the right pair, $(4,8)$")
    check("M7 primary predicted kappa_c", a["kappa_c_pred"], "0.0807", "threshold at $\\kappa=0.0807$")
    check("M7 primary offset below bracket (%)", -a["kappa_pred_rel_to_bracket_low"], "0.7", "$0.7\\,\\%$ below", scale=100)
    same("M7 primary P7.1 fails (prediction below the bracket)", not a["P7_1"] and a["kappa_c_pred"] < a["kappa_bracket"][0], "below the converged-orbit bracket")
    check("M7 primary bracket low", a["kappa_bracket"][0], "0.0813", "bracket of $0.0813$ to $0.0815$")
    check("M7 primary bracket high", a["kappa_bracket"][1], "0.0815", "bracket of $0.0813$ to $0.0815$")
    check("M7 primary share mode 8", a["floquet_shares"][0], "0.624", "shared $0.624$ and $0.305$")
    check("M7 primary share mode 4", a["floquet_shares"][1], "0.305", "shared $0.624$ and $0.305$")
    same("M7 secondary: station 10, mode 7", (b["station_1based"], b["mode_1based"]) == (10, 7), "Tracking mode $7$ with the spring at mass $10$")
    same("M7 secondary: predicted (1,14), Floquet (14,19)", b["pair_pred"] == [1, 14] and b["floquet_pair"] == [14, 19] and not b["P7_2"], "$(1,14)$ pair")
    check("M7 secondary predicted kappa_c", b["kappa_c_pred"], "0.0823", "at $\\kappa=0.0823$")
    check("M7 secondary offset (%)", -b["kappa_pred_rel_to_bracket_low"], "14", "$14\\,\\%$ higher", scale=100)
    check("M7 secondary bracket low", b["kappa_bracket"][0], "0.0959", "between $0.0959$ and $0.0963$")
    check("M7 secondary bracket high", b["kappa_bracket"][1], "0.0963", "between $0.0959$ and $0.0963$")
    same("M7 P7.3 in both", a["P7_3"] and b["P7_3"], "departs only above it")
    check("M7 max envelope error below boundary (%)", max(a["max_env_err_below"], b["max_env_err_below"]), "0.12", "within $0.12\\,\\%$ of the true coefficient below", scale=100)
    raw = load(os.path.join(R, "m7", "theory_second_config.json"))["primary"]["release"]
    r13 = next(r for r in raw if abs(r["mu"] - 1.3e-3) < 1e-12)
    check("M7 primary mu=1.3e-3 share", r13["e_frac"], "0.555", "share of only $0.555$")
    check("M7 primary mu=1.3e-3 |r_env - 1| (%)", abs(r13["r_env"] - 1), "0.28", "within $0.28\\,\\%$", scale=100)

    nfail = 0
    for ok, label, v, p in rows:
        nfail += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label:<58s} {v!s:<24.24s} {p}")
    # --- added 2026-10-25 with the consolidation into paper3.tex ---------------
    # Fig. friction's caption claimed the total-energy decay returns the true modal
    # damping "to six decimals". The quantity it plots is cbar_over_ctrue, which is
    # unity IDENTICALLY under this system's uniform damping (share_law_results.json
    # records all_equal = True), so the claim presented an algebraic identity as a
    # measurement. The caption now states the identity and quotes a fitted rate
    # instead; these two checks bind both halves of the replacement.
    _ck = os.path.join(HERE, "..", "..", "checkpoints")
    _g = [r["gamma_tot_over_c"] for r in load(os.path.join(_ck, "cubic_gamma_tot.json"))]
    _worst = 100 * max(abs(1.0 - x) for x in _g)
    same("cubic fitted gamma_tot/c stays within the printed 0.3 %",
         _worst <= 0.3, "to within\n$0.3\\,\\%$")
    _cb = [r["cbar_over_ctrue"] for r in
           load(os.path.join(_ck, "share_law_results.json"))["records"]]
    same("cbar/c_true is unity identically, so the caption must say so",
         max(abs(1.0 - x) for x in _cb) == 0.0, "unity\n\\emph{identically}")

    # --- M4, added 2026-10-25 under pre-declaration addendum 4 -----------------
    # Independent re-run (m4_growth_budget.json); reproduces Alt B's stored result
    # exactly. P4.1 and P4.3 passed, P4.2 was REFUTED: 16 of 28 records have an
    # observed share far above the linear budget, every one of them where the
    # prediction lies below the floor the records sit on. The manuscript states the
    # restricted form, so these checks bind the restriction, not the refuted claim.
    _m4 = load(os.path.join(_ck, "m_track", "m4_growth_budget.json"))
    check("M4 seeded hold: release pair-energy share", _m4["seed_pair_energy_share"],
          "1.3", snippet=r"$1.3\times10^{-6}$", scale=1e6)
    check("M4 seeded failure kappa, phase-resonant", _m4["kappa_fail_pred_PR"], "0.0695")
    check("M4 seeded failure kappa, omega_lin", _m4["kappa_fail_pred_LIN"], "0.0951")
    _rec = _m4["records"]
    _hi = [r for r in _rec if r["pred_pair_share_linear"] >= 1e-3]
    _ratio = max(r["obs_pair_share"] / r["pred_pair_share_linear"] for r in _hi)
    same("M4 budget is an upper bound once it predicts at least 1e-3",
         _ratio <= 0.95 and len(_hi) == 11, "ratio at most $0.95$")
    _lo = [r for r in _rec if r["pred_pair_share_linear"] < 1e-3]
    same("M4 observed floor below that regime stays under 5e-4",
         max(r["obs_pair_share"] for r in _lo) <= 5e-4, r"at most\n$5\times10^{-4}$")
    same("M4 record count", len(_rec) == 28, "$28$ records")

    print(f"\n{len(rows) - nfail} of {len(rows)} checks passed")
    json.dump([dict(pass_=ok, label=l, computed=str(v), printed=str(p)) for ok, l, v, p in rows],
              open(os.path.join(HERE, "results", "verify_m_track.json"), "w"), indent=1)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
