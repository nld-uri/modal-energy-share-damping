"""Figure: stability of the tracked periodic orbit.

Reads only onset_mechanism_results.json; recomputes nothing.

Label placement follows the hand adjustments made on the typeset figure:
the crossing-pair callout is set on two lines clear of the top tick label,
the "all other" note sits above the multiplier row rather than on it, and
panel (c)'s legend is moved off the rising curve.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BLUE, ORANGE, DKRED = "#0072BD", "#D95319", "#A2142F"
GREY_TEXT = "#555555"

# --- hand-adjusted label placement ------------------------------------------
CROSS_XY = (0.00335749, 1.014)        # crossing-pair callout, two lines, centred
OTHER_XY = (-0.16, 0.9922)      # "all other 46 multipliers", above the row
# The two traces now straddle the boundary rather than the bias band, leaving
# the middle-right of the panel empty: the saturated trace sits near the top
# and the decaying one near the bottom.
LEGEND_C = dict(loc="center right", bbox_to_anchor=(1.0, 0.56))


def make(ckpt=".", out="floquet_stability.pdf"):
    om = json.load(open(f"{ckpt}/onset_mechanism_results.json"))
    # Newton-shot periodic orbits at the nominal damping, NOT the probed
    # transients: a probe cannot resolve the threshold (see orbit_accuracy).
    FT = next(b for b in om["converged"]
              if abs(b["zeta1"] - om["config"]["zeta1"]) < 1e-12)["records"]
    mu_all = np.array([f["mu"] for f in FT])
    mx_all = np.array([f["max_multiplier"] for f in FT])

    def spec_at(mu):
        f = min(FT, key=lambda f: abs(f["mu"] - mu))
        return np.array(f["multipliers_re"]) + 1j * np.array(f["multipliers_im"])

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(174 / 25.4, 60 / 25.4))

    # (a) modulus against argument: the crossing is invisible on a unit disc
    axa.axhline(1.0, color="k", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for mu, col, lab in ((2.0e-4, BLUE, r"$\mu=2\times10^{-4}$"),
                         (8.0e-4, ORANGE, r"$\mu=8\times10^{-4}$")):
        ev = spec_at(mu)
        axa.plot(np.angle(ev) / np.pi, np.abs(ev), "o", color=col, ms=3.4,
                 mfc="none", mew=0.9, zorder=3, label=lab)
    ev8 = spec_at(8.0e-4)
    lead = ev8[np.abs(ev8) > 1.0]
    axa.plot(np.angle(lead) / np.pi, np.abs(lead), "o", color=DKRED, ms=5.0,
             zorder=5)
    axa.set_xlabel(r"$\arg\lambda/\pi$")
    axa.set_ylabel(r"$|\lambda|$")
    axa.set_title(r"Only one conjugate pair" "\n" r"crosses $|\lambda|=1$",
                  loc="left")
    axa.legend(loc="center left", frameon=False, handlelength=0.9,
               handletextpad=0.4, borderaxespad=0.4, labelspacing=0.2,
               fontsize=7, bbox_to_anchor=(0.0, 0.62))
    arg_pi = np.mean([r["arg_over_pi"]
                      for r in om["neimark_sacker"]["records"]])
    axa.text(*CROSS_XY, "the crossing pair" "\n" rf"$\arg\lambda=\pm{arg_pi:.3f}\pi$",
             color=DKRED, ha="center", va="bottom", fontsize=7,
             multialignment="center", linespacing=1.7)
    axa.text(*OTHER_XY, r"all other $46$ multipliers", color=GREY_TEXT,
             ha="center", va="bottom", fontsize=7)
    axa.set_ylim(0.9875, 1.0185)
    axa.set_xlim(-1.08, 1.08)

    # (b) where the boundary sits, against where the bias appears
    st = mu_all[mx_all <= 1].max()
    un = mu_all[mx_all > 1].min()
    axb.axhline(1.0, color="k", lw=0.7, ls=(0, (4, 3)), zorder=1)
    axb.axvspan(4.0e-4, 5.0e-4, color=META_GREY, alpha=0.3, lw=0, zorder=0)
    axb.plot(mu_all, mx_all, "-o", color=BLUE, ms=3.6, lw=1.4, zorder=3)
    axb.axvline(next(b for b in om["converged"]
                     if abs(b["zeta1"] - om["config"]["zeta1"]) < 1e-12)["threshold_mu"],
                color=DKRED, lw=0.9, ls=(0, (1, 2)), zorder=2)
    axb.set_xscale("log")
    # Default log minor labels collide at this panel width: put the decade on the
    # axis label and tick the mantissa.
    axb.xaxis.set_major_locator(mticker.FixedLocator([2e-4, 3e-4, 5e-4, 1e-3, 2e-3]))
    axb.xaxis.set_minor_locator(mticker.NullLocator())
    axb.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v * 1e4:g}$"))
    axb.set_xlabel(r"$\mu=k_{3}A^{2}$ $(\times10^{-4})$")
    axb.set_ylabel(r"$\max|\lambda|$")
    thr = next(b for b in om["converged"]
               if abs(b["zeta1"] - om["config"]["zeta1"]) < 1e-12)["threshold_mu"]
    axb.set_title(rf"Boundary at $\mu={thr * 1e4:.2f}\times10^{{-4}}$," "\n"
                  r"below where the bias appears", loc="left")
    axb.text(2.95e-4, 1.0075, r"Floquet", color=DKRED, ha="right", fontsize=7)
    axb.text(5.4e-4, 0.9905, r"bias onset", color=GREY_TEXT, ha="left",
             fontsize=7)
    axb.set_ylim(0.9885, 1.0155)

    # (c) the sideband growing, past the startup transient
    for case, col, side in ((0, BLUE, "below"), (1, ORANGE, "above")):
        mu_c = om["growth"][case]["mu"]
        assert (mu_c < thr) == (side == "below"), (
            f"growth case {case} at mu={mu_c:.3e} is not {side} the boundary {thr:.3e}")
        lab = rf"{side}, ${mu_c * 1e4:g}\times10^{{-4}}$"
        h = om["growth"][case]["history"]
        t = np.array(h["t"])
        e = np.array(h["E_upper"]) / np.array(h["E_carrier"])
        k = t >= 900.0
        axc.semilogy(t[k], e[k], color=col, lw=1.3, zorder=3, label=lab)
    axc.set_xlabel(r"time under forcing")
    axc.set_ylabel(r"$E_{9}/E_{7}$")
    hg = om["growth"][1]["history"]
    tg, eg = np.array(hg["t"]), (np.array(hg["E_upper"])
                                / np.array(hg["E_carrier"]))
    fold = eg[tg >= 900.0][-1] / eg[tg >= 900.0][0]
    axc.set_title(rf"Above it the sideband grows" "\n" rf"${fold:.0f}$-fold, then saturates",
                  loc="left")
    axc.legend(frameon=False, handlelength=1.2, borderaxespad=0.3,
               labelspacing=0.2, fontsize=7, **LEGEND_C)

    for ax, L in ((axa, "a"), (axb, "b"), (axc, "c")):
        panel_letter(ax, L, case="lower")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    return fig
