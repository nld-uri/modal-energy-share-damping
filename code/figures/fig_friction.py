"""Figure: the conservative/dissipative boundary.

Reads only stored JSONs (share_law_results.json, friction_boundary_results.json);
recomputes nothing.
"""
import json
import matplotlib.pyplot as plt

BLUE, ORANGE, DKRED = "#0072BD", "#D95319", "#A2142F"
PEAK_XY = (0.00055, 6.6)        # left of the onset line, which ran through the label at 2.6e-3


def make(ckpt=".", out="friction_boundary.pdf"):
    sl = json.load(open(f"{ckpt}/share_law_results.json"))
    fr = json.load(open(f"{ckpt}/friction_boundary_results.json"))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(174 / 25.4, 62 / 25.4))

    # (a) conservative cubic
    mu = [r["mu"] for r in sl["records"]]
    cbar = [r["cbar_over_ctrue"] for r in sl["records"]]
    renv = [r["r_env"] for r in sl["records"]]
    axa.axhline(1.0, color=META_GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    # Panel (a) plots cbar_over_ctrue, the ENERGY-WEIGHTED MEAN of the modal damping
    # coefficients, not a fitted decay rate. With the uniform damping of this system
    # it is unity identically (share_law_results.json: all_equal = True), so the label
    # and the annotation must not present it as a measured agreement.
    axa.plot(mu, cbar, "o-", color=BLUE, ms=3.4, lw=1.3,
             label=r"energy-weighted mean, $\bar c/c_{\rm true}$")
    axa.plot(mu, renv, "s-", color=ORANGE, ms=3.2, lw=1.3,
             label=r"single mode, $c_{\rm env}/c_{\rm true}$")
    axa.set_xscale("log")
    axa.set_xlabel(r"$\mu = k_3 A^2$")
    axa.set_ylabel(r"identified $/$ true")
    axa.set_ylim(0.38, 1.10)
    axa.set_title(r"Conservative: one value exists; the single-mode fit misses it",
                  fontsize=8)
    axa.legend(loc="lower left", frameon=False, fontsize=7,
               borderaxespad=0.15, handlelength=1.6)
    axa.text(mu[2], 1.035, r"unity by construction: uniform damping", fontsize=7,
             color=BLUE, ha="left", va="center")

    # (b) hysteretic Bouc-Wen
    amp = [r["A"] for r in fr["records"]]
    slip = [r["slip_ratio"] for r in fr["records"]]
    gtot = [r["gamma_tot_over_c"] for r in fr["records"]]
    genv = [r["r_env_over_c"] for r in fr["records"]]
    axb.axhline(1.0, color=META_GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    axb.plot(amp, gtot, "o-", color=BLUE, ms=3.4, lw=1.3,
             label=r"total energy, $\gamma_{\rm tot}/c_{\rm true}$")
    axb.plot(amp, genv, "s-", color=ORANGE, ms=3.2, lw=1.3,
             label=r"single mode, $c_{\rm env}/c_{\rm true}$")
    axb.set_xscale("log")
    axb.set_xlabel(r"forcing amplitude $A$")
    axb.set_ylabel(r"identified $/$ true")
    axb.set_ylim(0.0, 7.6)
    axb.set_title(r"Dissipative: no single value exists", fontsize=8)
    axb.legend(loc="upper left", frameon=False, fontsize=7,
               borderaxespad=0.15, handlelength=1.6)
    # macroslip sets in between the last amplitude with slip ratio < 1 and
    # the first with slip ratio ~ 1
    k = max(i for i, s in enumerate(slip) if s < 0.9)
    xc = (amp[k] * amp[k + 1]) ** 0.5
    axb.axvline(xc, color=META_GREY, lw=0.7, ls=(0, (1, 2)), zorder=1)
    axb.text(xc / 1.6, 0.35, "microslip", fontsize=7, color=META_GREY,
             ha="right")
    axb.text(xc * 1.6, 0.35, "macroslip", fontsize=7, color=META_GREY)
    i = max(range(len(gtot)), key=lambda kk: gtot[kk])
    axb.annotate(rf"${gtot[i]:.2f}$", xy=(amp[i], gtot[i]),
                 xytext=PEAK_XY, fontsize=7, color=DKRED,
                 arrowprops=dict(arrowstyle="->", color=DKRED, lw=0.6,
                                 shrinkA=1, shrinkB=3))

    for ax, L in ((axa, "a"), (axb, "b")):
        panel_letter(ax, L)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    return fig
