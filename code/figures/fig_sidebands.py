"""Figure: where the energy goes, and what that implies for instrumentation.

Reads only stored JSONs (onset_mechanism_results.json,
recovery_recipe_results.json); recomputes nothing.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, PLUM, DKRED = "#0072BD", "#D95319", "#67355C", "#A2142F"
SUBSETS = ["carrier only", "carrier +/- 1", "carrier +/- 2",
           "carrier +/- 4", "all modes"]
NICE = {"carrier only": r"$\{7\}$", "carrier +/- 1": r"$\{6,7,8\}$",
        "carrier +/- 2": r"$\{5,\dots,9\}$",
        "carrier +/- 4": r"$\{3,\dots,11\}$", "all modes": r"all $24$"}


def make(ckpt=".", out="sideband_recipe.pdf"):
    om = json.load(open(f"{ckpt}/onset_mechanism_results.json"))
    rr = json.load(open(f"{ckpt}/recovery_recipe_results.json"))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(174 / 25.4, 62 / 25.4))

    # (a) energy destination as mu grows
    dest = sorted(om["destinations"], key=lambda d: d["mu"])
    mus = [d["mu"] for d in dest]
    carrier, side = [], []
    for d in dest:
        sh = {t["mode"]: t["share"] for t in d["top"]}
        carrier.append(sh[7])
        side.append(sh.get(5, 0.0) + sh.get(9, 0.0))
    rest = [1.0 - c - s for c, s in zip(carrier, side)]
    x = np.arange(len(mus))
    axa.bar(x, carrier, 0.62, color=BLUE, label=r"carrier, mode $7$")
    axa.bar(x, side, 0.62, bottom=carrier, color=DKRED,
            label=r"sideband pair, modes $5$ and $9$")
    axa.bar(x, rest, 0.62, bottom=np.add(carrier, side), color=META_GREY,
            label="everything else")
    axa.set_xticks(x)
    axa.set_xticklabels([rf"${m*1e4:.0f}$" for m in mus])
    axa.set_xlabel(r"$\mu = k_3A^2$ (units of $10^{-4}$)")
    axa.set_ylabel(r"share of modal energy")
    axa.set_ylim(0, 1.32)
    axa.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axa.set_title(r"The carrier loses energy to one pair", fontsize=8)
    axa.legend(loc="upper left", frameon=False, fontsize=6.5,
               borderaxespad=0.15, handlelength=1.2, labelspacing=0.25)
    for xi, (c, s) in enumerate(zip(carrier, side)):
        axa.text(xi, c / 2, rf"${c:.2f}$", ha="center", va="center",
                 fontsize=6.5, color="white")
        if s > 0.05:
            axa.text(xi, c + s / 2, rf"${s:.2f}$", ha="center", va="center",
                     fontsize=6.5, color="white")

    # (b) what it costs to instrument less than the pair
    mean = [rr["gamma_summary"][s]["mean_abs_err"] for s in SUBSETS]
    worst = [rr["gamma_summary"][s]["worst_abs_err"] for s in SUBSETS]
    xb = np.arange(len(SUBSETS))
    axb.bar(xb, mean, 0.58, color=[DKRED, DKRED, BLUE, BLUE, BLUE],
            zorder=2)
    axb.plot(xb, worst, "_", color=ORANGE, ms=11, mew=1.4, zorder=3,
             label="worst record")
    axb.set_yscale("log")
    axb.set_xticks(xb)
    axb.set_xticklabels([NICE[s] for s in SUBSETS])
    axb.set_xlabel(r"modes entering the energy sum $S$")
    axb.set_ylabel(r"mean $|\gamma_S/c_{\rm true}-1|$")
    axb.set_ylim(5e-4, 3.0)
    axb.set_title(r"Adding the pair buys two decades", fontsize=8)
    axb.legend(loc="upper right", frameon=False, fontsize=7,
               borderaxespad=0.15, handlelength=1.0)
    axb.annotate(r"nearest neighbors" "\n" r"do not help",
                 xy=(1, mean[1]), xytext=(-0.45, 1.05), fontsize=7,
                 color=DKRED,
                 arrowprops=dict(arrowstyle="->", color=DKRED, lw=0.6,
                                 shrinkA=1, shrinkB=3))
    axb.annotate(rf"${mean[0]/mean[2]:.0f}\times$ better",
                 xy=(2, mean[2]), xytext=(2.05, 0.055), fontsize=7,
                 color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.6,
                                 shrinkA=1, shrinkB=3))

    for ax, L in ((axa, "a"), (axb, "b")):
        panel_letter(ax, L)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    return fig
