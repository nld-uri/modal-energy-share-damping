"""Figure: why the energy share is a proxy and not the variable.

Reads only bandwidth_axis_results.json and onset_mechanism_results.json
(for the modal frequencies); recomputes nothing.
"""
import collections
import json
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, DKRED = "#0072BD", "#D95319", "#A2142F"
BANDREACH_DXY = (0.056, 1.209)  # hand-adjusted
# Panel (b), the share-against-bias scatter, was removed: it is the same
# plot as panel (b) of the damping-law figure, on a subset of the records.
BETA_XY = (0.339, 0.318)        # hand-adjusted


def make(ckpt=".", out="bandwidth.pdf"):
    ba = json.load(open(f"{ckpt}/bandwidth_axis_results.json"))
    om = json.load(open(f"{ckpt}/onset_mechanism_results.json"))
    rec = ba["records"]

    fig, axa = plt.subplots(1, 1, figsize=(110 / 25.4, 62 / 25.4))

    # (a) share collapses with bandwidth, the identified damping does not
    by = collections.defaultdict(list)
    for r in rec:
        by[r["beta"]].append(r)
    betas = sorted(by)
    for key, color, mark, lab in (
            ("share_mean", DKRED, "s", r"energy share $E_m/E_{\rm tot}$"),
            ("r_env", BLUE, "o", r"identified $c_{\rm ident}/c_{\rm true}$")):
        med = [float(np.median([r[key] for r in by[b]])) for b in betas]
        axa.plot([r["beta"] for r in rec], [r[key] for r in rec], mark,
                 color=color, ms=2.4, alpha=0.30, mew=0, zorder=2)
        axa.plot(betas, med, mark + "-", color=color, ms=3.6, lw=1.3,
                 label=lab, zorder=3)
    axa.axhline(1.0, color=META_GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    # NB: the stored JSON key keeps its original spelling
    b_in = min(b for b in betas if by[b][0]["n_neighbours_in_band"] > 0)
    axa.axvline(b_in, color=META_GREY, lw=0.7, ls=(0, (1, 2)), zorder=1)
    axa.set_xlabel(r"excitation band halfwidth $\beta$")
    axa.set_ylabel(r"value")
    axa.set_ylim(0.0, 1.52)
    axa.set_title(r"Share and damping move independently", fontsize=8)
    axa.legend(loc="upper left", frameon=False, fontsize=6.5,
               borderaxespad=0.15, handlelength=1.5)
    axa.annotate(r"band reaches $\omega_{6},\omega_{8}$",
                 xy=(b_in, BANDREACH_DXY[1]), xytext=(b_in + BANDREACH_DXY[0], BANDREACH_DXY[1]), fontsize=7,
                 color=META_GREY, va="center")

    # (b) the proxy, tested directly
    sh = [r["share_mean"] for r in rec]
    ci = [r["r_env"] for r in rec]
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    return fig
