"""Figure: the drift law against the share proxy, over both protocols.

Reads share_law_results.json and bandwidth_axis_results.json; recomputes
nothing. Every record is plotted: the |c_ident/c_true| < 5 screen that an
earlier version of this analysis applied removed the single hardest record
from the set, which is not a neutral thing for a figure about prediction
error to do.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, DKRED = "#0072BD", "#D95319", "#A2142F"


def make(ckpt=".", out="damping_law.pdf"):
    sl = json.load(open(f"{ckpt}/share_law_results.json"))
    ba = json.load(open(f"{ckpt}/bandwidth_axis_results.json"))
    st, bd = sl["records"], ba["records"]

    x_law = ([r["r_derived"] for r in bd] + [r["r_derived"] for r in st])
    y = ([r["r_env_trimmed"] for r in bd] + [r["r_env"] for r in st])
    x_shr = ([r["share_mean"] for r in bd] + [r["share_mean"] for r in st])
    nb = len(bd)

    err = np.abs(np.array(x_law) - np.array(y))
    r_law = float(np.corrcoef(x_law, y)[0, 1])
    err_s = np.abs(np.array(x_shr) - np.array(y))
    r_shr = float(np.corrcoef(x_shr, y)[0, 1])

    # Limits from the data, so that the record the earlier screen removed is
    # inside the axes rather than clipped out of a figure that claims to show
    # every record.
    lo = min(min(x_law), min(y), min(x_shr)) - 0.4
    hi = max(max(x_law), max(y), max(x_shr)) + 0.4
    LIM = (float(lo), float(hi))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(174 / 25.4, 78 / 25.4))
    for ax, x, ttl, xlab in (
            (axa, x_law, r"The drift law predicts every record",
             r"predicted $\bar c-\mathrm{d}\ln f/\mathrm{d}t$, over $c_{\rm true}$"),
            (axb, x_shr, r"The energy share alone does not",
             r"modal energy share $E_m/E_{\rm tot}$")):
        ax.plot(LIM, LIM, "k--", lw=0.7, dashes=(4, 3), zorder=1)
        ax.plot(x[:nb], y[:nb], "o", mfc="none", mew=0.9, ms=4.0,
                color=ORANGE, zorder=3, label="band-limited forcing")
        ax.plot(x[nb:], y[nb:], "o", ms=4.0, color=BLUE, zorder=4,
                label="resonant single tone")
        ax.set_xlim(*LIM)
        ax.set_ylim(*LIM)
        ax.set_aspect("equal")
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"identified $c_{\rm ident}/c_{\rm true}$")
        ax.set_title(ttl, fontsize=8)
    axa.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.0,
               borderaxespad=0.2, labelspacing=0.25, handletextpad=0.4)
    axa.text(0.97, 0.06, rf"$r={r_law:.5f}$" "\n" rf"worst error {err.max():.3f}",
             transform=axa.transAxes, ha="right", va="bottom", fontsize=7,
             color=BLUE)
    axb.text(0.05, 0.94, rf"$r={r_shr:.2f}$" "\n" rf"worst error {err_s.max():.1f}",
             transform=axb.transAxes, ha="left", va="top", fontsize=7,
             color=DKRED)

    for ax, L in ((axa, "a"), (axb, "b")):
        panel_letter(ax, L)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    return fig
