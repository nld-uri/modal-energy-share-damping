"""Figure: the problem has one parameter.

Reads only bias_map_results.json; recomputes nothing.
"""
import collections
import json
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, PLUM, DKRED, DGREEN = (
    "#0072BD", "#D95319", "#67355C", "#A2142F", "#055432")
CALLOUT_XY = (0.04, 0.97)   # axes fraction; the earlier hand-tuned data
                            # coordinates belonged to the retired y quantity
YFLOOR = 1.0e-3             # so the k3 = 0 row is visible on a log y axis
MARK = {0.0: "o", 0.5: "s", 1.0: "^", 2.0: "D", 4.0: "v"}
COL = {0.0: META_GREY, 0.5: BLUE, 1.0: ORANGE, 2.0: DGREEN, 4.0: PLUM}
FLOOR = 2e-7  # so the k3 = 0 column is visible on a log axis


def make(ckpt=".", out="similarity.pdf"):
    bm = json.load(open(f"{ckpt}/bias_map_results.json"))
    g = bm["grid"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(174 / 25.4, 62 / 25.4))

    # The plotted quantity is the identified frequency shift, which varies by
    # a factor of about 300 across the grid. The damping ratio would be the more
    # direct choice but is unity to within 0.35 per cent everywhere below the
    # onset, so a collapse in it is not visible; the invariance of both is
    # quantified in the caption and stored in similarity_check.
    for ax, xkey, xlabel in ((axa, "mu", r"$\mu = k_3A^2$"),
                             (axb, "A", r"forcing amplitude $A$")):
        for k3 in sorted({c["k3"] for c in g}):
            cells = [c for c in g if c["k3"] == k3]
            x = [max(c["k3"] * c["A"] ** 2, FLOOR) if xkey == "mu" else c["A"]
                 for c in cells]
            order = np.argsort(x)
            # panel (a) is a locus, not a family of curves: joining points
            # within one k3 would suggest structure that the collapse denies
            ax.plot(np.array(x)[order],
                    np.maximum(np.array([abs(c["stiff_pct"]) for c in cells]),
                               YFLOOR)[order],
                    MARK[k3] + ("" if ax is axa else "-"), color=COL[k3],
                    ms=3.6, lw=1.0, label=rf"$k_3={k3:g}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylim(YFLOOR * 0.45, 1.6)

    axa.set_ylabel(r"identified frequency shift (\%)")
    axb.set_ylabel(r"identified frequency shift (\%)")
    axa.set_title(r"One curve in $\mu$", fontsize=8)
    axb.set_title(r"Five curves in amplitude", fontsize=8)
    axb.legend(loc="upper left", frameon=False, fontsize=6.5, ncol=2,
               borderaxespad=0.15, handlelength=1.4, columnspacing=0.9,
               labelspacing=0.25)

    # the invariant pairs: different (k3, A), identical mu, identical answer
    by = collections.defaultdict(list)
    for c in g:
        by[round(c["k3"] * c["A"] ** 2, 12)].append(c)
    pairs = {m: v for m, v in by.items() if len(v) > 1 and m > 0}
    sc = bm["similarity_check"]          # stored, not recomputed here
    spread_env, spread_gc = sc["max_spread_r_env"], sc["max_spread_r_gcvd"]
    for m, v in pairs.items():
        axa.plot([m], [abs(v[0]["stiff_pct"])], "o", ms=9, mfc="none", mew=0.9,
                 color=DKRED, zorder=4)
    mx = max(pairs)
    axa.annotate(r"$\lambda$-invariant pairs: two different"
                 "\n" r"$(k_3,A)$, one answer to $10^{-9}$",
                 xy=(mx, abs(pairs[mx][0]["stiff_pct"])), xytext=CALLOUT_XY,
                 textcoords="axes fraction", fontsize=7, color=DKRED,
                 ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=DKRED, lw=0.6,
                                 shrinkA=1, shrinkB=5))
    axa.text(FLOOR * 1.25, YFLOOR * 0.72, r"$k_3=0$: no shift", fontsize=7,
             color=META_GREY, va="top")

    for ax, L in ((axa, "a"), (axb, "b")):
        panel_letter(ax, L)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    return fig, spread_env, spread_gc
