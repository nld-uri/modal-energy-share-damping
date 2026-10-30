"""Figure: the three estimates across the similarity parameter.

Reads only bias_map_results.json; recomputes nothing. (The original version of
this figure recomputed the sweep; it now reads the stored records, so the
figure and the manuscript audit cannot disagree.)

The "identity" label is set clear of the dashed line, per the hand adjustment
made on the typeset figure.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# META_GREY and panel_letter are shared style objects. make_figures.py puts them
# in builtins before importing this module; run on its own, take them from there.
try:
    META_GREY, panel_letter
except NameError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_figures as _mf
    _mf.apply_style()
    META_GREY, panel_letter = _mf.META_GREY, _mf.panel_letter

BLUE, ORANGE, PLUM, DKRED = "#0072BD", "#D95319", "#67355C", "#A2142F"
IDENTITY_XY = (0.62, 0.447)     # hand-adjusted: below the dashed line


def make(ckpt=".", out="damping_bias_map.pdf"):
    bm = json.load(open(f"{ckpt}/bias_map_results.json"))
    cells = {}
    for c in bm["grid"] + bm["threshold_sweep"]:
        m = c["k3"] * c["A"] ** 2
        if m > 0:
            cells.setdefault(round(m, 14), c)
    mu = np.array(sorted(cells))
    D = [cells[m] for m in mu]
    env = np.array([c["r_env"] for c in D])
    gcv = np.array([c["r_gcvd"] for c in D])
    flx = np.array([c["r_flux"] for c in D])
    shr = np.array([c["e_frac"] for c in D])
    k = int(np.argmin(shr))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(174 / 25.4, 70 / 25.4))

    axa.axhline(1.0, color="k", lw=0.7, ls=(0, (4, 3)), zorder=1)
    axa.axvspan(4.0e-4, 5.0e-4, color=META_GREY, alpha=0.25, lw=0, zorder=0)
    axa.plot(mu, gcv, "-s", color=ORANGE, ms=3.2, lw=1.0, zorder=3,
             label=r"GCVD state relation")
    axa.plot(mu, flx, "-^", color=PLUM, ms=3.2, lw=1.0, zorder=3,
             label=r"energy-flux prediction")
    axa.plot(mu, env, "-o", color=BLUE, ms=4.0, lw=1.8, zorder=4,
             label=r"envelope fit (reference)")
    axa.set_xscale("log")
    axa.set_xlabel(r"similarity parameter $\mu=k_3A^2$")
    axa.set_ylabel(r"$c_{\mathrm{ident}}/c_{\mathrm{true}}$")
    axa.set_title(r"Three estimates agree below the onset," "\n"
                  r"and separate above it", loc="left")
    axa.text(5.2e-4, 2.42, r"onset", color="#555555", ha="left", va="top")
    axa.legend(loc="lower left", frameon=False, handlelength=1.6,
               borderaxespad=0.3, labelspacing=0.25)
    axa.margins(x=0.06)
    axa.set_ylim(0.30, 2.52)

    lo, hi = 0.28, 1.05
    band = shr < 0.99
    axb.plot([lo, hi], [lo, hi], color="k", lw=0.7, ls=(0, (4, 3)), zorder=1)
    axb.plot(shr[~band], env[~band], "o", color=BLUE, ms=4.0, mfc="white",
             mew=1.2, zorder=3)
    axb.plot(shr[band], env[band], "o", color=BLUE, ms=4.5, zorder=4)
    axb.annotate(r"relation breaks down" "\n" r"($\mu=2\times10^{-3}$)",
                 xy=(shr[k], env[k]), xytext=(0.43, 1.30), color=DKRED,
                 arrowprops=dict(arrowstyle="-", color=DKRED, lw=0.7),
                 va="center")
    axb.plot(shr[k], env[k], "o", color=DKRED, ms=4.5, zorder=5)
    axb.set_xlabel(r"modal energy share $E_m/E_{\mathrm{tot}}$ in the window")
    axb.set_ylabel(r"$c_{\mathrm{ident}}/c_{\mathrm{true}}$")
    axb.set_title(r"Recovered damping tracks the mode's" "\n"
                  r"share of the window energy", loc="left")
    axb.text(1.01, 0.80, r"single-mode" "\n" r"records", color=BLUE,
             ha="right", va="top")
    # Rotation is NOT a fixed number: the dashed guide is y = x in data units, so
    # its on-screen angle depends on the axes box and the two limit ranges, which
    # differ here. It is set from the data transform after tight_layout below, so
    # the label and the line always share an angle.
    id_txt = axb.text(*IDENTITY_XY, r"identity", color="k", ha="center",
                      rotation_mode="anchor")
    axb.set_xlim(lo, hi)
    axb.set_ylim(0.32, 1.62)

    for ax, L in ((axa, "a"), (axb, "b")):
        panel_letter(ax, L, case="lower")
    fig.tight_layout(w_pad=4.0)
    _p0 = axb.transData.transform((lo, lo))
    _p1 = axb.transData.transform((hi, hi))
    id_txt.set_rotation(np.degrees(np.arctan2(_p1[1] - _p0[1], _p1[0] - _p0[0])))
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "."
    make(ckpt=ckpt, out=sys.argv[2] if len(sys.argv) > 2 else "damping_bias_map.pdf")
