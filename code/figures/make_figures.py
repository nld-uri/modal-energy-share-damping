"""Build every figure in the paper from the stored checkpoints.

    python code/figures/make_figures.py [--ckpt DIR] [--out DIR] [NAME ...]

Each fig_*.py module exposes make(ckpt, out) and reads only the JSON files in
the checkpoint directory, so figures can never show a number that is not on
disk.  Style and palette are set here rather than in the modules, so that all
figures share one definition; the modules take the palette names and
panel_letter from builtins.

Requires a LaTeX installation (text.usetex=True): the figures are typeset in
Computer Modern so that symbols match the surrounding document.
"""
import argparse
import builtins
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- palette -----------------------------------------------------------------
# MATLAB's default ColorOrder, with two substitutions for colour-vision
# deficiency: MATLAB's purple and yellow are replaced by a darker plum and a
# dark green, which stay distinguishable under simulated deuteranopia.
BLUE = "#0072BD"
ORANGE = "#D95319"
PLUM = "#67355C"
DKRED = "#A2142F"
DGREEN = "#055432"
LBLUE = "#4DBEEE"
META_GREY = "#7F7F7F"

FIGURES = ["biasmap", "similarity", "bandwidth", "sidebands", "friction",
           "noise_sensing", "floquet", "law"]
OUTNAME = {"biasmap": "damping_bias_map.pdf", "similarity": "similarity.pdf",
           "bandwidth": "bandwidth.pdf", "sidebands": "sideband_recipe.pdf",
           "friction": "friction_boundary.pdf",
           "noise_sensing": "noise_sensing.pdf",
           "floquet": "floquet_stability.pdf",
           "law": "damping_law.pdf"}


def apply_style(sizes=(9, 8, 7)):
    """Publication style: real LaTeX text, Computer Modern, no top/right spines."""
    base, legend, tick = sizes
    # font.family is left at "sans-serif" deliberately.  Under text.usetex the
    # glyphs come from font.serif (Computer Modern Roman) in either case, but
    # matplotlib takes its text-extent estimates -- and hence tick-label
    # density and tight-bbox size -- from the family.  The manually placed
    # callout coordinates in the fig_*.py modules were tuned against these
    # metrics; changing the family to "serif" shifts them by a few points.
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": legend,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "axes.prop_cycle": plt.cycler(
            color=[BLUE, ORANGE, PLUM, DKRED, DGREEN, LBLUE, META_GREY]),
    })


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    """Bold panel letter outside the top-left corner of the axes."""
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


def export(names, ckpt, out):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    for nm in ("BLUE", "ORANGE", "PLUM", "DKRED", "DGREEN", "LBLUE",
               "META_GREY", "panel_letter"):
        setattr(builtins, nm, globals()[nm])
    apply_style()
    os.makedirs(out, exist_ok=True)
    for name in names:
        mod = importlib.import_module(f"fig_{name}")
        target = os.path.join(out, OUTNAME[name])
        mod.make(ckpt=ckpt, out=target)
        plt.close("all")
        print(f"  {OUTNAME[name]}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", default=None)
    p.add_argument("--ckpt", default=os.path.join(root, "checkpoints"))
    p.add_argument("--out", default=os.path.join(root, "figures"))
    a = p.parse_args()
    export(a.names or FIGURES, a.ckpt, a.out)
