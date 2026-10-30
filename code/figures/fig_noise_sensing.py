"""Figure: the estimator, the sensor array, and the diagnostic that finds it.

Reads only noise_sensing_results.json; recomputes nothing.

(a) error against analysis-window length, for the log-linear fit and the
    floor-aware fit of Eq. (floorfit) on the SAME records: the growth with
    window length belongs to the estimator.
(b) error against sensor count, both estimators: spatial aliasing, and the
    disagreement between two estimators that a test can observe.
(c) the fitted noise floor against sensor count, with the true measurement
    pedestal marked: the diagnostic that identifies an aliased array without
    knowing the damping coefficient.
"""
import collections
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BLUE, ORANGE, PLUM, DKRED, DGREEN = (
    "#0072BD", "#D95319", "#67355C", "#A2142F", "#055432")
OPT_NOTE_XY = (0.792981, 0.945496)     # hand-adjusted, axes fraction
M8_XY = (12.6, 0.017)              # data; right of the steep 8->12 descent, which crossed the label at 10.4
FLOOR = 3e-6  # plotting floor: errors below this are at solver noise


def _mean_by(rows, key, by):
    d = collections.defaultdict(list)
    for r in rows:
        d[r[by]].append(r[key])
    return {k: float(np.mean(v)) for k, v in d.items()}


def make(ckpt=".", out="noise_sensing.pdf"):
    ns = json.load(open(f"{ckpt}/noise_sensing_results.json"))
    W, S = ns["window"], ns["sensing"]

    err = {}
    for key, tag in (("gamma_sub_over_c", "floor"),
                     ("gamma_sub_naive_over_c", "naive")):
        agg = collections.defaultdict(list)
        for w in W:
            agg[(w["snr_db"], w["span"])].append(abs(w[key] - 1.0))
        err[tag] = {k: float(np.mean(v)) for k, v in agg.items()}
    spans = sorted({w["span"] for w in W})
    # 20 dB and noise-free only. The intermediate ratios lie on top of the
    # noise-free curve under the floor-aware fit (0.00442 against 0.00441 at
    # 40 dB), so plotting them adds legend entries with no visible line; the
    # full grid is in the table.
    snrs = [20.0, None]
    color = {20.0: DKRED, None: META_GREY}
    label = {None: "noiseless"}

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(174 / 25.4, 58 / 25.4))

    # One legend entry per plotted curve. A style key drawn in a colour that
    # appears nowhere in the panel (a black "log-linear" sample against red and
    # purple data) is a legend that does not match its figure.
    for s in snrs:
        nm = label[s] if s in label else rf"${s:.0f}$ dB"
        axa.plot(spans, [max(err["naive"][(s, sp)], FLOOR) for sp in spans],
                 "s--", color=color[s], ms=3.0, lw=1.0, zorder=2,
                 label=rf"{nm}, log-linear")
    for s in snrs:
        nm = label[s] if s in label else rf"${s:.0f}$ dB"
        axa.plot(spans, [max(err["floor"][(s, sp)], FLOOR) for sp in spans],
                 "o-", color=color[s], ms=3.2, lw=1.4, zorder=3,
                 label=rf"{nm}, floor-aware")
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xticks(spans)
    axa.set_xticklabels([f"${int(s)}$" for s in spans])
    axa.xaxis.set_minor_locator(mticker.NullLocator())
    axa.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    axa.yaxis.set_minor_formatter(mticker.NullFormatter())
    axa.set_xlabel(r"analysis window (time units)")
    axa.set_ylabel(r"$|\gamma_S/c_{\rm true}-1|$")
    axa.set_title("Growth with window length\nis the estimator", fontsize=8,
                  loc="left")
    # Lower left sat on the noiseless log-linear curve at 1200; upper left is empty
    # (every curve is below 1e-2 for windows up to 1200).
    axa.legend(loc="upper left", frameon=False, fontsize=6.2, ncol=1,
               borderaxespad=0.1, handlelength=1.8, handletextpad=0.4,
               labelspacing=0.22)

    m = sorted({r["n_sensors"] for r in S})
    ef = _mean_by(S, "gamma_sub_over_c", "n_sensors")
    en = _mean_by(S, "gamma_sub_naive_over_c", "n_sensors")
    axb.plot(m, [max(abs(en[k] - 1), FLOOR) for k in m], "s--", color=META_GREY,
             ms=3.0, lw=1.0, label="log-linear", zorder=2)
    axb.plot(m, [max(abs(ef[k] - 1), FLOOR) for k in m], "o-", color=BLUE,
             ms=3.6, lw=1.4, label="floor-aware", zorder=3)
    j = int(np.argmax([abs(ef[k] - 1) for k in m]))
    axb.plot([m[j]], [abs(ef[m[j]] - 1)], "o", color=DKRED, ms=5.0, zorder=4)
    axb.set_yscale("log")
    axb.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    axb.yaxis.set_minor_formatter(mticker.NullFormatter())
    axb.set_xticks(m)
    axb.set_xlabel(r"number of sensors $m$")
    axb.set_ylabel(r"$|\gamma_S/c_{\rm true}-1|$")
    axb.set_title("Aliasing, not channel count", fontsize=8, loc="left")
    # Lower left sits on the descending branch of both curves; upper right is the
    # one empty corner of this panel (both curves are below 5e-3 at m >= 12).
    axb.legend(loc="upper right", frameon=False, fontsize=6.5,
               borderaxespad=0.15, handlelength=1.6, labelspacing=0.25)
    axb.annotate(r"$m=8$ aliases the" "\n" r"mode shape", xy=(m[j], abs(ef[m[j]] - 1)),
                 xytext=M8_XY, fontsize=7, color=DKRED,
                 arrowprops=dict(arrowstyle="->", color=DKRED, lw=0.6,
                                 shrinkA=1, shrinkB=3))

    fl = _mean_by(S, "floor_rel", "n_sensors")
    axc.plot(m, [max(fl[k], 1e-16) for k in m], "o-", color=ORANGE, ms=3.6,
             lw=1.4, zorder=3)
    noise60 = float(np.mean([r["floor_rel"] for r in ns["noise"]
                             if r["snr_db"] == 60.0]))
    axc.axhline(noise60, color=META_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axc.set_yscale("log")
    axc.set_xticks(m)
    axc.set_xlabel(r"number of sensors $m$")
    axc.set_ylabel(r"fitted floor $b/E_0$")
    axc.set_title("The floor flags the aliased\narrays", fontsize=8, loc="left")
    # At 30 % height on the right the label lay across the orange curve. The empty
    # region of this panel is the bottom left, immediately above the dashed floor
    # the label names, where the curve is four orders of magnitude higher.
    axc.annotate("true measurement floor\n" r"at $60$ dB", xy=(m[-2], noise60),
                 xytext=(0.03, 0.045), textcoords="axes fraction", ha="left",
                 va="bottom", fontsize=7, color=META_GREY)

    for ax, L in ((axa, "a"), (axb, "b"), (axc, "c")):
        panel_letter(ax, L, dx=-0.30)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    return fig
