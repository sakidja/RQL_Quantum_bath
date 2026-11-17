# Figures 3, 4, 5 and 8 of the paper, drawn from the saved result tables of
# dual_rail_structured_bath.py, with one case only: no direct dephasing on the
# rails (gamma_sys = 0). Computes nothing; reads table2_round_channels.csv and
# table3_surface_code.csv and plots what is in them.
#
# Colours, markers, line styles, grid and white background are those of the
# figures of the paper:
#   amber  #eda100  down triangle  dotted    clock, uniform decoder
#   orange #eb6834  square         dashed    clock, frame calibrated
#   blue   #2a78d6  circle         solid     structured bath, ML then fit
#   green  #1baf7a  diamond        dash dot  structured bath, exact (bound)
#
#   SQB_ATTACH=one   python plot_results_one_case.py     -> Figures 3, 4, 5
#   SQB_ATTACH=split python plot_results_one_case.py     -> Figure 8 (and the
#                                                           two split companions)
#
# The result folder is found automatically; to name it directly:
#   SQB_RESULTS=results_v2_gamma_sys_0.0 SQB_ATTACH=one python plot_results_one_case.py
#
# Files written (same names as before, so they replace the old figures):
#   first attachment : figure_round_channels.png
#                      figure_logical_error_vs_clock.png
#                      figure_logical_error_vs_distance.png
#   split attachment : the same three names with _split
#
# Needs pandas and matplotlib only.
#
# AI (LLM) assistance (Claude, Anthropic) was used in preparing this code. The
# framework, calculations and analysis are those of R. Sakidja, who takes full
# responsibility for the scientific content.
# ======================================================================
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ATTACH = os.environ.get("SQB_ATTACH", "split")
TAG = "" if ATTACH == "one" else "_split"
T_DISTANCE = float(os.environ.get("SQB_T_DISTANCE", "4"))   # clock period of the distance figure

# Folders of the main run with gamma_sys = 0, in the order they are looked for.
CANDIDATES = {
    "one":   ["results_v2_gamma_sys_0.0", "results_v3_one_gamma_sys_0.0",
              "results_v2_gamma_sys_0", "results_v3_one_gamma_sys_0"],
    "split": ["results_v3_split_gamma_sys_0.0", "results_v3_split_gamma_sys_0"],
}

# arm, colour, marker, line style, label; the order is the order of the legend
ARMS = (
    ("Clock, frame calibrated, uniform decoder", "#eda100", "v", ":",
     "Clock, frame calibrated, uniform decoder"),
    ("Clock, frame calibrated", "#eb6834", "s", "--", "Clock, frame calibrated"),
    ("Structured bath, ML then fit, balanced check", "#2a78d6", "o", "-",
     "Structured bath, ML then fit, balanced check"),
    ("Structured bath, exact, balanced check (bound)", "#1baf7a", "D", "-.",
     "Structured bath, exact, balanced check (bound)"),
)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "white"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def folder():
    named = os.environ.get("SQB_RESULTS")
    if named:
        if not os.path.isdir(named):
            sys.exit(f"SQB_RESULTS={named} is not a folder here.")
        return named
    for f in CANDIDATES[ATTACH]:
        if os.path.isdir(f):
            return f
    seen = sorted(glob.glob("results_*gamma_sys_0*"))
    sys.exit("No result folder of the main run with gamma_sys = 0 was found for "
             f"SQB_ATTACH={ATTACH}.\nLooked for: {', '.join(CANDIDATES[ATTACH])}\n"
             + ("Folders here that could be it: " + ", ".join(seen) if seen else "No results_*gamma_sys_0* folder here.")
             + "\nName the folder with SQB_RESULTS=<folder>.")


def load(f, name, columns):
    path = os.path.join(f, name)
    if not os.path.isfile(path):
        sys.exit(f"{path} is missing.")
    d = pd.read_csv(path)
    missing = [c for c in columns if c not in d.columns]
    if missing:
        sys.exit(f"{path} has no column {', '.join(missing)}. Columns found: {', '.join(d.columns)}")
    return d


def arms_present(d, path):
    have = set(d.Arm.unique())
    out = [a for a in ARMS if a[0] in have]
    missing = [a[0] for a in ARMS if a[0] not in have]
    if missing:
        print(f"   not in {path}, left out of the figure: {'; '.join(missing)}")
    if not out:
        sys.exit(f"None of the arms of the figure are in {path}. Arms found: {'; '.join(sorted(have))}")
    return out


def save(fig, out):
    """Write the figure, keeping any file already there under <name>.before_one_case.png."""
    if os.path.isfile(out):
        kept = out[:-4] + ".before_one_case.png"
        if not os.path.exists(kept):
            os.rename(out, kept)
            print(f"   the earlier {out} was kept as {kept}")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("   wrote", out)


def grid(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def log_ticks(ax):
    """Decades, plus 2 and 5 within a decade, so a range under one decade still carries labels."""
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2.0, 5.0), numticks=20))
    ax.yaxis.set_minor_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False,
                                                                 minor_thresholds=(10, 0.4)))


def fit_log(ax, values):
    """Axis range fitted to the data."""
    v = np.log10(np.array([x for x in values if x > 0]))
    pad = max(0.06 * (v.max() - v.min()), 0.05)
    ax.set_ylim(10 ** (v.min() - pad), 10 ** (v.max() + pad))


def fit_linear(ax, values):
    v = np.array(values, dtype=float)
    lo, hi = v.min(), v.max()
    pad = 0.08 * (hi - lo) if hi > lo else 0.1 * abs(hi) + 1e-9
    ax.set_ylim(max(0.0, lo - pad), hi + pad)


def period_axis(ax, xs):
    """Clock periods on the x axis. When they span a decade or more (1 to 50), the axis is
    logarithmic, so that the short periods are not squeezed into one edge."""
    if max(xs) / min(xs) >= 10:
        ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%g" % x for x in xs])
    ax.minorticks_off() if max(xs) / min(xs) >= 10 else None


def legend(fig, ax, bottom):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, bottom, 1, 1))


def figure_round(d, out):
    """Figure 3: erasure and unflagged Pauli error of one round against the clock period."""
    tc = sorted(d.T_clock.unique())
    arms = arms_present(d, out)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), squeeze=False)
    for c, (col, name) in enumerate((("p_erase", "Erasure"), ("p_pauli_unflagged", "Unflagged Pauli error"))):
        ax, seen = axes[0, c], []
        for arm, colr, mk, ls, lab in arms:
            s = d[d.Arm == arm].set_index("T_clock").loc[tc]
            ax.plot(tc, s[col], ls, color=colr, marker=mk, ms=6, lw=2, markeredgecolor=SURFACE, label=lab)
            seen += list(s[col])
        if c == 1:
            ax.set_yscale("log")
            fit_log(ax, seen)
            log_ticks(ax)
        else:
            fit_linear(ax, seen)
        period_axis(ax, tc)
        ax.set_xlabel("Clock period T")
        ax.set_ylabel(name)
        ax.set_title(name, fontsize=10.5)
        grid(ax)
    fig.suptitle("One round per data qubit", fontsize=12)
    legend(fig, axes[0, 0], 0.15)
    save(fig, out)


def figure_code(d, xcol, xs, xlabel, title, out):
    """Figures 4 and 8 (against the clock period) and Figure 5 (against the code distance)."""
    arms = arms_present(d, out)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), squeeze=False)
    seen = []
    for c, (basis, name) in enumerate((("Z", "Bit flip memory"), ("X", "Phase flip memory"))):
        ax, sub = axes[0, c], d[d.Memory_basis == basis]
        floor = 0.5 / sub.Shots.iloc[0]          # a point with zero counted errors is drawn at this floor
        for arm, colr, mk, ls, lab in arms:
            s = sub[sub.Arm == arm].set_index(xcol).loc[xs]
            y = np.maximum(s.LER, floor)
            ax.plot(xs, y, ls, color=colr, marker=mk, ms=6, lw=2, markeredgecolor=SURFACE, label=lab)
            seen += list(y)
        ax.set_yscale("log")
        log_ticks(ax)
        if xcol == "T_clock":
            period_axis(ax, xs)
        else:
            ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Logical error rate")
        ax.set_title(name, fontsize=10.5)
        grid(ax)
    for ax in axes[0]:                       # one range for both panels, labelled on both
        fit_log(ax, seen)
    fig.suptitle(title, fontsize=12)
    legend(fig, axes[0, 0], 0.15)
    save(fig, out)


def main():
    f = folder()
    print(f"SQB_ATTACH = {ATTACH}, results read from {f}/")
    d2 = load(f, "table2_round_channels.csv", ["T_clock", "Arm", "p_erase", "p_pauli_unflagged"])
    d3 = load(f, "table3_surface_code.csv", ["T_clock", "Arm", "Memory_basis", "Distance", "LER", "Shots"])
    figure_round(d2, f"figure_round_channels{TAG}.png")

    dmax = d3.Distance.max()
    a = d3[d3.Distance == dmax]
    figure_code(a, "T_clock", sorted(a.T_clock.unique()), "Clock period T",
                f"Logical error rate at distance {dmax}", f"figure_logical_error_vs_clock{TAG}.png")

    if T_DISTANCE not in set(d3.T_clock.unique()):
        sys.exit(f"Clock period {T_DISTANCE:g} is not in the table. Periods present: "
                 + ", ".join(f"{t:g}" for t in sorted(d3.T_clock.unique()))
                 + ". Choose one with SQB_T_DISTANCE.")
    b = d3[d3.T_clock == T_DISTANCE]
    figure_code(b, "Distance", sorted(b.Distance.unique()), "Code distance",
                f"Logical error rate against code distance, clock period {T_DISTANCE:g}",
                f"figure_logical_error_vs_distance{TAG}.png")


if __name__ == "__main__":
    main()
