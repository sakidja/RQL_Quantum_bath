# Figures 6 and 7 of the paper (noise ladder), drawn from the saved tables of noise_ladder.py.
# Computes nothing new; needs only pandas and matplotlib. Run in the folder of the ladder results:
#   python plot_ladder_final.py
#   Figure 6  figure_6_ladder_parameters.png      R2 of the six bath parameters on each rung, 3 by 2 panels
#   Figure 7  figure_7_ladder_logical_error.png   logical error rate at distance 7, clock periods 2 and 4
# AI (LLM) assistance (Claude, Anthropic) was used in preparing this code. The framework, calculations
# and analysis are those of R. Sakidja, who takes full responsibility for the scientific content.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.environ.get("SQB_LADDER", "results_v3_ladder_fair_split_gamma_sys_0.0")
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#ffffff"   # white background
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
L1 = pd.read_csv(os.path.join(D, "tableL1_inference_on_ladder.csv"))
L3 = pd.read_csv(os.path.join(D, "tableL3_surface_code_on_ladder.csv"))
RUNGS = list(dict.fromkeys(L1.Rung))
TICKS = ["Exact" if r == "exact" else r.replace("rung", "Rung") for r in RUNGS]
X = np.arange(len(RUNGS))
SYMBOL = {"J_sb": r"$J_{sb}$", "J_L1": r"$J_{L1}$", "J_L12": r"$J_{L12}$", "J_L2": r"$J_{L2}$",
          "gamma_L1": r"$\gamma_{L1}$", "gamma_L2": r"$\gamma_{L2}$"}


def grid(ax):
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def figure_6():
    methods = (("ML, trained on exact signals", "#9a9892", "s", "--", "Regression trained on exact signals"),
               ("ML, trained on noisy signals", "#8a5cd0", "^", ":", "Regression trained on noisy signals"),
               ("ML (trained on noisy signals), then fit", "#2a78d6", "o", "-", "Trained on noisy signals, then fit"))
    lo = np.floor(10 * L1.R2.min()) / 10              # the axis starts just below the lowest value
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 8.6), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), SYMBOL):
        sub = L1[L1.Parameter == p].set_index(["Method", "Rung"]).R2
        for key, col, mk, ls, lab in methods:
            ax.plot(X, [sub[(key, r)] for r in RUNGS], ls, color=col, marker=mk, ms=6, lw=2, markeredgecolor=SURFACE, label=lab)
        ax.set_ylim(lo, 1.04)
        ax.set_title(SYMBOL[p], fontsize=11)
        ax.set_xticks(X)
        ax.set_xticklabels(TICKS)
        grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("R² on the unseen baths")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=9.5, frameon=False)
    fig.suptitle("Bath parameters recovered on the noise ladder", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig("figure_6_ladder_parameters.png", dpi=200)
    print("Saved figure_6_ladder_parameters.png")


def figure_7():
    arms = (("Clock, frame calibrated", "#eb6834", "s", "--", "Clock, frame calibrated"),
            ("Structured bath, ML inferred, balanced check", "#8a5cd0", "^", ":", "Structured bath, regression alone"),
            ("Structured bath, ML then fit, balanced check", "#2a78d6", "o", "-", "Structured bath, regression then fit"),
            ("Structured bath, exact at calibration, balanced check (bound)", "#1baf7a", "D", "-.", "Structured bath, exact (bound)"))
    d = L3[L3.Distance == L3.Distance.max()]
    periods = sorted(d.T_clock.unique())
    fig, axes = plt.subplots(len(periods), 2, figsize=(9.6, 3.7 * len(periods)), sharex=True, squeeze=False)
    for r, tc in enumerate(periods):
        for c, (basis, name) in enumerate((("Z", "Bit flip memory"), ("X", "Phase flip memory"))):
            ax = axes[r, c]
            sub = d[(d.T_clock == tc) & (d.Memory_basis == basis)].set_index(["Arm", "Rung"]).LER
            vals = []
            for key, col, mk, ls, lab in arms:
                y = [sub[(key, rr)] for rr in RUNGS]
                vals += y
                ax.plot(X, y, ls, color=col, marker=mk, ms=6, lw=2, markeredgecolor=SURFACE, label=lab)
            ax.set_yscale("log")
            ax.set_ylim(min(vals) / 1.5, max(vals) * 1.5)
            ax.set_title("%s, clock period %g" % (name, tc), fontsize=10.5)
            ax.set_ylabel("Logical error rate")
            ax.set_xticks(X)
            ax.set_xticklabels(TICKS)
            grid(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9.5, frameon=False)
    fig.suptitle("Logical error rate at distance 7 on the noise ladder", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("figure_7_ladder_logical_error.png", dpi=200)
    print("Saved figure_7_ladder_logical_error.png")


if __name__ == "__main__":
    figure_6()
    figure_7()
