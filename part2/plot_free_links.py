# Figures 9 to 12 of the paper (Part 2), drawn from the saved tables of free_link_training.py and
# free_link_check_placement.py. Computes nothing new; needs only pandas and matplotlib.
# Style as in the figures of Part 1 (noise_ladder.py, plot_ladder.py).
#   python plot_free_links.py
# AI (LLM) assistance (Claude, Anthropic) was used in preparing this code. The framework, calculations
# and analysis are those of R. Sakidja, who takes full responsibility for the scientific content.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRE_TRAPPED, PRE_PART1 = "preflight_300_trapped", "preflight_300_old_rates"
CHK_TRAPPED = "results_free_links_trapped_gamma_sys_%s_check_placement"
CHK_PART1 = "results_free_links_oldrates_multistart_gamma_sys_%s_check_placement"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#ffffff"   # white background
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
ARMS = (("Clock, frame calibrated", "#eb6834", "s", "--", "Clock, frame calibrated"),
        ("Vertical links only", "#9a9892", "v", ":", "Structured bath, vertical links only"),
        ("Full map without the layer 2 bonds", "#8a5cd0", "^", ":", "Structured bath, without the bonds of layer 2"),
        ("Structured bath, ML then fit, three inputs", "#2a78d6", "o", "-", "Structured bath, full map"),
        ("Structured bath, exact (bound)", "#1baf7a", "D", "-.", "Structured bath, exact (bound)"))
LABELS = {"J_sb_L1": r"$J_{sb}(L,1)$", "J_sb_R2": r"$J_{sb}(R,2)$", "J_L1_12": r"$J_{L1}(1,2)$",
          "J_L12_13": r"$J_{L12}(1,3)$", "J_L12_14": r"$J_{L12}(1,4)$", "J_L12_24": r"$J_{L12}(2,4)$",
          "J_L12_25": r"$J_{L12}(2,5)$", "J_L2_34": r"$J_{L2}(3,4)$", "J_L2_45": r"$J_{L2}(4,5)$"}


def grid(ax):
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def log_ticks(ax):
    """On a logarithmic axis that spans less than two decades, label 1, 2 and 5 of every decade,
    so that every panel carries tick labels."""
    lo, hi = ax.get_ylim()
    if hi / lo < 100:
        ticks = [m * 10.0 ** e for e in range(-6, 1) for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
        if len(ticks) >= 2:
            ax.set_yticks(ticks)
            ax.set_yticklabels(["%g" % t for t in ticks])
            ax.minorticks_off()


def periods(ax, tc):
    ax.set_xscale("log")
    ax.set_xticks(tc)
    ax.set_xticklabels(["%g" % t for t in tc])
    ax.minorticks_off()


def legend(fig, ax, ncol, bottom):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, bottom, 1, 1))


def figure_9():
    a = pd.read_csv(os.path.join(PRE_PART1, "tableF1_preflight_identifiability.csv")).set_index("Parameter")
    b = pd.read_csv(os.path.join(PRE_TRAPPED, "tableF1_preflight_identifiability.csv")).set_index("Parameter")
    names = list(LABELS)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for df, col, lab, c, mk, ls in ((a, "bound_balanced", "Balanced input, rates of Part 1", "#eb6834", "s", "--"),
                                    (b, "bound_balanced", "Balanced input, trapped rates", "#8a5cd0", "^", ":"),
                                    (a, "bound_three_inputs", "Three inputs, rates of Part 1", "#1baf7a", "D", "-."),
                                    (b, "bound_three_inputs", "Three inputs, trapped rates", "#2a78d6", "o", "-")):
        ax.plot(x, df.loc[names, col], linestyle="none", color=c, marker=mk, ms=8, markeredgecolor=SURFACE, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], fontsize=9)
    ax.set_ylabel("Bound, in units of the range")
    ax.set_title("Bound on every coupling", fontsize=11)
    grid(ax)
    legend(fig, ax, 2, 0.13)
    fig.savefig("figure_9_bound_on_every_coupling.png", dpi=200)
    print("Saved figure_9_bound_on_every_coupling.png")


def figure_10():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), squeeze=False, sharex=True)
    for r, g in enumerate(("0.0",)):
        d = pd.read_csv(os.path.join(CHK_TRAPPED % g, "tableP1_round_free_links.csv"))
        tc = sorted(d.T_clock.unique())
        for c, (col, name) in enumerate((("p_erase", "Erasure"), ("p_pauli_unflagged", "Unflagged Pauli error"))):
            ax = axes[r, c]
            for arm, colr, mk, ls, lab in ARMS:
                s = d[d.Arm == arm].set_index("T_clock").loc[tc]
                ax.plot(tc, s[col], ls, color=colr, marker=mk, ms=6, lw=2, markeredgecolor=SURFACE, label=lab)
            if c == 1:
                ax.set_yscale("log")
                log_ticks(ax)
            periods(ax, tc)
            ax.set_ylabel(name)
            ax.set_title(name, fontsize=10.5)
            grid(ax)
            ax.set_xlabel("Clock period T")
    fig.suptitle("One round per data qubit with every link free", fontsize=12)
    legend(fig, axes[0, 0], 3, 0.13)
    fig.savefig("figure_10_round_free_links.png", dpi=200)
    print("Saved figure_10_round_free_links.png")


def figure_11():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), squeeze=False, sharex=True)
    for r, g in enumerate(("0.0",)):
        d = pd.read_csv(os.path.join(CHK_TRAPPED % g, "tableP2_surface_code_free_links.csv"))
        d = d[d.Distance == d.Distance.max()]
        tc = sorted(d.T_clock.unique())
        for c, (basis, title) in enumerate((("Z", "Bit flip memory"), ("X", "Phase flip memory"))):
            ax, sub = axes[r, c], d[d.Memory_basis == basis]
            floor = 0.5 / sub.Shots.iloc[0]
            for arm, colr, mk, ls, lab in ARMS:
                s = sub[sub.Arm == arm].set_index("T_clock").loc[tc]
                ax.plot(tc, np.maximum(s.LER, floor), ls, color=colr, marker=mk, ms=6, lw=2,
                        markeredgecolor=SURFACE, label=lab)
            ax.set_yscale("log")
            log_ticks(ax)
            periods(ax, tc)
            ax.set_ylabel("Logical error rate")
            ax.set_title(title, fontsize=10.5)
            grid(ax)
            ax.set_xlabel("Clock period T")
    fig.suptitle("Logical error rate at distance 7 with every link free", fontsize=12)
    legend(fig, axes[0, 0], 3, 0.13)
    fig.savefig("figure_11_logical_error_free_links.png", dpi=200)
    print("Saved figure_11_logical_error_free_links.png")


def figure_12():
    arm = "Structured bath, ML then fit, three inputs"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), squeeze=False, sharey=True)
    for folder, colr, mk, ls, lab in ((CHK_PART1 % "0.0", "#eb6834", "s", "--", "Full map, far field rates of Part 1"),
                                      (CHK_TRAPPED % "0.0", "#2a78d6", "o", "-", "Full map, trapped far field rates")):
        d = pd.read_csv(os.path.join(folder, "tableP2_surface_code_free_links.csv"))
        d = d[(d.Distance == d.Distance.max()) & (d.Arm == arm)]
        tc = sorted(d.T_clock.unique())
        for c, (basis, title) in enumerate((("Z", "Bit flip memory"), ("X", "Phase flip memory"))):
            ax, s = axes[0, c], d[d.Memory_basis == basis].set_index("T_clock").loc[tc]
            ax.plot(tc, np.maximum(s.LER, 0.5 / s.Shots.iloc[0]), ls, color=colr, marker=mk, ms=6, lw=2,
                    markeredgecolor=SURFACE, label=lab)
            ax.set_yscale("log")
            periods(ax, tc)
            ax.set_xlabel("Clock period T")
            ax.set_title(title, fontsize=10.5)
            grid(ax)
    axes[0, 0].set_ylabel("Logical error rate")
    fig.suptitle("Far field rates and the check at distance 7", fontsize=12)
    legend(fig, axes[0, 0], 2, 0.1)
    fig.savefig("figure_12_far_field_rates.png", dpi=200)
    print("Saved figure_12_far_field_rates.png")


if __name__ == "__main__":
    for f in (figure_9, figure_10, figure_11, figure_12):
        f()
