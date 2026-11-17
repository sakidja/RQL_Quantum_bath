# Figures 1, 9, 10 and 11 of the paper in the final regime, drawn from saved files only.
#   Figure 1   signal and FFT fingerprint of three unseen baths (lowest, median, highest J_sb),
#              from raw_signals_and_spectra.npz of the Part 1 run, first attachment
#   Figure 9   bound on every coupling, from tableF1 of the final Part 2 training run
#   Figure 10  one round with every link free, from tableP1 of the final check placement
#   Figure 11  logical error rate at distance 7 with every link free, from tableP2
# Computes nothing new: the spectrum of Figure 1 is the published log enhanced, normalized FFT
# of the stored signal. Run in the folder that holds plot_free_links.py:
#   python plot_final_figures.py
# AI (LLM) assistance (Claude, Anthropic) was used in preparing this code. The framework, calculations
# and analysis are those of R. Sakidja, who takes full responsibility for the scientific content.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_free_links as pf

PART1 = os.environ.get("SQB_PART1", "../part1/results_v3_final8_one_gamma_sys_0.0")
PART2 = os.environ.get("SQB_PART2", "results_free_links_final16h_gamma_sys_0.0")
N_TEST = 49


def figure_1():
    z = np.load(os.path.join(PART1, "raw_signals_and_spectra.npz"), allow_pickle=True)
    t, P, S = z["time"], z["true_parameters"][-N_TEST:], z["signal_xyz"][-N_TEST:]
    names = [str(n) for n in z["parameter_names"]]
    jsb, jl12 = P[:, names.index("J_sb")], P[:, names.index("J_L12")]
    order = np.argsort(jsb)
    picks = [order[0], order[len(order) // 2], order[-1]]
    freq = np.arange(len(t) // 2) / (len(t) * (t[1] - t[0]))
    fig, axes = plt.subplots(3, 2, figsize=(10, 7.6))
    for r, (k, col) in enumerate(zip(picks, ("#2a78d6", "#eb6834", "#1baf7a"))):
        x = S[k, 0].astype(float)                       # logical <X> of the balanced input
        raw = np.abs(np.fft.fft(x - x.mean())) / len(x)
        lg = np.log10(2 * raw[: len(x) // 2] + 1e-12)
        lg -= lg.min()
        lg /= lg.max() + 1e-12
        a, b = axes[r]
        a.plot(t, x, color=col, lw=1.8)
        a.set_xlim(0, 60)
        a.set_ylim(-1.05, 1.05)
        a.set_ylabel("Logical <X>")
        a.set_title("J_sb = %.3f, J_L12 = %.3f" % (jsb[k], jl12[k]), fontsize=10.5)
        show = freq <= 1.0
        b.plot(freq[show], lg[show], color=col, lw=1.8)
        b.set_xlim(0, 1.0)
        b.set_ylim(max(0.0, 0.98 * lg[show][1:].min()), 1.03)
        b.set_ylabel("Normalized log FFT")
        b.set_title("FFT fingerprint of <X>", fontsize=10.5)
        for ax in (a, b):
            pf.grid(ax)
    axes[2, 0].set_xlabel("Time")
    axes[2, 1].set_xlabel("Frequency")
    fig.suptitle("Calibration signal and FFT fingerprint of three unseen baths", fontsize=12)
    fig.tight_layout()
    fig.savefig("figure_1_signal_and_fingerprint.png", dpi=200)
    print("Saved figure_1_signal_and_fingerprint.png   J_sb of the three baths:", np.round(jsb[picks], 3),
          " J_L12:", np.round(jl12[picks], 3))


def figure_9():
    d = pd.read_csv(os.path.join(PART2, "tableF1_preflight_identifiability.csv")).set_index("Parameter")
    names = list(pf.LABELS)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for col, lab, c, mk in (("bound_balanced", "Balanced input", "#eb6834", "s"),
                            ("bound_three_inputs", "Three inputs", "#2a78d6", "o")):
        ax.plot(x, d.loc[names, col], linestyle="none", color=c, marker=mk, ms=8, markeredgecolor=pf.SURFACE, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([pf.LABELS[n] for n in names], fontsize=9)
    ax.set_ylabel("Bound, in units of the range")
    ax.set_title("Bound on every coupling", fontsize=11)
    pf.grid(ax)
    pf.legend(fig, ax, 2, 0.1)
    fig.savefig("figure_9_bound_on_every_coupling.png", dpi=200)
    print("Saved figure_9_bound_on_every_coupling.png")


if __name__ == "__main__":
    pf.CHK_TRAPPED = PART2.replace("gamma_sys_0.0", "gamma_sys_%s") + "_check_placement"
    figure_1()
    figure_9()
    pf.figure_10()
    pf.figure_11()
