# Schematic of the structured bath of the dual rail qubit.
# Draws the coupling graph of dual_rail_structured_bath.py: sites 0 and 1 are the
# rails L and R, sites 2 and 3 are layer 1, sites 4 to 6 are layer 2.
#   H_SRC   first attachment : rail R to Q1 and to Q2, both J_sb
#           split attachment : rail L to Q1 at (1 - a) J_sb, rail R to Q2 at (1 + a) J_sb
#   H_L1    Q1 to Q2                                   J_L1
#   H_INTER Q1 to Q3, Q1 to Q4, Q2 to Q4, Q2 to Q5     0.8, 1.1, 0.9, 1.2 times J_L12
#   H_L2    Q3 to Q4, Q4 to Q5                         0.5, 1.5 times J_L2
#   thermal emission and absorption at gamma_L1 on layer 1 and gamma_L2 on layer 2
# Draws only; computes nothing. Writes figure_bath_schematic.png.
#
# AI (LLM) assistance (Claude, Anthropic) was used in preparing this code. The
# framework, calculations and analysis are those of R. Sakidja, who takes full
# responsibility for the scientific content.
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, SURFACE, PANEL = "#0b0b0b", "#52514e", "white", "#f2f2f0"
RAIL, LAY1, LAY2 = "#2a78d6", "#1baf7a", "#eb6834"
plt.rcParams.update({"font.size": 10, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

POS = {"L": (0.0, 3.0), "R": (0.0, 1.4),
       "Q1": (2.6, 3.0), "Q2": (2.6, 1.4),
       "Q3": (5.0, 3.7), "Q4": (5.0, 2.2), "Q5": (5.0, 0.7)}
FACE = {"L": RAIL, "R": RAIL, "Q1": LAY1, "Q2": LAY1, "Q3": LAY2, "Q4": LAY2, "Q5": LAY2}
GROUPS = ((("L", "R"), RAIL, "Dual rail qubit"),
          (("Q1", "Q2"), LAY1, "Layer 1  ($\\gamma_{L1}$)"),
          (("Q3", "Q4", "Q5"), LAY2, "Layer 2  ($\\gamma_{L2}$)"))
R = 0.34
LABEL_Y = 0.02          # one line for the three group labels


def group_box(ax, names, color, label):
    xs = [POS[n][0] for n in names]
    ys = [POS[n][1] for n in names]
    x0, x1 = min(xs) - R - 0.26, max(xs) + R + 0.26
    y0, y1 = min(ys) - R - 0.26, max(ys) + R + 0.26
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.22",
                                facecolor=PANEL, edgecolor="none", zorder=0))
    ax.text((x0 + x1) / 2, LABEL_Y, label, color=color, ha="center", va="top", fontsize=10)


def node(ax, name):
    x, y = POS[name]
    ax.add_patch(plt.Circle((x, y), R, facecolor=FACE[name], edgecolor=SURFACE, lw=1.5, zorder=3))
    ax.text(x, y, name, ha="center", va="center", color="white", fontsize=10, zorder=4)


def bond(ax, a, b, color=MUTED, lw=1.6, label=None, at=0.5):
    (x0, y0), (x1, y1) = POS[a], POS[b]
    ax.plot([x0, x1], [y0, y1], "-", color=color, lw=lw, zorder=2, solid_capstyle="round")
    if label:
        ax.text(x0 + at * (x1 - x0), y0 + at * (y1 - y0), label, color=color, fontsize=10,
                ha="center", va="center", zorder=5,
                bbox=dict(facecolor=PANEL, edgecolor="none", pad=1.4))


L12 = (("Q1", "Q3", 0.8), ("Q1", "Q4", 1.1), ("Q2", "Q4", 0.9), ("Q2", "Q5", 1.2))   # relative strengths of the links into layer 2
L2 = (("Q3", "Q4", 0.5), ("Q4", "Q5", 1.5))                                         # relative strengths of the bonds of layer 2


def panel(ax, split, title):
    for names, color, label in GROUPS:
        group_box(ax, names, color, label)
    for a, b, c in L12:                       # the width of a line follows its relative strength
        bond(ax, a, b, lw=1.7 * c, label="$%g\\,J_{L12}$" % c, at=0.45)
    for a, b, c in L2:
        bond(ax, a, b, lw=1.7 * c)
        (x0, y0), (x1, y1) = POS[a], POS[b]
        ax.text(x0 + R + 0.12, (y0 + y1) / 2, "$%g\\,J_{L2}$" % c, color=MUTED, fontsize=10, ha="left", va="center")
    bond(ax, "Q1", "Q2", label="$J_{L1}$")
    if split:
        bond(ax, "L", "Q1", color=INK, lw=2.0, label="$(1-a)\\,J_{sb}$")
        bond(ax, "R", "Q2", color=INK, lw=2.0, label="$(1+a)\\,J_{sb}$")
    else:
        bond(ax, "R", "Q1", color=INK, lw=2.0, label="$J_{sb}$", at=0.58)
        bond(ax, "R", "Q2", color=INK, lw=2.0, label="$J_{sb}$")
    for name in POS:
        node(ax, name)
    ax.text(2.9, 4.95, title, ha="center", va="center", fontsize=10.5)
    ax.set_xlim(-0.80, 6.60)
    ax.set_ylim(-0.50, 5.25)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
panel(axes[0], False, "First attachment")
panel(axes[1], True, "Split attachment")
fig.tight_layout()
fig.savefig("figure_bath_schematic.png", dpi=200)
print("wrote figure_bath_schematic.png")
