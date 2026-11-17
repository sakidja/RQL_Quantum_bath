#!/usr/bin/env python
"""
free_link_training.py

Dual rail qubit in a structured quantum bath with EVERY LINK FREE.

Purpose (part two of the dual rail letter): in the runs of the letter the
links of the near field were tied (one J_sb with a fixed asymmetry of 0.2,
one J_L12 on all four inter layer links, one J_L2 on both layer 2 bonds) and
the calibration input was the balanced state only. The training therefore
never saw a horizontal tilt. Here every link is an independent random draw,
and the calibration can use three input states (balanced, rail L, rail R).
The question answered: which links, the horizontal bonds in particular, are
read from the signal of the pair once the training contains the tilt.

Model: Heisenberg network of arXiv:2508.17514, Eqs. (13) to (26), with the
system qubit replaced by the two rails (split attachment).
  sites: 0 = rail L, 1 = rail R, 2 = Q1, 3 = Q2 (layer 1),
         4 = Q3, 5 = Q4, 6 = Q5 (layer 2)
  free parameters (9 couplings and 2 rates):
    J_sb_L1  rail L to Q1          J_sb_R2  rail R to Q2
    J_L1_12  Q1 to Q2 (horizontal, layer 1)
    J_L12_13, J_L12_14, J_L12_24, J_L12_25  (vertical, layer 1 to layer 2)
    J_L2_34, J_L2_45  (horizontal, layer 2)
    gamma_L1, gamma_L2  (far field rates of layer 1 and layer 2)

Arms:
  balanced      calibration from the balanced input only (as in the letter)
  three_inputs  calibration from balanced, rail L and rail R inputs

Steps: preflight (sensitivity and Cramer Rao bound of every link, per arm),
then training (FFT features, PCA, XGBoost), then ML followed by fit, on
unseen baths. Raw data are saved per bath.

Stage one uses exact signals. The noise ladder, the check placement and the
surface code of the letter are added from noise_ladder.py and
dual_rail_structured_bath.py in stage two.

Note on provenance: the feature function and the staged fit are written
from the description in the letter, not copied from the letter's scripts.

AI (LLM) assistance (Claude, Anthropic) was used in preparing code and text.
The framework, calculations and analysis are those of R. Sakidja, who takes
full responsibility for the scientific content.
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

# ----------------------------------------------------------------- constants
OMEGA = 1.0
BETA = 12.0
N_TH = 1.0 / (np.exp(BETA * OMEGA) - 1.0)
T_MAX = 1500.0
N_POINTS = 5000
N_SITES = 7
DIM = 8                      # vacuum + 7 single excitations
I_L, I_R = 1, 2              # sector indices of rail L and rail R

BONDS = [                    # (name, site a, site b, kind, (low, high))
    ("J_sb_L1", 0, 2, "vertical, rail to layer 1", (0.1, 0.5)),
    ("J_sb_R2", 1, 3, "vertical, rail to layer 1", (0.1, 0.5)),
    ("J_L1_12", 2, 3, "horizontal, layer 1", (0.5, 0.99)),
    ("J_L12_13", 2, 4, "vertical, layer 1 to layer 2", (0.01, 0.1)),
    ("J_L12_14", 2, 5, "vertical, layer 1 to layer 2", (0.01, 0.1)),
    ("J_L12_24", 3, 5, "vertical, layer 1 to layer 2", (0.01, 0.1)),
    ("J_L12_25", 3, 6, "vertical, layer 1 to layer 2", (0.01, 0.1)),
    ("J_L2_34", 4, 5, "horizontal, layer 2", (0.5, 0.99)),
    ("J_L2_45", 5, 6, "horizontal, layer 2", (0.5, 0.99)),
]
RATES = [
    ("gamma_L1", (2, 3), "far field rate, layer 1", (0.0001, 0.002)),
    ("gamma_L2", (4, 5, 6), "far field rate, layer 2", (0.0001, 0.002)),
]
# The far field rate is the thermal insulation of the near field. It is kept
# small (trapped regime of arXiv:2508.17514: 0.0008 in its non Markovian
# case, 0.0001 in its low dissipation case) so that the excitation is held
# in the layers and exchanges along the horizontal bonds. The rates are
# drawn uniformly in the logarithm; the couplings uniformly.
NAMES = [b[0] for b in BONDS] + [r[0] for r in RATES]
KINDS = [b[3] for b in BONDS] + [r[2] for r in RATES]
LOW = np.array([b[4][0] for b in BONDS] + [r[3][0] for r in RATES])
HIGH = np.array([b[4][1] for b in BONDS] + [r[3][1] for r in RATES])
WIDTH = HIGH - LOW
N_PAR = len(NAMES)
LOG_DRAW = np.array([n.startswith("gamma_") for n in NAMES])

INPUTS = {"balanced": 0.0, "rail_L": -np.pi / 4, "rail_R": np.pi / 4}
ARMS = {"balanced": ["balanced"],
        "three_inputs": ["balanced", "rail_L", "rail_R"]}
FIT_WINDOWS = [40, 100, 250, 500, 1000]
N_FREQ = 1250                # Fourier bins kept per signal component

# -------------------------------------------------- operators, full space
_SX = np.array([[0, 1], [1, 0]], complex)
_SY = np.array([[0, -1j], [1j, 0]], complex)
_SZ = np.array([[1, 0], [0, -1]], complex)
_SM = np.array([[0, 0], [1, 0]], complex)   # excited (index 0) to ground
_I2 = np.eye(2, dtype=complex)


def _op(single, site):
    out = np.array([[1.0 + 0j]])
    for k in range(N_SITES):
        out = np.kron(out, single if k == site else _I2)
    return out


def _heis(a, b):
    return sum(_op(s, a) @ _op(s, b) for s in (_SX, _SY, _SZ))


def _isometry():
    """Columns: vacuum (all sites in the ground state), then the excitation
    on site 0 .. 6. With H_local = (omega / 2) sum sigma_z the ground state
    of a site is basis index 1."""
    v = np.zeros((2 ** N_SITES, DIM), complex)
    v[2 ** N_SITES - 1, 0] = 1.0
    for s in range(N_SITES):
        v[(2 ** N_SITES - 1) - 2 ** (N_SITES - 1 - s), 1 + s] = 1.0
    return v


def _super_h(h):
    eye = np.eye(DIM)
    return -1j * (np.kron(h, eye) - np.kron(eye, h.T))


def _super_c(c):
    eye = np.eye(DIM)
    cdc = c.conj().T @ c
    return (np.kron(c, c.conj()) - 0.5 * np.kron(cdc, eye)
            - 0.5 * np.kron(eye, cdc.T))


def build_superoperators():
    """The Liouvillian is linear in the parameters:
    L = L0 + gamma_sys * L_deph + sum_k p_k * L_k (row major vectorization)."""
    v = _isometry()
    proj = lambda a: v.conj().T @ a @ v
    h_local = 0.5 * OMEGA * sum(_op(_SZ, s) for s in range(N_SITES))
    l0 = _super_h(proj(h_local))
    l_deph = sum(_super_c(proj(_op(_SZ, s))) for s in (0, 1))
    parts = [_super_h(proj(_heis(a, b))) for _, a, b, _, _ in BONDS]
    for _, sites, _, _ in RATES:
        d = np.zeros((DIM * DIM, DIM * DIM), complex)
        for s in sites:
            d += (1.0 + N_TH) * _super_c(proj(_op(_SM, s)))
            d += N_TH * _super_c(proj(_op(_SM, s).conj().T))
        parts.append(d)
    return l0, l_deph, np.array(parts)


L0, L_DEPH, L_PARTS = build_superoperators()


def liouvillian(params, gamma_sys):
    return L0 + gamma_sys * L_DEPH + np.tensordot(params, L_PARTS, axes=1)


def input_state(tilt, phase=0.0):
    """cos(pi/4 + tilt)|0> + exp(i phase) sin(pi/4 + tilt)|1>, logical 0 =
    excitation on rail L, logical 1 = excitation on rail R, bath empty."""
    psi = np.zeros(DIM, complex)
    psi[I_L] = np.cos(np.pi / 4 + tilt)
    psi[I_R] = np.exp(1j * phase) * np.sin(np.pi / 4 + tilt)
    return np.outer(psi, psi.conj()).reshape(-1)


def signal(params, gamma_sys, input_names, n_points=N_POINTS):
    """Calibration signal. Returns array (inputs, 4, n_points):
    component 0: 2 Re rho[R, L]   (logical X)
    component 1: 2 Im rho[R, L]   (logical Y)
    component 2: rho[L, L] - rho[R, R]   (logical Z)
    component 3: rho[L, L] + rho[R, R]   (population left on the rails)."""
    dt = T_MAX / (N_POINTS - 1)
    step = expm(liouvillian(params, gamma_sys) * dt)
    v = np.stack([input_state(INPUTS[n]) for n in input_names], axis=1)
    out = np.empty((len(input_names), 4, n_points))
    kc, kl, kr = I_R * DIM + I_L, I_L * DIM + I_L, I_R * DIM + I_R
    for k in range(n_points):
        out[:, 0, k] = 2 * v[kc].real
        out[:, 1, k] = 2 * v[kc].imag
        out[:, 2, k] = (v[kl] - v[kr]).real
        out[:, 3, k] = (v[kl] + v[kr]).real
        v = step @ v
    return out


def fft_features(sig):
    """Log enhanced, normalized Fourier spectrum of every component."""
    x = sig.reshape(-1, sig.shape[-1])
    spec = np.log1p(np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True),
                                       axis=1)))[:, :N_FREQ]
    spec /= spec.max(axis=1, keepdims=True) + 1e-12
    return spec.astype(np.float32).reshape(-1)


def draw(rng, n):
    u = rng.random((n, N_PAR))
    return np.where(LOG_DRAW, LOW * (HIGH / LOW) ** u, LOW + WIDTH * u)


def occupancy(params, gamma_sys, input_name="balanced"):
    """Excitation held in each part of the system during the record:
    time average and peak of the population on the rails, in layer 1
    (Q1, Q2) and in layer 2 (Q3, Q4, Q5), and what is left at the end."""
    dt = T_MAX / (N_POINTS - 1)
    step = expm(liouvillian(params, gamma_sys) * dt)
    v = input_state(INPUTS[input_name])
    diag = np.arange(DIM) * (DIM + 1)
    pop = np.empty((N_POINTS, DIM))
    for k in range(N_POINTS):
        pop[k] = v[diag].real
        v = step @ v
    rails, l1, l2 = pop[:, 1:3].sum(1), pop[:, 3:5].sum(1), pop[:, 5:8].sum(1)
    return [rails.mean(), l1.mean(), l1.max(), l2.mean(), l2.max(),
            rails[-1] + l1[-1] + l2[-1]]


# ------------------------------------------------------------- validation
def validate_against_full(gamma_sys, seed=7):
    """Sector solution against the full 7 qubit master equation in QuTiP."""
    import qutip as qt
    p = draw(np.random.default_rng(seed), 1)[0]
    dims = [[2] * N_SITES, [2] * N_SITES]
    q = lambda a: qt.Qobj(a, dims=dims).to("csr")   # sparse: dense needs GBs
    h = 0.5 * OMEGA * sum(_op(_SZ, s) for s in range(N_SITES))
    for k, (_, a, b, _, _) in enumerate(BONDS):
        h = h + p[k] * _heis(a, b)
    c_ops = []
    if gamma_sys > 0:
        c_ops += [q(np.sqrt(gamma_sys) * _op(_SZ, s)) for s in (0, 1)]
    for k, (_, sites, _, _) in enumerate(RATES):
        g = p[len(BONDS) + k]
        for s in sites:
            c_ops.append(q(np.sqrt(g * (1 + N_TH)) * _op(_SM, s)))
            c_ops.append(q(np.sqrt(g * N_TH) * _op(_SM, s).conj().T))
    v = _isometry()
    n_chk = 120
    tl = np.arange(n_chk) * T_MAX / (N_POINTS - 1)
    worst = 0.0
    for name in ("balanced", "rail_L"):
        rho0 = v @ input_state(INPUTS[name]).reshape(DIM, DIM) @ v.conj().T
        res = qt.mesolve(q(h), q(rho0), tl, c_ops,
                         options={"atol": 1e-10, "rtol": 1e-8,
                                  "nsteps": 100000})
        sec = signal(p, gamma_sys, [name], n_chk)[0]
        for k, st in enumerate(res.states):
            r = v.conj().T @ st.full() @ v
            full = [2 * r[I_R, I_L].real, 2 * r[I_R, I_L].imag,
                    (r[I_L, I_L] - r[I_R, I_R]).real,
                    (r[I_L, I_L] + r[I_R, I_R]).real]
            worst = max(worst, np.max(np.abs(sec[:, k] - full)))
    return worst


# --------------------------------------------------------------- preflight
def preflight_job(args):
    p, gamma_sys, n_points = args
    names = ARMS["three_inputs"]
    base = signal(p, gamma_sys, names, n_points)
    jac = np.empty((N_PAR,) + base.shape)
    for k in range(N_PAR):
        h = 1e-4 * WIDTH[k]
        up, dn = p.copy(), p.copy()
        up[k] += h
        dn[k] -= h
        jac[k] = (signal(up, gamma_sys, names, n_points)
                  - signal(dn, gamma_sys, names, n_points)) / 2e-4
    out = {}
    for arm, sel in (("balanced", [0]), ("three_inputs", [0, 1, 2])):
        j = jac[:, sel].reshape(N_PAR, -1).T      # (points, parameters)
        s = base[sel].reshape(-1)
        fisher = j.T @ j
        sv = np.linalg.svd(j, compute_uv=False)
        crb = np.sqrt(np.abs(np.diag(np.linalg.pinv(fisher))))
        out[arm] = (np.linalg.norm(j, axis=0) / np.linalg.norm(s), crb,
                    sv[0] / sv[-1])
    out["occupancy"] = occupancy(p, gamma_sys)
    return out


def run_preflight(pool, rng, gamma_sys, n_baths, n_points, outdir, sigma):
    """Sensitivity of the signal to every link, and the Cramer Rao bound on
    every link for noise sigma per signal point, in units of the range of
    the parameter. A bound near or above 1 means the link is not read."""
    baths = draw(rng, n_baths)
    res = pool.map(preflight_job, [(p, gamma_sys, n_points) for p in baths])
    occ = np.array([r.pop("occupancy") for r in res])
    occ_head = ["rails_time_average", "layer1_time_average", "layer1_peak",
                "layer2_time_average", "layer2_peak", "total_left_at_end"]
    with open(os.path.join(outdir, "raw_preflight_occupancy_per_bath.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Bath"] + NAMES + occ_head)
        w.writerows([[i] + list(baths[i]) + list(occ[i])
                     for i in range(n_baths)])
    with open(os.path.join(outdir, "tableF0_preflight_occupancy.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Quantity", "median", "lowest_tenth", "highest_tenth"])
        print("\nTable F0. Excitation held in the layers, balanced input, "
              "record of %g time units (%d baths)." % (T_MAX, n_baths))
        print("%-22s %10s %14s %14s" % ("Quantity", "median",
                                        "lowest tenth", "highest tenth"))
        for k, h in enumerate(occ_head):
            row = [h, np.median(occ[:, k]), np.quantile(occ[:, k], 0.1),
                   np.quantile(occ[:, k], 0.9)]
            w.writerow(row)
            print("%-22s %10.4f %14.4f %14.4f" % tuple(row))
    with open(os.path.join(outdir, "raw_preflight_per_bath.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Bath", "Arm", "Parameter", "Kind", "true_value",
                    "relative_sensitivity", "cramer_rao_bound_in_ranges",
                    "condition_number"])
        for i, r in enumerate(res):
            for arm, (sens, crb, cond) in r.items():
                for k in range(N_PAR):
                    w.writerow([i, arm, NAMES[k], KINDS[k], baths[i, k],
                                sens[k], sigma * crb[k], cond])
    rows = []
    for k in range(N_PAR):
        row = [NAMES[k], KINDS[k]]
        med = {}
        for arm in ARMS:
            med[arm] = (np.median([r[arm][0][k] for r in res]),
                        sigma * np.median([r[arm][1][k] for r in res]))
            row += list(med[arm])
        row.append(med["balanced"][1] / med["three_inputs"][1])
        rows.append(row)
    head = ["Parameter", "Kind", "sensitivity_balanced", "bound_balanced",
            "sensitivity_three_inputs", "bound_three_inputs",
            "bound_ratio_balanced_over_three"]
    with open(os.path.join(outdir, "tableF1_preflight_identifiability.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    print("\nTable F1. Preflight identifiability of every link "
          "(median over %d baths, %d points, noise %.3g per point)."
          % (n_baths, n_points, sigma))
    print("Bound = Cramer Rao bound in units of the parameter range.")
    print("%-10s %-30s %12s %12s %12s %12s %8s" % (
        "Parameter", "Kind", "sens bal", "bound bal", "sens three",
        "bound three", "ratio"))
    for r in rows:
        print("%-10s %-30s %12.4g %12.4g %12.4g %12.4g %8.2f" % tuple(r))
    print("A ratio of 1.73 is what three times more data gives by itself; "
          "a larger ratio is information added by the tilted inputs.")
    return rows


# ------------------------------------------------------ training and fit
N_ENV_WINDOWS, N_LINES = 25, 6


def extra_features(sig):
    """Per signal row: the decay envelope (log10 RMS in N_ENV_WINDOWS time windows) and the
    positions and log heights of the N_LINES strongest spectral lines, in order of frequency.
    These bypass the PCA. The far field rates are narrower than one frequency bin, so they
    show in how the oscillation dies in time, not in the spectrum."""
    from scipy.signal import find_peaks
    out = []
    for row in sig.reshape(-1, sig.shape[-1]):
        s = row - row.mean()
        n = len(s) // N_ENV_WINDOWS * N_ENV_WINDOWS
        out.extend(np.log10(np.sqrt((s[:n].reshape(N_ENV_WINDOWS, -1) ** 2).mean(axis=1)) + 1e-9).tolist())
        lg = np.log10(2 * np.abs(np.fft.rfft(s))[: len(s) // 2] / len(s) + 1e-14)
        idx, _ = find_peaks(lg, prominence=0.5)
        if len(idx) == 0:
            idx = np.array([int(np.argmax(lg))])
        top = np.sort(idx[np.argsort(lg[idx])[::-1][:N_LINES]])
        pos, height = np.full(N_LINES, -1.0), np.full(N_LINES, -14.0)
        pos[: len(top)], height[: len(top)] = top, lg[top]
        out.extend(pos.tolist() + height.tolist())
    return np.array(out)


def feature_job(args):
    p, gamma_sys = args
    sig = signal(p, gamma_sys, ARMS["three_inputs"])
    return (fft_features(sig[:1]), fft_features(sig), sig,
            extra_features(sig[:1]), extra_features(sig))


EXACT_COST = 1e-16           # a fit below this cost reproduces the signal
I34, I45 = NAMES.index("J_L2_34"), NAMES.index("J_L2_45")
LAYER2_SIDE = [k for k, n in enumerate(NAMES)
               if n.startswith("J_L12_") or n.startswith("J_L2_")]


def _fit_once(x, target, gamma_sys, names):
    lo = np.maximum(LOW - 0.5 * WIDTH, 1e-5)
    hi = HIGH + 0.5 * WIDTH
    x = np.clip(x, lo, hi)
    for n in FIT_WINDOWS:
        res = least_squares(
            lambda q: (signal(q, gamma_sys, names, n)
                       - target[:, :, :n]).reshape(-1),
            x, bounds=(lo, hi), x_scale=WIDTH, method="trf",
            diff_step=1e-6, max_nfev=200)
        x = res.x
    return x, res.cost


def fit_job(args):
    """Fit from several starting points; the lowest cost is kept.
    start 0: the ML estimate.
    start 1: the best result so far with the two layer 2 bonds exchanged.
    start 2 and later: the best result so far with the links into layer 2
    and the layer 2 bonds drawn again at random.
    Stops as soon as a start reproduces the signal (cost below EXACT_COST).
    Returns the best parameters, the best cost, and every start."""
    p0, target, gamma_sys, names, n_starts, seed = args[:6]
    handoff = args[6] if len(args) > 6 else None
    rng = np.random.default_rng(seed)
    starts, best_x, best_c = [], None, np.inf
    for k in range(n_starts):
        if k == 0:
            x0, kind = p0.copy(), "ML estimate"
        elif k == 1:
            x0, kind = best_x.copy(), "layer 2 bonds exchanged"
            x0[I34], x0[I45] = best_x[I45], best_x[I34]
        elif k < 6:
            x0, kind = best_x.copy(), "layer 2 side drawn again"
            x0[LAYER2_SIDE] = draw(rng, 1)[0][LAYER2_SIDE]
        else:
            x0, kind = draw(rng, 1)[0].copy(), "every link drawn again"
        x, c = _fit_once(x0, target, gamma_sys, names)
        starts.append((k, kind, c, x))
        if c < best_c:
            best_x, best_c = x, c
        if best_c < EXACT_COST:
            break
    if handoff is not None and best_c >= EXACT_COST:
        x, c = _fit_once(handoff.copy(), target, gamma_sys, names)
        starts.append((len(starts), "fit of the balanced record handed over",
                       c, x))
        if c < best_c:
            best_x, best_c = x, c
    return best_x, best_c, starts


def classify(true, fitted):
    """exact: every parameter within 0.001 of its range.
    mirror: not exact, and the two layer 2 bonds within 0.05 of the range of
    each other's true value. other: neither."""
    err = np.abs(fitted - true) / WIDTH
    if err.max() < 1e-3:
        return "exact"
    if (abs(fitted[I34] - true[I45]) / WIDTH[I34] < 0.05
            and abs(fitted[I45] - true[I34]) / WIDTH[I45] < 0.05):
        return "mirror"
    return "other"


def r2(true, pred):
    return 1.0 - np.sum((true - pred) ** 2, axis=0) / np.sum(
        (true - true.mean(axis=0)) ** 2, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma_sys", type=float, default=0.0)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=49)
    ap.add_argument("--n_pc", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n_preflight", type=int, default=24)
    ap.add_argument("--preflight_points", type=int, default=N_POINTS)
    ap.add_argument("--preflight_sigma", type=float, default=0.01)
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--skip_validation", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--full_window", action="store_true",
                    help="continue the staged fit to 2500 and 5000 points "
                         "(the letter stops at 1000)")
    ap.add_argument("--n_starts", type=int, default=6,
                    help="starting points of the fit per bath (1 = the ML "
                         "estimate only, as in the first run)")
    ap.add_argument("--vertical_range", type=float, nargs=2, default=None,
                    metavar=("LOW", "HIGH"),
                    help="range of the four links from layer 1 to layer 2 "
                         "(default 0.01 0.1, the range of Ref. [1])")
    ap.add_argument("--gamma_range", type=float, nargs=2, default=None,
                    metavar=("LOW", "HIGH"),
                    help="range of the far field rates gamma_L1 and "
                         "gamma_L2 (default 0.0001 0.002, trapped regime; "
                         "the first runs used 0.001 0.01)")
    ap.add_argument("--rate_draw", choices=("log", "uniform"), default="log",
                    help="how the far field rates are drawn (the first "
                         "runs, an earlier run and preflight_300_gamma_sys_0.0, "
                         "drew them uniformly on 0.001 to 0.01)")
    a = ap.parse_args()
    if a.rate_draw == "uniform":
        LOG_DRAW[:] = False
    if a.gamma_range:
        for k, n in enumerate(NAMES):
            if n.startswith("gamma_"):
                LOW[k], HIGH[k] = a.gamma_range
        WIDTH[:] = HIGH - LOW
    if a.vertical_range:
        for k, n in enumerate(NAMES):
            if n.startswith("J_L12_"):
                LOW[k], HIGH[k] = a.vertical_range
        WIDTH[:] = HIGH - LOW
    if a.full_window:
        FIT_WINDOWS.extend([2500, N_POINTS])
    outdir = a.outdir or "results_free_links_gamma_sys_%s" % a.gamma_sys
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    print("Free link training. gamma_sys = %s, parameters = %d (%d couplings"
          ", %d rates)" % (a.gamma_sys, N_PAR, len(BONDS), len(RATES)))
    print("Fit windows (points): %s" % FIT_WINDOWS)
    print("Starting points of the fit per bath: up to %d" % a.n_starts)
    print("Range of the far field rates: %g to %g (drawn %s)"
          % (LOW[NAMES.index("gamma_L2")], HIGH[NAMES.index("gamma_L2")],
             "in the logarithm" if a.rate_draw == "log" else "uniformly"))
    print("Range of the links from layer 1 to layer 2: %g to %g"
          % (LOW[NAMES.index("J_L12_13")], HIGH[NAMES.index("J_L12_13")]))

    if not a.skip_validation:
        worst = validate_against_full(a.gamma_sys)
        print("Validation against the full 7 qubit master equation: largest "
              "difference of the signal = %.3e" % worst)
        if worst > 1e-4:
            raise SystemExit("Sector model does not match the full model.")

    rng = np.random.default_rng(a.seed)
    with Pool(a.workers) as pool:
        run_preflight(pool, np.random.default_rng(a.seed + 1), a.gamma_sys,
                      a.n_preflight, a.preflight_points, outdir,
                      a.preflight_sigma)
        print("Preflight done, %.0f s" % (time.time() - t0))
        if a.preflight_only:
            return

        from sklearn.decomposition import PCA
        from xgboost import XGBRegressor
        p_train, p_test = draw(rng, a.n_train), draw(rng, a.n_test)
        jobs = [(p, a.gamma_sys) for p in np.vstack([p_train, p_test])]
        res = pool.map(feature_job, jobs, chunksize=8)
        feats = {"balanced": np.stack([r[0] for r in res]),
                 "three_inputs": np.stack([r[1] for r in res])}
        extra = {"balanced": np.stack([r[3] for r in res]),
                 "three_inputs": np.stack([r[4] for r in res])}
        sig_test = np.stack([r[2] for r in res[a.n_train:]])
        del res
        print("Signals done, %.0f s" % (time.time() - t0))
        np.savez_compressed(os.path.join(outdir, "raw_signals_test.npz"),
                            signals=sig_test, parameters=p_test,
                            names=NAMES, inputs=ARMS["three_inputs"])
        with open(os.path.join(outdir, "raw_training_parameters.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["Bath"] + NAMES)
            w.writerows([[i] + list(p) for i, p in enumerate(p_train)])

        est, fit, cost, all_starts = {}, {}, {}, {}
        for arm, names in ARMS.items():
            pca = PCA(n_components=a.n_pc, svd_solver="randomized",
                      random_state=0).fit(feats[arm][:a.n_train])
            z = np.hstack([pca.transform(feats[arm]), extra[arm]])   # only the spectrum goes through the PCA
            pred = np.empty((a.n_test, N_PAR))
            for k in range(N_PAR):
                m = XGBRegressor(n_estimators=400, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 n_jobs=a.workers, random_state=0)
                m.fit(z[:a.n_train], p_train[:, k])
                pred[:, k] = m.predict(z[a.n_train:])
            est[arm] = pred
            sel = [ARMS["three_inputs"].index(n) for n in names]
            out = pool.map(fit_job, [(pred[i], sig_test[i][sel], a.gamma_sys,
                                      names, a.n_starts, a.seed + 1000 + i)
                                     + ((fit["balanced"][i],)
                                        if arm == "three_inputs" else ())
                                     for i in range(a.n_test)], chunksize=1)
            fit[arm] = np.array([o[0] for o in out])
            cost[arm] = np.array([o[1] for o in out])
            all_starts[arm] = [o[2] for o in out]
            print("Arm %s done, %.0f s" % (arm, time.time() - t0))

    with open(os.path.join(outdir, "raw_parameters_per_bath.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Arm", "Bath"] + ["true_" + n for n in NAMES]
                   + ["regression_" + n for n in NAMES]
                   + ["fit_" + n for n in NAMES]
                   + ["fit_cost", "starts_used", "outcome"])
        for arm in ARMS:
            for i in range(a.n_test):
                w.writerow([arm, i] + list(p_test[i]) + list(est[arm][i])
                           + list(fit[arm][i])
                           + [cost[arm][i], len(all_starts[arm][i]),
                              classify(p_test[i], fit[arm][i])])
    with open(os.path.join(outdir, "raw_fit_starts_per_bath.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Arm", "Bath", "Start", "Start_kind", "fit_cost",
                    "outcome"] + ["fit_" + n for n in NAMES])
        for arm in ARMS:
            for i in range(a.n_test):
                for k, kind, c, x in all_starts[arm][i]:
                    w.writerow([arm, i, k, kind, c, classify(p_test[i], x)]
                               + list(x))

    with open(os.path.join(outdir, "tableF2_r2_free_links.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Kind", "R2_ML_balanced",
                    "R2_ML_then_fit_balanced", "R2_ML_three_inputs",
                    "R2_ML_then_fit_three_inputs"])
        cols = [r2(p_test, est["balanced"]), r2(p_test, fit["balanced"]),
                r2(p_test, est["three_inputs"]),
                r2(p_test, fit["three_inputs"])]
        print("\nTable F2. R2 on %d unseen baths, every link free, %d "
              "training baths, gamma_sys = %s." % (a.n_test, a.n_train,
                                                    a.gamma_sys))
        print("%-10s %-30s %10s %10s %10s %10s" % (
            "Parameter", "Kind", "ML bal", "fit bal", "ML three",
            "fit three"))
        for k in range(N_PAR):
            row = [NAMES[k], KINDS[k]] + [c[k] for c in cols]
            w.writerow(row)
            print("%-10s %-30s %10.4f %10.4f %10.4f %10.4f" % tuple(row))
    with open(os.path.join(outdir, "tableF3_fit_outcomes.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Arm", "Baths", "Exact", "Mirror", "Other",
                    "Exact_at_start_0", "Median_starts_used"])
        print("\nTable F3. Outcome of the fit on %d unseen baths, up to %d "
              "starting points, gamma_sys = %s." % (a.n_test, a.n_starts,
                                                    a.gamma_sys))
        print("exact: every parameter within 0.001 of its range. mirror: "
              "the two layer 2 bonds exchanged. other: neither.")
        print("%-14s %6s %6s %7s %6s %17s %14s" % (
            "Arm", "Baths", "Exact", "Mirror", "Other", "Exact at start 0",
            "Median starts"))
        for arm in ARMS:
            out = [classify(p_test[i], fit[arm][i]) for i in range(a.n_test)]
            first = sum(classify(p_test[i], all_starts[arm][i][0][3])
                        == "exact" for i in range(a.n_test))
            row = [arm, a.n_test, out.count("exact"), out.count("mirror"),
                   out.count("other"), first,
                   float(np.median([len(v) for v in all_starts[arm]]))]
            w.writerow(row)
            print("%-14s %6d %6d %7d %6d %17d %14.1f" % tuple(row))
    print("Total time %.0f s. Results in %s" % (time.time() - t0, outdir))


if __name__ == "__main__":
    main()
