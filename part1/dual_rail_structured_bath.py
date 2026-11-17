# ======================================================================
# Dual rail qubit in a structured quantum bath: erasure checks placed
# by the ML inferred bath against erasure checks placed by a clock.
#
# Bath model: the layered bath of QuTiP_Quantum_Bath_v2.ipynb
# (github.com/sakidja/quantum_bath, arXiv:2508.17514): layer 1 (Q1, Q2),
# layer 2 (Q3, Q4, Q5), Heisenberg couplings, thermal Lindblad
# operators on the bath, pure dephasing on the system, the same
# parameter ranges, time grid and log enhanced FFT features, followed
# by PCA and XGBoost regression.
#
# What is new here (extensions, all listed in SETTINGS):
#   1. The system slot holds a dual rail pair (rail L, rail R) in place
#      of the single qubit Q0. Logical 0 = excitation on L, logical 1 =
#      excitation on R.
#   2. The bath is cold (BETA = 12, i.e. 5 GHz at 20 mK). At the paper's
#      BETA = 0.5 the bath qubits are 38 percent excited and a dual rail
#      code space cannot exist.
#   3. With a cold bath the dynamics stay in the sector of at most one
#      excitation (dimension 8), so the master equation is solved
#      exactly there. validate_against_full() checks this against the
#      full 7 qubit QuTiP solution.
#
# AI (LLM) assistance (Claude, Anthropic) was used in preparing this
# code. The framework, calculations and analysis are the author's.
# ======================================================================
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.linalg import expm, polar
from qutip import basis, ket2dm, tensor, qeye, sigmax, sigmay, sigmaz, sigmap, sigmam, mesolve

# ============================== SETTINGS ===============================
ATTACH = os.environ.get("SQB_ATTACH", "split")   # set the environment variable SQB_ATTACH=one for the first attachment
                         # "split": rail L couples to layer 1 node Q1 and rail R to node Q2;
                         # "one": layer 1 couples to rail R only (version 2); "both": both nodes to both rails
SOURCE_ASYMMETRY = 0.2   # "split" only: the couplings are (1 - a) J_sb for rail L and (1 + a) J_sb
                         # for rail R. Two rails sit at two places on the chip; unequal couplings expose
                         # the bond inside layer 1 (J_L1) to the signal of the pair.
BETA, OMEGA = 12.0, 1.0  # cold bath (paper: BETA = 0.5)
GAMMA_SYS_LIST = (0.0,)   # Lindblad dephasing put directly on each rail.
                         # 0.0: all phase noise reaches the qubit through the structured bath (near field).
                         # 0.005: the notebook's value for the system qubit, a rate with no bath behind it.
T_MAX, N_POINTS = 1500, 5000   # calibration time grid of the paper
PARAM_NAMES = ("J_sb", "J_L1", "J_L12", "J_L2", "gamma_L1", "gamma_L2")
PARAM_RANGES = ((0.1, 0.5), (0.5, 0.99), (0.1, 0.5), (0.5, 0.99), (0.0001, 0.002), (0.0001, 0.002))

N_TRAIN = 4000           # random baths for training, the size Part 2 trains on
N_TEST = 49              # unseen baths; one per data qubit of the distance 7 code
PCA_LIST = (10, 30)      # 30 is where the regression stops gaining, measured
PCA_USED = 30            # the PCA size whose inferred baths drive the predicted arm
XGB = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=0)   # placeholders: put your own values here

T_CLOCK_LIST = (1.0, 2.0, 3.0, 4.0, 6.0, 12.0, 25.0, 50.0)   # clocked check periods to compare
WINDOW = 0.5             # predicted check may fall in [T_clock, T_clock * (1 + WINDOW)]:
                         # never earlier than the clock, so it never checks more often
N_TAU = 61               # candidate check times inside that window
WINDOW_WIDE = 1.5        # wider window of the balanced check rule: [T_clock, T_clock * (1 + WINDOW_WIDE)]
N_TAU_WIDE = 151
ERASURE_COST = 0.125     # balanced rule minimizes (unflagged Pauli error + ERASURE_COST * erasure):
                         # a heralded erasure costs the surface code about one eighth of a Pauli error
FIT_STAGES = (40, 80, 160, 320, 640, 1000)   # signal lengths (points) used in turn by the fit
N_FIT_POINTS = FIT_STAGES[-1]
N_DRAWS = 5              # random placements of the unseen baths on the data qubits, per point
P_CHECK = 1e-3           # Pauli error added by every erasure check (all arms alike)
P_MEAS = 1e-3            # syndrome measurement and reset flip probability

DISTANCES = (3, 5, 7)
BASES = ("z", "x")
SHOTS = 20000
P_FLOOR = 1e-8
SEED = 20261112
SAVE_RAW = True          # write the raw data of every bath behind the tables (parameters, signals, spectra)
CACHE = f"dual_rail_dataset_{ATTACH}_final_v3c.npz"   # holds the parameters, the signals and the FFT features of every bath
OUTDIR = f"results_v3_final8_{ATTACH}"

# ===================== MODEL (from the notebook) =======================
N_Q = 7                  # 0: rail L, 1: rail R, 2,3: layer 1 (Q1, Q2), 4,5,6: layer 2 (Q3, Q4, Q5)
def _op(op, i):
    ops = [qeye(2)] * N_Q
    ops[i] = op
    return tensor(ops)

SX = [_op(sigmax(), i) for i in range(N_Q)]
SY = [_op(sigmay(), i) for i in range(N_Q)]
SZ = [_op(sigmaz(), i) for i in range(N_Q)]
SM = [_op(sigmam(), i) for i in range(N_Q)]
SP = [_op(sigmap(), i) for i in range(N_Q)]

def _heis(a, b):
    return SX[a] * SX[b] + SY[a] * SY[b] + SZ[a] * SZ[b]

H_LOCAL = sum(0.5 * OMEGA * SZ[i] for i in range(N_Q))
if ATTACH == "split":
    H_SRC = (1 - SOURCE_ASYMMETRY) * _heis(0, 2) + (1 + SOURCE_ASYMMETRY) * _heis(1, 3)
else:
    H_SRC = _heis(1, 2) + _heis(1, 3)                   # rail R to layer 1, as Q0 in the notebook
if ATTACH == "both":
    H_SRC = H_SRC + _heis(0, 2) + _heis(0, 3)
H_L1 = _heis(2, 3)
L12_RATIOS = (0.8, 1.1, 0.9, 1.2)   # Q1 to Q3, Q1 to Q4, Q2 to Q4, Q2 to Q5
H_INTER = (L12_RATIOS[0] * _heis(2, 4) + L12_RATIOS[1] * _heis(2, 5)
           + L12_RATIOS[2] * _heis(3, 5) + L12_RATIOS[3] * _heis(3, 6))
L2_TILT = 0.5           # Q3 to Q4 at (1 - t), Q4 to Q5 at (1 + t); breaks the layer 2 mirror
H_L2 = (1 - L2_TILT) * _heis(4, 5) + (1 + L2_TILT) * _heis(5, 6)
N_TH = 1.0 / (np.exp(BETA * OMEGA) - 1.0)

def full_model(params):
    """Full 7 qubit Hamiltonian and collapse operators, as in the notebook.
    params = six bath parameters followed by the rail dephasing rate."""
    J_s, J_1, J_i, J_2, g1, g2, GAMMA_SYS = params
    H = H_LOCAL + J_s * H_SRC + J_1 * H_L1 + J_i * H_INTER + J_2 * H_L2
    c_ops = [np.sqrt(GAMMA_SYS) * SZ[0], np.sqrt(GAMMA_SYS) * SZ[1]]
    for i in range(2, N_Q):
        g = g1 if i in (2, 3) else g2
        c_ops += [np.sqrt(g * (1 + N_TH)) * SM[i], np.sqrt(g * N_TH) * SP[i]]
    return H, c_ops

# ---- sector of at most one excitation: |vac>, then the excitation on site 0..6 ----
# In the notebook's convention basis(2, 0) is the upper level (energy +omega/2).
_GROUND, _EXC = basis(2, 1), basis(2, 0)
_kets = [tensor([_GROUND] * N_Q)] + [tensor([_EXC if j == i else _GROUND for j in range(N_Q)])
                                      for i in range(N_Q)]
V = np.hstack([k.full() for k in _kets])                 # 128 x 8 isometry
def _proj(A):
    return V.conj().T @ A.full() @ V

_Hs = [_proj(h) for h in (H_LOCAL, H_SRC, H_L1, H_INTER, H_L2)]
_DEPH = [_proj(SZ[0]), _proj(SZ[1])]
_DECAY = [_proj(SM[i]) for i in range(2, N_Q)]
DIM = 8
I_L, I_R = 1, 2                                          # sector indices of the two rails
_EYE = np.eye(DIM)

def sector_model(params):
    """Hamiltonian and collapse operators in the 8 dimensional sector.
    Thermal absorption (rate gamma * n_th, about 6e-6 * gamma at BETA = 12) is dropped."""
    J_s, J_1, J_i, J_2, g1, g2, GAMMA_SYS = params
    H = _Hs[0] + J_s * _Hs[1] + J_1 * _Hs[2] + J_i * _Hs[3] + J_2 * _Hs[4]
    cs = [np.sqrt(GAMMA_SYS) * d for d in _DEPH]
    cs += [np.sqrt((g1 if k < 2 else g2) * (1 + N_TH)) * _DECAY[k] for k in range(5)]
    return H, cs

def liouvillian(params):
    """Generator for vec(rho) in row major order: vec(A rho B) = kron(A, B.T) vec(rho)."""
    H, cs = sector_model(params)
    L = -1j * (np.kron(H, _EYE) - np.kron(_EYE, H.T))
    for c in cs:
        cdc = c.conj().T @ c
        L += np.kron(c, c.conj()) - 0.5 * np.kron(cdc, _EYE) - 0.5 * np.kron(_EYE, cdc.T)
    return L

def logical_input(i, j):
    """|i><j| on the rails (0 = L, 1 = R), bath in its ground state, as a vector."""
    rho = np.zeros((DIM, DIM), dtype=complex)
    rho[I_L + i, I_L + j] = 1.0
    return rho.reshape(-1)

INPUT_TILTS = {"balanced": 0.0, "rail_L": -np.pi / 4, "rail_R": np.pi / 4}
INPUT_SETS = {"balanced": ("balanced",),
              "three_inputs": ("balanced", "rail_L", "rail_R")}
INPUTS_USED = INPUT_SETS[os.environ.get("SQB_INPUTS", "three_inputs")]


def logical_input_state(tilt):
    """cos(pi/4 + tilt)|L> + sin(pi/4 + tilt)|R>, bath in its ground state, as a vector.
    tilt 0 is the plus state of the paper; -pi/4 loads rail L and +pi/4 loads rail R.
    This is input_state() of free_link_training.py."""
    psi = np.zeros(DIM, dtype=complex)
    psi[I_L] = np.cos(np.pi / 4 + tilt)
    psi[I_R] = np.sin(np.pi / 4 + tilt)
    return np.outer(psi, psi.conj()).reshape(-1)


def calibration_signal(params, n_points=N_POINTS):
    """Logical Bloch vector <X>, <Y>, <Z> on the paper's grid, from every input state in
    INPUTS_USED, stacked: rows 3i, 3i+1, 3i+2 are X, Y, Z of input i."""
    tl = np.linspace(0, T_MAX, N_POINTS)
    step = expm(liouvillian(params) * (tl[1] - tl[0]))
    v = np.stack([logical_input_state(INPUT_TILTS[n]) for n in INPUTS_USED], axis=1)
    out = np.zeros((3 * len(INPUTS_USED), n_points))
    kc, kl, kr = I_R * DIM + I_L, I_L * DIM + I_L, I_R * DIM + I_R
    for k in range(n_points):
        out[0::3, k] = 2 * v[kc].real
        out[1::3, k] = 2 * v[kc].imag
        out[2::3, k] = (v[kl] - v[kr]).real
        v = step @ v
    return out

N_ENV_WINDOWS = 25        # time windows of the decay envelope, per signal row
N_LINES = 6               # strongest spectral lines kept per signal row
N_EXTRA = (N_ENV_WINDOWS + 2 * N_LINES) * 3 * len(INPUTS_USED)   # features that bypass the PCA
N_RAW_FIT = 400           # training baths whose fit is saved as raw data (in no table)


def extra_features(bloch):
    """Per signal row: the decay envelope (log10 RMS in N_ENV_WINDOWS time windows) and the
    positions and log heights of the N_LINES strongest spectral lines, in order of frequency.
    The far field rates are narrower than one frequency bin of the record, so they show in
    how the oscillation dies in time, not in the spectrum."""
    from scipy.signal import find_peaks
    out = []
    for signal in bloch:
        s = signal - np.mean(signal)
        n = len(s) // N_ENV_WINDOWS * N_ENV_WINDOWS
        out.extend(np.log10(np.sqrt((s[:n].reshape(N_ENV_WINDOWS, -1) ** 2).mean(axis=1)) + 1e-9).tolist())
        lg = np.log10(2 * np.abs(fft(s))[: len(s) // 2] / len(s) + 1e-14)
        idx, _ = find_peaks(lg, prominence=0.5)
        if len(idx) == 0:
            idx = np.array([int(np.argmax(lg))])
        top = np.sort(idx[np.argsort(lg[idx])[::-1][:N_LINES]])
        pos, height = np.full(N_LINES, -1.0), np.full(N_LINES, -14.0)
        pos[: len(top)], height[: len(top)] = top, lg[top]
        out.extend(pos.tolist() + height.tolist())
    return np.array(out)


def fft_features(bloch, eps=1e-12):
    """The log enhanced, normalized FFT vector of the notebook (all rows merged), followed by
    the N_EXTRA features of extra_features."""
    out = []
    for signal in bloch:
        signal = signal - np.mean(signal)
        raw = np.abs(fft(signal)) / len(signal)
        pos = 2 * raw[: len(signal) // 2]
        lg = np.log10(pos + eps)
        lg -= np.min(lg)
        out.extend(np.clip(lg / (np.max(lg) + eps), 0, 1).tolist())
    return np.concatenate([np.array(out), extra_features(bloch)])


class FeaturePCA:
    """PCA of the spectrum; the last N_EXTRA features pass through unchanged."""
    def __init__(self, n_components, random_state=0):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X):
        self.pca.fit(X[:, :-N_EXTRA])
        return self

    def transform(self, X):
        return np.hstack([self.pca.transform(X[:, :-N_EXTRA]), X[:, -N_EXTRA:]])

def validate_against_full(params, t_end=40.0, n=201):
    """Largest difference between the sector solution and the full 7 qubit QuTiP solution."""
    tl = np.linspace(0, t_end, n)
    H, c_ops = full_model(params)
    rb = (-BETA * 0.5 * OMEGA * sigmaz()).expm()
    rb = rb / rb.tr()
    plus = (tensor(_EXC, _GROUND) + tensor(_GROUND, _EXC)).unit()
    rho0 = tensor([ket2dm(plus)] + [rb] * 5)
    XL = 0.5 * (SX[0] * SX[1] + SY[0] * SY[1])
    YL = 0.5 * (SX[0] * SY[1] - SY[0] * SX[1])
    ZL = 0.5 * (SZ[0] - SZ[1])
    res = mesolve(H, rho0, tl, c_ops, e_ops=[XL, YL, ZL],
                  options={"atol": 1e-10, "rtol": 1e-8, "nsteps": 10 ** 7, "progress_bar": False})
    full = np.array([np.real(e) for e in res.expect])
    step = expm(liouvillian(params) * (tl[1] - tl[0]))
    v = 0.5 * sum(logical_input(i, j) for i in (0, 1) for j in (0, 1))
    sec = np.zeros((3, n))
    for k in range(n):
        rho = v.reshape(DIM, DIM)
        sec[:, k] = (2 * rho[I_R, I_L].real, 2 * rho[I_R, I_L].imag, (rho[I_L, I_L] - rho[I_R, I_R]).real)
        v = step @ v
    d = np.abs(full - sec).max(axis=1)
    if d[1] > 10 * max(d[0], d[2], 1e-9):               # sign convention of Y only
        d[1] = np.abs(full[1] + sec[1]).max()
    return float(d.max())

# ================= ONE ROUND AS A LOGICAL CHANNEL ======================
_PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.array([[1, 0], [0, -1]])}
_PHI = np.array([1, 0, 0, 1]) / np.sqrt(2)

def round_channel(L, tau, U_corr=None):
    """Channel of one round of length tau that starts from a logical state with the bath
    in its ground state and ends with an erasure check. Returns p_erase and the Pauli
    probabilities of the unflagged part (Pauli twirl), after the correction U_corr."""
    S = expm(L * tau)
    J = np.zeros((4, 4), dtype=complex)
    for i in (0, 1):
        for j in (0, 1):
            E = (S @ logical_input(i, j)).reshape(DIM, DIM)[I_L:I_R + 1, I_L:I_R + 1]
            if U_corr is not None:
                E = U_corr.conj().T @ E @ U_corr
            J[np.ix_([2 * i, 2 * i + 1], [2 * j, 2 * j + 1])] = E
    chi = {}
    for name, P in _PAULI.items():
        phi = np.kron(np.eye(2), P) @ _PHI
        chi[name] = float(np.real(phi.conj() @ (J / 2) @ phi))
    p_code = sum(chi.values())
    return {"p_erase": 1 - p_code, "chi_I": chi["I"], "p_x": chi["X"] / p_code,
            "p_y": chi["Y"] / p_code, "p_z": chi["Z"] / p_code}

def correction(params, tau):
    """Logical rotation with which the excitation returns: unitary part of the rail block
    of exp(-i H_eff tau), H_eff = H - (i/2) sum of C^dagger C."""
    H, cs = sector_model(params)
    H_eff = H - 0.5j * sum(c.conj().T @ c for c in cs)
    A = expm(-1j * H_eff * tau)[I_L:I_R + 1, I_L:I_R + 1]
    return polar(A)[0]

FIT_STARTS = int(os.environ.get("SQB_FIT_STARTS", "6"))   # starting points of the fit per bath
EXACT_COST = 1e-16       # a fit below this cost reproduces the signal; same value as Part 2
LAYER2_SIDE = [2, 3]     # J_L12 and J_L2: the part of the graph the signal constrains least


def _fit_once(q, signal, gamma_sys, lo, hi):
    """One staged least squares. The window is lengthened in stages so the fit follows the
    oscillations without slipping a cycle. Returns the parameters and the final cost."""
    from scipy.optimize import least_squares
    q = np.clip(q, lo, hi)
    res = None
    for n in FIT_STAGES:
        target = signal[:, :n]
        res = least_squares(lambda x: (calibration_signal(tuple(x) + (gamma_sys,), n) - target).ravel(),
                            x0=q, bounds=(lo, hi), x_scale=hi - lo,
                            diff_step=1e-6, max_nfev=200)
        q = res.x
    return q, res.cost


def refine_by_fit(p_start, signal, gamma_sys):
    """ML then fit, from several starting points; the lowest cost is kept.
    Start 0 is the ML estimate. Each later start takes the best result so far and draws
    the layer 2 side again. Stops as soon as a start reproduces the signal. This is the
    scheme of Part 2 (free_link_training.py, fit_job)."""
    lo = np.array([r[0] for r in PARAM_RANGES])
    hi = np.array([r[1] for r in PARAM_RANGES])
    rng = np.random.default_rng(int(abs(np.sum(signal)) * 1e6) % (2 ** 32))   # same bath, same answer
    best_x, best_c = _fit_once(p_start, signal, gamma_sys, lo, hi)
    for _ in range(1, max(1, FIT_STARTS)):
        if best_c < EXACT_COST:
            break
        q0 = best_x.copy()
        for j in LAYER2_SIDE:
            q0[j] = rng.uniform(lo[j], hi[j])
        x, c = _fit_once(q0, signal, gamma_sys, lo, hi)
        if c < best_c:
            best_x, best_c = x, c
    return best_x

def best_check_time(params, t_clock, rule="fidelity"):
    """Check time chosen from the bath model 'params', never earlier than the clock.
    rule "fidelity": highest unflagged process fidelity in [t_clock, t_clock * (1 + WINDOW)].
    rule "balance":  lowest (unflagged Pauli error + ERASURE_COST * erasure) in the wider window
                     [t_clock, t_clock * (1 + WINDOW_WIDE)]; with no direct rail dephasing the
                     unflagged Pauli error is the imbalance of the two rail amplitudes."""
    L = liouvillian(params)
    if rule == "fidelity":
        taus = np.linspace(t_clock, t_clock * (1 + WINDOW), N_TAU)
        scores = [round_channel(L, t, correction(params, t))["chi_I"] for t in taus]
        return float(taus[int(np.argmax(scores))])
    taus = np.linspace(t_clock, t_clock * (1 + WINDOW_WIDE), N_TAU_WIDE)
    cost = []
    for t in taus:
        c = round_channel(L, t, correction(params, t))
        cost.append(c["p_x"] + c["p_y"] + c["p_z"] + ERASURE_COST * c["p_erase"])
    return float(taus[int(np.argmin(cost))])

ARM_CLOCK_UNIFORM = "Clock, frame calibrated, uniform decoder"
ARM_CLOCK = "Clock, frame calibrated"

def device_arms(p_true, p_ml, p_fit, t_clock):
    """All arms for one data qubit (one bath) at one clock period. Every arm returns the
    channel that really acts ("true") and the channel its decoder believes ("prior")."""
    L_true = liouvillian(p_true)
    def arm(p_model, rule):
        t = best_check_time(p_model, t_clock, rule)
        U = correction(p_model, t)
        return {"true": dict(round_channel(L_true, t, U), tau=t),
                "prior": dict(round_channel(liouvillian(p_model), t, U), tau=t)}
    clock = dict(round_channel(L_true, t_clock, correction(p_true, t_clock)), tau=t_clock)
    return {
        ARM_CLOCK_UNIFORM: {"true": clock, "prior": None},      # prior = chip average, filled in later
        ARM_CLOCK: {"true": clock, "prior": clock},             # decoder knows every qubit (generous)
        "Structured bath, ML inferred": arm(p_ml, "fidelity"),
        "Structured bath, ML then fit": arm(p_fit, "fidelity"),
        "Structured bath, ML then fit, balanced check": arm(p_fit, "balance"),
        "Structured bath, exact (bound)": arm(p_true, "fidelity"),
        "Structured bath, exact, balanced check (bound)": arm(p_true, "balance"),
    }

# ================= SURFACE CODE WITH HERALDED ERASURE ==================
def build_circuit(distance, basis_name, p_x, p_z, p_e, p_meas):
    """Rotated surface code memory. Once per round data qubit q gets X_ERROR(p_x[q]),
    Z_ERROR(p_z[q]) and HERALDED_ERASE(p_e[q]); arrays are indexed by data qubit order.
    Every herald is its own detector; record lookbacks are rewritten."""
    import stim
    base = stim.Circuit.generated(f"surface_code:rotated_memory_{basis_name}", distance=distance,
                                  rounds=distance, before_measure_flip_probability=p_meas,
                                  after_reset_flip_probability=p_meas).flattened()
    final_meas = [i for i in base if i.name in ("M", "MX")][-1]
    data = [t.value for t in final_meas.targets_copy()]
    out, is_herald, new_index, new_n, h_seen = stim.Circuit(), [], [], 0, 0
    for inst in base:
        if inst.name == "H":
            if h_seen % 2 == 0:                          # start of a round
                for k, q in enumerate(data):
                    out.append("X_ERROR", [q], max(float(p_x[k]), P_FLOOR))
                    out.append("Z_ERROR", [q], max(float(p_z[k]), P_FLOOR))
                for k, q in enumerate(data):
                    if p_e[k] > 0:
                        out.append("HERALDED_ERASE", [q], float(p_e[k]))
                        new_n += 1
                        out.append("DETECTOR", [stim.target_rec(-1)])
                        is_herald.append(True)
            h_seen += 1
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            targets = [stim.target_rec(new_index[len(new_index) + t.value] - new_n)
                       if t.is_measurement_record_target else t for t in inst.targets_copy()]
            out.append(inst.name, targets, inst.gate_args_copy())
            if inst.name == "DETECTOR":
                is_herald.append(False)
        else:
            out.append(inst)
            for _ in range(stim.Circuit(str(inst)).num_measurements):
                new_index.append(new_n)
                new_n += 1
    assert h_seen == 2 * distance, "unexpected circuit layout"
    return out, np.array(is_herald, dtype=bool)

class ErasureDecoder:
    """Matching decoder that uses the heralds: a heralded erasure is an X flip and a Z flip
    each with probability one half at a known place, so its edges get weight zero for
    that shot only."""
    def __init__(self, circuit, is_herald):
        import pymatching
        self.is_herald = is_herald
        node = np.cumsum(~is_herald) - 1
        herald_id = np.cumsum(is_herald) - 1
        dem = circuit.detector_error_model(approximate_disjoint_errors=True).flattened()
        edges, cond, self.skipped = {}, {}, 0
        for inst in dem:
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
            obs = frozenset(t.val for t in inst.targets_copy() if t.is_logical_observable_id())
            heralds = [int(herald_id[d]) for d in dets if is_herald[d]]
            key = tuple(sorted(int(node[d]) for d in dets if not is_herald[d]))
            if len(key) == 0:
                continue
            if heralds:
                cond.setdefault(heralds[0], []).append(key)
            elif len(key) > 2:
                self.skipped += 1
            else:
                p0, o0 = edges.get(key, (0.0, obs))
                edges[key] = (p0 * (1 - p) + p * (1 - p0), o0)
        self.cond = {}
        for k, keys in cond.items():                     # the Y outcome is the union of X and Z: drop it
            keys = sorted(set(keys), key=len)
            self.cond[k] = [q for q in (keys[:-1] if len(keys) == 3 else keys) if len(q) <= 2 and q in edges]
        self.edges = {k: (float(np.log((1 - p) / p)), o) for k, (p, o) in edges.items()}
        self.m = pymatching.Matching()
        for key, (w, o) in self.edges.items():
            self._set(key, w, o, "disallow")

    def _set(self, key, w, o, merge):
        if len(key) == 2:
            self.m.add_edge(key[0], key[1], fault_ids=set(o), weight=w, merge_strategy=merge)
        else:
            self.m.add_boundary_edge(key[0], fault_ids=set(o), weight=w, merge_strategy=merge)

    def decode(self, dets):
        her, syn = dets[:, self.is_herald], dets[:, ~self.is_herald][:, : self.m.num_detectors]
        clean = ~her.any(axis=1)
        nobs = max(self.m.num_fault_ids, 1)
        pred = np.zeros((len(dets), nobs), dtype=np.uint8)
        if clean.any():
            pred[clean] = self.m.decode_batch(syn[clean])[:, :nobs]
        for s in np.flatnonzero(~clean):
            touched = {q for k in np.flatnonzero(her[s]) for q in self.cond.get(int(k), ())}
            for q in touched:
                self._set(q, 0.0, self.edges[q][1], "replace")
            pred[s] = self.m.decode(syn[s])[:nobs]
            for q in touched:
                self._set(q, self.edges[q][0], self.edges[q][1], "replace")
        return pred

def _rates(chans):
    # Y errors are applied as an X and a Z error of the same probability (uncorrelated matching)
    p_x = np.array([c["p_x"] + c["p_y"] + P_CHECK for c in chans])
    p_z = np.array([c["p_z"] + c["p_y"] + P_CHECK for c in chans])
    p_e = np.clip(np.array([c["p_erase"] for c in chans]), 1e-6, 1.0)
    return p_x, p_z, p_e

def logical_errors(distance, basis_name, true_channels, prior_channels, shots, seed):
    """Errors are sampled from the true channels; the decoder is built from the prior
    channels. Every data qubit draws its bath at random from the unseen baths; N_DRAWS
    placements per point, the same placements for all arms and clock periods."""
    n_data, errors = distance * distance, 0
    for j in range(N_DRAWS):
        draw = np.random.default_rng(SEED + 1000 * distance + j).integers(0, len(true_channels), n_data)
        circuit, is_herald = build_circuit(distance, basis_name, *_rates([true_channels[k] for k in draw]), P_MEAS)
        prior, is_herald2 = build_circuit(distance, basis_name, *_rates([prior_channels[k] for k in draw]), P_MEAS)
        assert np.array_equal(is_herald, is_herald2)
        dets, obs = circuit.compile_detector_sampler(seed=seed + 7919 * j).sample(shots // N_DRAWS,
                                                                                 separate_observables=True)
        dec = ErasureDecoder(prior, is_herald)
        assert dec.skipped == 0, "non graphlike error found"
        pred = dec.decode(dets)
        errors += int(np.sum(np.any(pred[:, : obs.shape[1]] != obs, axis=1)))
    return errors

# ================================ MAIN =================================
def run_case(GAMMA_SYS):
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score, mean_absolute_error
    from xgboost import XGBRegressor
    outdir = f"{OUTDIR}_gamma_sys_{GAMMA_SYS}"
    cache = f"gamma_sys_{GAMMA_SYS}_{CACHE}"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    full = lambda p: tuple(p) + (GAMMA_SYS,)
    print("\n" + "=" * 78)
    print(f"GAMMA_SYS = {GAMMA_SYS}, ATTACH = {ATTACH}, BETA = {BETA}, n_th = {N_TH:.2e}")
    print("=" * 78)

    # ---- Step 0: the sector solution against the full 7 qubit model ----
    p_val = full(0.5 * (lo + hi) for lo, hi in PARAM_RANGES)
    t0 = time.time()
    print(f"Sector against full 7 qubit QuTiP, largest difference in the logical Bloch vector: "
          f"{validate_against_full(p_val):.2e} ({time.time() - t0:.0f} s)")

    # ---- Step 1: calibration signals and FFT features ----
    if os.path.exists(cache) and len(np.load(cache)["P"]) == N_TRAIN + N_TEST:
        d = np.load(cache)
        P_all, X_all, S_all = d["P"], d["X"], d["S"]
        print(f"Loaded {len(P_all)} cached baths from {cache}")
    else:
        P_all = np.array([[rng.uniform(lo, hi) for lo, hi in PARAM_RANGES] for _ in range(N_TRAIN + N_TEST)])
        t0 = time.time()
        S_all = np.array(Parallel(n_jobs=-1)(delayed(calibration_signal)(full(p)) for p in P_all))
        X_all = np.array([fft_features(sig) for sig in S_all])
        np.savez_compressed(cache, P=P_all, X=X_all, S=S_all)
        print(f"Simulated {len(P_all)} baths in {time.time() - t0:.0f} s")
    P_tr, X_tr, P_te, X_te = P_all[:N_TRAIN], X_all[:N_TRAIN], P_all[N_TRAIN:], X_all[N_TRAIN:]

    # ---- Step 2: PCA + XGBoost, then the fit that refines the ML estimate ----
    def score(method, pred):
        out = []
        for j, name in enumerate(PARAM_NAMES):
            mae = mean_absolute_error(P_te[:, j], pred[:, j])
            out.append({"Method": method, "Parameter": name, "R2": r2_score(P_te[:, j], pred[:, j]), "MAE": mae,
                        "MAE_percent_of_range": 100 * mae / (PARAM_RANGES[j][1] - PARAM_RANGES[j][0])})
        return out
    rows1, P_hat, P_hat_train = [], {}, {}
    for n_pca in PCA_LIST:
        pca = FeaturePCA(n_components=min(n_pca, N_TRAIN - 1), random_state=0).fit(X_tr)
        Z_tr, Z_te = pca.transform(X_tr), pca.transform(X_te)
        pred, pred_tr = np.zeros_like(P_te), np.zeros_like(P_tr)
        for j, name in enumerate(PARAM_NAMES):
            model = XGBRegressor(**XGB).fit(Z_tr, P_tr[:, j])
            pred[:, j] = np.clip(model.predict(Z_te), *PARAM_RANGES[j])
            pred_tr[:, j] = np.clip(model.predict(Z_tr), *PARAM_RANGES[j])   # raw data only (in sample)
        P_hat[n_pca], P_hat_train[n_pca] = pred, pred_tr
        rows1 += score(f"ML, PCA {n_pca}", pred)
    t0 = time.time()
    signals = Parallel(n_jobs=-1)(delayed(calibration_signal)(full(p), N_FIT_POINTS) for p in P_te)
    P_fit = np.array(Parallel(n_jobs=-1)(delayed(refine_by_fit)(P_hat[PCA_USED][k], signals[k], GAMMA_SYS)
                                         for k in range(N_TEST)))
    rows1 += score(f"ML, PCA {PCA_USED}, then fit", P_fit)
    # raw data: the same fit for the training baths (not used by any table)
    P_fit_train = np.array(Parallel(n_jobs=-1)(delayed(refine_by_fit)(P_hat_train[PCA_USED][k],
                                                                      S_all[k][:, :N_FIT_POINTS], GAMMA_SYS)
                                               for k in range(min(N_TRAIN, N_RAW_FIT)))) if SAVE_RAW else None
    if P_fit_train is not None and len(P_fit_train) < N_TRAIN:      # the rest: empty cells
        P_fit_train = np.vstack([P_fit_train, np.full((N_TRAIN - len(P_fit_train), P_fit_train.shape[1]), np.nan)])
    df1 = pd.DataFrame(rows1)
    print(f"\nTable 1. Bath parameters inferred from the dual rail signal "
          f"({N_TRAIN} training baths, {N_TEST} unseen baths; fit took {time.time() - t0:.0f} s)")
    print(df1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- Step 3: one round per bath ----
    t0 = time.time()
    jobs = [(k, tc) for tc in T_CLOCK_LIST for k in range(N_TEST)]
    res = Parallel(n_jobs=-1)(delayed(device_arms)(full(P_te[k]), full(P_hat[PCA_USED][k]), full(P_fit[k]), tc)
                              for k, tc in jobs)
    arms = list(res[0].keys())
    true_ch = {(tc, arm): [] for tc in T_CLOCK_LIST for arm in arms}
    prior_ch = {(tc, arm): [] for tc in T_CLOCK_LIST for arm in arms}
    for (k, tc), r in zip(jobs, res):
        for arm in arms:
            true_ch[(tc, arm)].append(r[arm]["true"])
            prior_ch[(tc, arm)].append(r[arm]["prior"])
    for tc in T_CLOCK_LIST:                      # the uniform decoder knows only the chip average
        chans = true_ch[(tc, ARM_CLOCK_UNIFORM)]
        mean = {key: float(np.mean([c[key] for c in chans])) for key in ("p_erase", "p_x", "p_y", "p_z")}
        prior_ch[(tc, ARM_CLOCK_UNIFORM)] = [mean] * len(chans)
    rows2 = []
    for (tc, arm), chans in true_ch.items():
        rows2.append({"T_clock": tc, "Arm": arm, "Mean_check_time": np.mean([c["tau"] for c in chans]),
                      "p_erase": np.mean([c["p_erase"] for c in chans]),
                      "p_pauli_unflagged": np.mean([c["p_x"] + c["p_y"] + c["p_z"] for c in chans]),
                      "Unflagged_process_fidelity": np.mean([c["chi_I"] for c in chans])})
    df2 = pd.DataFrame(rows2)
    print(f"\nTable 2. One round per data qubit, mean over {N_TEST} unseen baths ({time.time() - t0:.0f} s)")
    print(df2.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ---- Step 4: surface code logical error rate ----
    t0 = time.time()
    pts = [(tc, arm, b, d) for tc in T_CLOCK_LIST for arm in arms for b in BASES for d in DISTANCES]
    errs = Parallel(n_jobs=-1)(delayed(logical_errors)(d, b, true_ch[(tc, arm)], prior_ch[(tc, arm)], SHOTS,
                                                       SEED + i)
                               for i, (tc, arm, b, d) in enumerate(pts))
    shots = (SHOTS // N_DRAWS) * N_DRAWS
    rows3 = []
    for (tc, arm, b, d), e in zip(pts, errs):
        tau = np.mean([c["tau"] for c in true_ch[(tc, arm)]])
        ler = e / shots
        rows3.append({"T_clock": tc, "Arm": arm, "Memory_basis": b.upper(), "Distance": d,
                      "Logical_errors": e, "Shots": shots, "LER": ler,
                      "LER_per_unit_time": 1 - (1 - min(ler, 0.5)) ** (1.0 / (d * tau))})
    df3 = pd.DataFrame(rows3)
    print(f"\nTable 3. Surface code logical error rate, {shots} shots per point over {N_DRAWS} bath placements "
          f"({time.time() - t0:.0f} s)")
    print(df3.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    df1.to_csv(os.path.join(outdir, "table1_inference.csv"), index=False)
    df2.to_csv(os.path.join(outdir, "table2_round_channels.csv"), index=False)
    df3.to_csv(os.path.join(outdir, "table3_surface_code.csv"), index=False)
    print(f"\nSaved tables 1 to 3 in {outdir}/")

    # ---- Raw data behind the tables ----
    if SAVE_RAW:
        raw = pd.DataFrame({"Bath": np.arange(N_TRAIN + N_TEST),
                            "Set": ["training"] * N_TRAIN + ["unseen"] * N_TEST})
        for j, name in enumerate(PARAM_NAMES):
            raw[f"true_{name}"] = P_all[:, j]
        for j, name in enumerate(PARAM_NAMES):
            raw[f"regression_{name}"] = np.concatenate([P_hat_train[PCA_USED][:, j], P_hat[PCA_USED][:, j]])
        for j, name in enumerate(PARAM_NAMES):
            raw[f"fit_{name}"] = np.concatenate([P_fit_train[:, j], P_fit[:, j]])
        raw.to_csv(os.path.join(outdir, "raw_parameters_per_bath.csv"), index=False)
        rows = []
        for (k, tc), r in zip(jobs, res):
            for arm in arms:
                c = r[arm]["true"]
                rows.append({"Bath": N_TRAIN + k, "T_clock": tc, "Arm": arm, "Check_time": c["tau"],
                             "p_erase": c["p_erase"], "p_x": c["p_x"], "p_y": c["p_y"], "p_z": c["p_z"],
                             "Unflagged_process_fidelity": c["chi_I"]})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "raw_round_channels_per_bath.csv"), index=False)
        np.savez_compressed(os.path.join(outdir, "raw_signals_and_spectra.npz"),
                            time=np.linspace(0, T_MAX, N_POINTS), parameter_names=np.array(PARAM_NAMES),
                            true_parameters=P_all, signal_xyz=S_all.astype(np.float32),
                            fft_features=X_all.astype(np.float32))
        print(f"Saved the raw data of all {N_TRAIN + N_TEST} baths in {outdir}/ (raw_parameters_per_bath.csv, "
              f"raw_round_channels_per_bath.csv, raw_signals_and_spectra.npz)")

def main():
    for g in GAMMA_SYS_LIST:
        run_case(g)

if __name__ == "__main__":
    main()
