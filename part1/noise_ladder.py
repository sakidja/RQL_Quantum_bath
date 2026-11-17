# ======================================================================
# Noise ladder (version 3): the structured bath method with a noisy
# calibration signal, the ML trained on noisy signals.
#
# Everything is taken from dual_rail_structured_bath.py (version 3,
# split rail attachment): the same baths (same seed), the same
# inference (FFT features, PCA, XGBoost, then the fit), the same
# balanced check rule, the same surface code.
#
# On every rung of the ladder four kinds of noise act together:
#   shots    : each point of each logical Bloch component is estimated
#              from a finite number of shots (+1, -1, or "erased")
#   readout  : every +1 or -1 result is flipped with this probability
#   prep     : the logical plus state is prepared with a random tilt and
#              a random phase (radians, standard deviation = this value),
#              different for every bath and not known to the inference
#   drift    : between the calibration and the run every bath parameter
#              moves by a random relative amount (standard deviation =
#              this value). The inference sees the bath of the
#              calibration; the errors of the run come from the drifted
#              bath. The clock arm and the bound are subject to the same
#              drift (their frame is calibrated before the drift).
#
# Two feature definitions, selected with the environment variable SQB_FEATURE:
#
#   published (default): the log spectrum of the notebook of arXiv:2508.17514,
#       normalized between its own lowest and highest value. The lowest value
#       of an exact signal is a numerical floor, that of a measured signal is
#       the noise floor, so exact and measured signals are normalized
#       differently. The regression that drives the fit and the arms is
#       trained on signals carrying the noise of the same rung; the regression
#       trained on exact signals is listed in Table L1 for comparison.
#       This mode reproduces results_v3_ladder_<attachment>_gamma_sys_*.
#
#   fixed_floor: the same log spectrum, normalized to its peak and cut at a
#       floor fixed at the noise amplitude of the measurement,
#       FLOOR_FACTOR * 2 / sqrt(N_POINTS * shots), which is known from the
#       number of shots. Exact and measured signals are then normalized in
#       the same way. The regression that drives the fit and the arms is
#       trained on EXACT signals only; the regression trained on noisy
#       signals is listed in Table L1 for comparison.
#       Results go to results_v3_ladder_fixedfloor_<attachment>_gamma_sys_*.
#
#   python noise_ladder.py
#   SQB_FEATURE=fixed_floor python noise_ladder.py
#
# AI (LLM) assistance (Claude, Anthropic) was used in preparing this
# code. The framework, calculations and analysis are the author's.
# ======================================================================
import os
import time
import numpy as np
import pandas as pd
from scipy.linalg import expm
import dual_rail_structured_bath as m
m.N_FIT_POINTS = m.N_POINTS          # the fit reads the whole record, as the regression does

# ============================== SETTINGS ===============================
#          name          shots    readout  prep    drift
LADDER = (("exact",      None,    0.0,     0.0,    0.0),
          ("rung 1",     10000,   0.005,   0.005,  0.001),
          ("rung 2",     1000,    0.01,    0.01,   0.003),
          ("rung 3",     1000,    0.02,    0.02,   0.01))
# The rungs are those of a real calibration (1000 to 10000 shots per point). The earlier
# rungs of 300 and 100 shots per point, used by no hardware calibration, are removed.
GAMMA_SYS_LIST = (0.0,)
T_CLOCK_LIST = (2.0, 4.0)
N_TRAIN, N_TEST = m.N_TRAIN, m.N_TEST
SHOTS = m.SHOTS                      # surface code shots per point
CONTRAST_RANGE = (0.5, 1.1)          # the fit has one more free number, the contrast of the signal
FEATURE = os.environ.get("SQB_FEATURE", "published")    # "published" or "fixed_floor" (see the header)
assert FEATURE in ("published", "fixed_floor")
FIXED = FEATURE == "fixed_floor"
FLOOR_FACTOR = 1.0                   # fixed_floor: the floor in units of the noise amplitude 2 / sqrt(N_POINTS * shots)
FLOOR_DECADES = 3.0                  # fixed_floor: decades of the spectrum mapped onto the feature range 0 to 1
OUTDIR = (f"results_v3_ladder_fair_fixedfloor_{m.ATTACH}" if FIXED
          else f"results_v3_ladder_fair_{m.ATTACH}")
TRAINED_ON = "exact" if FIXED else "noisy"     # the training of the regression that drives the fit and the arms
LABEL_FIT = f"ML (trained on {TRAINED_ON} signals), then fit"
SAVE_RAW = m.SAVE_RAW                # write the raw data of every bath on every rung
ARM_CLOCK = m.ARM_CLOCK
ARM_ML = "Structured bath, ML inferred, balanced check"
ARM_FIT = "Structured bath, ML then fit, balanced check"
ARM_EXACT = "Structured bath, exact at calibration, balanced check (bound)"
ARMS = (ARM_CLOCK, ARM_ML, ARM_FIT, ARM_EXACT)

# ======================== NOISY CALIBRATION SIGNAL ======================
INPUT_TILTS = [m.INPUT_TILTS[n] for n in m.INPUTS_USED]   # 0, -pi/4, +pi/4 of the main script


def exact_signal(params, tilt=0.0, phase=0.0):
    """Logical <X>, <Y>, <Z> and the probability of being on the rails, on the paper's grid,
    from every input state in m.INPUTS_USED: rows 4i to 4i+3 belong to input i, whose state
    is cos(pi/4 + t0 + tilt)|0> + exp(i phase) sin(pi/4 + t0 + tilt)|1> with t0 the nominal
    tilt of that input. tilt and phase are the preparation error of the rung, added to every
    input."""
    tl = np.linspace(0, m.T_MAX, m.N_POINTS)
    step = expm(m.liouvillian(params) * (tl[1] - tl[0]))
    out = np.zeros((4 * len(INPUT_TILTS), m.N_POINTS))
    for i, t0 in enumerate(INPUT_TILTS):
        a = (np.cos(np.pi / 4 + t0 + tilt), np.exp(1j * phase) * np.sin(np.pi / 4 + t0 + tilt))
        v = sum(a[p] * np.conj(a[q]) * m.logical_input(p, q) for p in (0, 1) for q in (0, 1))
        for k in range(m.N_POINTS):
            rho = v.reshape(m.DIM, m.DIM)
            out[4 * i + 0, k] = 2 * rho[m.I_R, m.I_L].real
            out[4 * i + 1, k] = 2 * rho[m.I_R, m.I_L].imag
            out[4 * i + 2, k] = (rho[m.I_L, m.I_L] - rho[m.I_R, m.I_R]).real
            out[4 * i + 3, k] = (rho[m.I_L, m.I_L] + rho[m.I_R, m.I_R]).real
            v = step @ v
    return out

def measured_signal(sig4, n_shots, readout, rng):
    """Estimate of the three Bloch components of every input from n_shots shots per point
    with readout flips. Row 3i + c is component c of input i, the row order of
    m.calibration_signal, so the fit compares like with like."""
    n_in = sig4.shape[0] // 4
    out = np.empty((3 * n_in, sig4.shape[1]))
    for i in range(n_in):
        pc = np.clip(sig4[4 * i + 3], 0.0, 1.0)
        for c in range(3):
            if n_shots is None:
                out[3 * i + c] = (1 - 2 * readout) * sig4[4 * i + c]
                continue
            mean = (1 - 2 * readout) * np.clip(sig4[4 * i + c], -pc, pc)
            pv = np.clip(np.stack([(pc + mean) / 2, (pc - mean) / 2, 1 - pc], axis=1), 0.0, 1.0)
            pv /= pv.sum(axis=1, keepdims=True)
            k = rng.multinomial(n_shots, pv)
            out[3 * i + c] = (k[:, 0] - k[:, 1]) / n_shots
    return out

def bath_job(p_full, rung, seed):
    """Noisy calibration signal of one bath: features, the part used by the fit, and the
    exact features (for the regression trained without noise)."""
    _, n_shots, readout, prep, _ = rung
    rng = np.random.default_rng(seed)
    tilt, phase = rng.normal(0.0, prep, 2) if prep > 0 else (0.0, 0.0)
    noisy = measured_signal(exact_signal(p_full, tilt, phase), n_shots, readout, rng)
    return m.fft_features(noisy), noisy

def fixed_floor_features(signals, n_shots):
    """Log spectrum normalized to its peak and cut at a floor fixed at the noise amplitude of
    n_shots shots per point. signals: array (baths, 3, N_POINTS). One peak per bath (over the
    three components), so the contrast lost to the readout error drops out."""
    S = np.asarray(signals, dtype=float)
    S = S - S.mean(axis=-1, keepdims=True)
    A = 2 * np.abs(np.fft.fft(S, axis=-1))[..., : m.N_POINTS // 2] / m.N_POINTS
    peak = A.max(axis=(1, 2), keepdims=True)
    floor = FLOOR_FACTOR * (2 / np.sqrt(m.N_POINTS * n_shots)) / peak
    F = np.clip(np.log10(np.maximum(A / peak, floor) / floor) / FLOOR_DECADES, 0.0, 1.0)
    return np.hstack([F.reshape(len(S), -1), np.array([m.extra_features(s) for s in S])])

SCAN_N = int(os.environ.get("SQB_LADDER_SCAN", "5"))      # grid points per horizontal bond in the scan
FULL_STAGES = tuple(int(v) for v in os.environ.get("SQB_LADDER_FULL", f"2500,{m.N_POINTS}").split(","))
PREP_BOUND = 0.3                     # bound (rad) of the fitted tilt and phase of the preparation
KEEP_MID = int(os.environ.get("SQB_LADDER_KEEP_MID", "4"))     # candidates continued to 1000 points
KEEP_FULL = int(os.environ.get("SQB_LADDER_KEEP_FULL", "2"))   # candidates polished on the whole record
_KC, _KL, _KR = m.I_R * m.DIM + m.I_L, m.I_L * m.DIM + m.I_L, m.I_R * m.DIM + m.I_R


def _model(x, n, gamma_sys):
    """Bloch rows of every input (the row order of m.calibration_signal), first n points, for
    the bath x[:6], with three more free numbers: the contrast x[6] (readout error), and the
    tilt x[7] and the phase x[8] of the preparation, common to all inputs as in exact_signal."""
    tl = np.linspace(0, m.T_MAX, m.N_POINTS)
    step = expm(m.liouvillian(tuple(x[:6]) + (gamma_sys,)) * (tl[1] - tl[0]))
    cols = []
    for t0 in INPUT_TILTS:
        c = (np.cos(np.pi / 4 + t0 + x[7]), np.exp(1j * x[8]) * np.sin(np.pi / 4 + t0 + x[7]))
        cols.append(sum(c[p] * np.conj(c[q]) * m.logical_input(p, q) for p in (0, 1) for q in (0, 1)))
    v = np.stack(cols, axis=1)
    out = np.zeros((3 * len(cols), n))
    for k in range(n):
        out[0::3, k] = 2 * v[_KC].real
        out[1::3, k] = 2 * v[_KC].imag
        out[2::3, k] = (v[_KL] - v[_KR]).real
        v = step @ v
    return x[6] * out


def _fit_stages(q, signal, gamma_sys, lo, hi, stages):
    """Staged least squares over the given record lengths. Returns the parameters and the cost."""
    from scipy.optimize import least_squares
    q = np.clip(q, lo, hi)
    res = None
    for n in stages:
        n = min(n, signal.shape[1])
        res = least_squares(lambda x: (_model(x, n, gamma_sys) - signal[:, :n]).ravel(), x0=q, bounds=(lo, hi),
                            x_scale=hi - lo, diff_step=1e-6, max_nfev=200)
        q = res.x
    return q, res.cost


def fit_with_contrast(p_start, signal, gamma_sys):
    """ML then fit. The fit is given what the record contains: the six bath parameters, the
    contrast of the signal (the readout error is not given to the fit), and the tilt and the
    phase of the preparation (the preparation error is not given to the fit either).
    Search: the regression estimate and a grid of SCAN_N x SCAN_N values of the two horizontal
    bonds J_L1 and J_L2 (the other numbers at the regression estimate) are fitted on the short
    record; the best KEEP_MID continue to 1000 points; the best KEEP_FULL are polished on the whole
    record; the lowest cost is kept."""
    lo = np.array([r[0] for r in m.PARAM_RANGES] + [CONTRAST_RANGE[0], -PREP_BOUND, -PREP_BOUND])
    hi = np.array([r[1] for r in m.PARAM_RANGES] + [CONTRAST_RANGE[1], PREP_BOUND, PREP_BOUND])
    w = hi - lo
    q0 = np.concatenate([np.asarray(p_start, dtype=float)[:6], [1.0, 0.0, 0.0]])
    cands = [q0]
    for u in (np.arange(SCAN_N) + 0.5) / SCAN_N:
        for t in (np.arange(SCAN_N) + 0.5) / SCAN_N:
            q = q0.copy()
            q[1], q[3] = lo[1] + u * w[1], lo[3] + t * w[3]
            cands.append(q)
    short, mid = m.FIT_STAGES[:4], m.FIT_STAGES[4:]
    c = sorted((_fit_stages(q, signal, gamma_sys, lo, hi, short) for q in cands), key=lambda r: r[1])
    if mid:
        c = sorted((_fit_stages(r[0], signal, gamma_sys, lo, hi, mid) for r in c[:KEEP_MID]), key=lambda r: r[1])
    best = min((_fit_stages(r[0], signal, gamma_sys, lo, hi, FULL_STAGES) for r in c[:KEEP_FULL]), key=lambda r: r[1])
    return best[0][:6]


def arms_job(p_run, p_cal, p_ml, p_fit, t_clock):
    """One data qubit at one clock period. The errors come from the bath of the run (p_run);
    every arm works from what it knew at calibration."""
    L_run = m.liouvillian(p_run)
    def arm(p_model):
        t = m.best_check_time(p_model, t_clock, "balance")
        U = m.correction(p_model, t)
        return {"true": dict(m.round_channel(L_run, t, U), tau=t),
                "prior": dict(m.round_channel(m.liouvillian(p_model), t, U), tau=t)}
    U = m.correction(p_cal, t_clock)
    return {ARM_CLOCK: {"true": dict(m.round_channel(L_run, t_clock, U), tau=t_clock),
                        "prior": dict(m.round_channel(m.liouvillian(p_cal), t_clock, U), tau=t_clock)},
            ARM_ML: arm(p_ml), ARM_FIT: arm(p_fit), ARM_EXACT: arm(p_cal)}

# ================================ MAIN =================================
def run_case(gamma_sys):
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor
    outdir = f"{OUTDIR}_gamma_sys_{gamma_sys}"
    os.makedirs(outdir, exist_ok=True)
    full = lambda p: tuple(p) + (gamma_sys,)
    rng = np.random.default_rng(m.SEED)          # the same baths as in the main run
    P_all = np.array([[rng.uniform(lo, hi) for lo, hi in m.PARAM_RANGES] for _ in range(N_TRAIN + N_TEST)])
    P_tr, P_te = P_all[:N_TRAIN], P_all[N_TRAIN:]
    lo = np.array([r[0] for r in m.PARAM_RANGES])
    hi = np.array([r[1] for r in m.PARAM_RANGES])
    print("\n" + "=" * 78)
    print(f"NOISE LADDER, GAMMA_SYS = {gamma_sys}, ATTACH = {m.ATTACH}, FEATURE = {FEATURE}, "
          f"{N_TRAIN} training baths, {N_TEST} unseen baths; the fit and the arms start from the regression "
          f"trained on {TRAINED_ON} signals")
    print("=" * 78)

    def regress(X_train, X_test, with_train=False):
        pca = m.FeaturePCA(n_components=min(m.PCA_USED, N_TRAIN - 1), random_state=0).fit(X_train)
        Z_tr, Z_te = pca.transform(X_train), pca.transform(X_test)
        P, P_in = np.zeros_like(P_te), np.zeros_like(P_tr)
        for j in range(len(m.PARAM_NAMES)):
            model = XGBRegressor(**m.XGB).fit(Z_tr, P_tr[:, j])
            P[:, j] = np.clip(model.predict(Z_te), *m.PARAM_RANGES[j])
            P_in[:, j] = np.clip(model.predict(Z_tr), *m.PARAM_RANGES[j])    # raw data only (in sample)
        return (P, P_in) if with_train else P

    rows1, rows2, rows3, X_exact, S_exact, raw_par, raw_ch = [], [], [], None, None, [], []
    for li, rung in enumerate(LADDER):
        name, n_shots, readout, prep, drift = rung
        t0 = time.time()
        res = Parallel(n_jobs=-1)(delayed(bath_job)(full(p), rung, [m.SEED, 31, li, i]) for i, p in enumerate(P_all))
        X = np.array([r[0] for r in res])
        S = np.array([r[1] for r in res])           # the noisy signals of all baths (raw data)
        if S_exact is None:
            S_exact = S                              # the first rung is the exact signal
        if FIXED and n_shots is not None:           # the same floor for the exact and the measured signals
            X = fixed_floor_features(S, n_shots)
            X_exact = fixed_floor_features(S_exact, n_shots)
        elif X_exact is None or FIXED:
            X_exact = X                              # published feature; on the exact rung both modes use it
        sig_te = [r[1][:, : m.N_FIT_POINTS] for r in res[N_TRAIN:]]
        P_ml_noisy, P_ml_noisy_train = regress(X[:N_TRAIN], X[N_TRAIN:], True)       # trained on noisy signals
        P_ml_clean = regress(X_exact[:N_TRAIN], X[N_TRAIN:])                         # trained on exact signals
        P_ml_clean_train = None
        if FIXED:                                   # the regression trained on exact signals drives the fit and the arms
            pca_c = m.FeaturePCA(n_components=min(m.PCA_USED, N_TRAIN - 1), random_state=0).fit(X_exact[:N_TRAIN])
            Zc_tr, Zc_meas = pca_c.transform(X_exact[:N_TRAIN]), pca_c.transform(X[:N_TRAIN])
            P_ml_clean_train = np.zeros_like(P_tr)   # the measured signals of the training baths (raw data only)
            for j in range(len(m.PARAM_NAMES)):
                P_ml_clean_train[:, j] = np.clip(XGBRegressor(**m.XGB).fit(Zc_tr, P_tr[:, j]).predict(Zc_meas),
                                                 *m.PARAM_RANGES[j])
            P_ml, P_ml_train = P_ml_clean, P_ml_clean_train
        else:
            P_ml, P_ml_train = P_ml_noisy, P_ml_noisy_train
        P_fit = np.array(Parallel(n_jobs=-1)(delayed(fit_with_contrast)(P_ml[k], sig_te[k], gamma_sys)
                                             for k in range(N_TEST)))
        for label, P in (("ML, trained on exact signals", P_ml_clean), ("ML, trained on noisy signals", P_ml_noisy),
                         (LABEL_FIT, P_fit)):
            for j, pname in enumerate(m.PARAM_NAMES):
                mae = np.mean(np.abs(P[:, j] - P_te[:, j]))
                rows1.append({"Rung": name, "Shots_per_point": "exact" if n_shots is None else n_shots,
                              "Readout_error": readout, "Prep_error": prep, "Drift": drift, "Method": label,
                              "Parameter": pname, "R2": r2_score(P_te[:, j], P[:, j]), "MAE": mae,
                              "MAE_percent_of_range": 100 * mae / (hi[j] - lo[j])})

        if SAVE_RAW:                                # raw data: the same fit for the training baths
            P_fit_train = np.array(Parallel(n_jobs=-1)(delayed(fit_with_contrast)(
                P_ml_train[k], S[k][:, : m.N_FIT_POINTS], gamma_sys) for k in range(min(N_TRAIN, m.N_RAW_FIT))))
            if len(P_fit_train) < N_TRAIN:                     # the rest: empty cells
                P_fit_train = np.vstack([P_fit_train, np.full((N_TRAIN - len(P_fit_train), P_fit_train.shape[1]), np.nan)])
        drng = np.random.default_rng([m.SEED, 47, li])
        P_run = np.clip(P_te * (1 + drift * drng.standard_normal(P_te.shape)), 0.0, None)   # the bath of the run
        jobs = [(k, tc) for tc in T_CLOCK_LIST for k in range(N_TEST)]
        out = Parallel(n_jobs=-1)(delayed(arms_job)(full(P_run[k]), full(P_te[k]), full(P_ml[k]), full(P_fit[k]), tc)
                                  for k, tc in jobs)
        chan = {}
        for (k, tc), r in zip(jobs, out):
            for arm in ARMS:
                chan.setdefault((tc, arm), {"true": [], "prior": []})
                chan[(tc, arm)]["true"].append(r[arm]["true"])
                chan[(tc, arm)]["prior"].append(r[arm]["prior"])
        head = {"Rung": name, "Shots_per_point": "exact" if n_shots is None else n_shots,
                "Readout_error": readout, "Prep_error": prep, "Drift": drift}
        for tc in T_CLOCK_LIST:
            for arm in ARMS:
                c = chan[(tc, arm)]["true"]
                rows2.append(dict(head, T_clock=tc, Arm=arm, Mean_check_time=np.mean([x["tau"] for x in c]),
                                  p_erase=np.mean([x["p_erase"] for x in c]),
                                  p_pauli_unflagged=np.mean([x["p_x"] + x["p_y"] + x["p_z"] for x in c])))
        pts = [(tc, arm, b, d) for tc in T_CLOCK_LIST for arm in ARMS for b in m.BASES for d in m.DISTANCES]
        errs = Parallel(n_jobs=-1)(delayed(m.logical_errors)(d, b, chan[(tc, arm)]["true"], chan[(tc, arm)]["prior"],
                                                             SHOTS, m.SEED + 17 * i)
                                   for i, (tc, arm, b, d) in enumerate(pts))
        shots = (SHOTS // m.N_DRAWS) * m.N_DRAWS
        for (tc, arm, b, d), e in zip(pts, errs):
            tau = np.mean([x["tau"] for x in chan[(tc, arm)]["true"]])
            rows3.append(dict(head, T_clock=tc, Arm=arm, Memory_basis=b.upper(), Distance=d, Logical_errors=e,
                              Shots=shots, LER=e / shots,
                              LER_per_unit_time=1 - (1 - min(e / shots, 0.5)) ** (1.0 / (d * tau))))
        if SAVE_RAW:
            tag = name.replace(" ", "_")
            raw = pd.DataFrame({"Rung": name, "Bath": np.arange(N_TRAIN + N_TEST),
                                "Set": ["training"] * N_TRAIN + ["unseen"] * N_TEST})
            nan = np.full(N_TRAIN, np.nan)
            for j, pname in enumerate(m.PARAM_NAMES):
                raw[f"true_{pname}"] = P_all[:, j]
            for j, pname in enumerate(m.PARAM_NAMES):
                raw[f"run_after_drift_{pname}"] = np.concatenate([nan, P_run[:, j]])
            for j, pname in enumerate(m.PARAM_NAMES):
                raw[f"regression_trained_noisy_{pname}"] = np.concatenate([P_ml_noisy_train[:, j], P_ml_noisy[:, j]])
            for j, pname in enumerate(m.PARAM_NAMES):
                raw[f"regression_trained_exact_{pname}"] = np.concatenate(
                    [nan if P_ml_clean_train is None else P_ml_clean_train[:, j], P_ml_clean[:, j]])
            for j, pname in enumerate(m.PARAM_NAMES):
                raw[f"fit_{pname}"] = np.concatenate([P_fit_train[:, j], P_fit[:, j]])
            raw_par.append(raw)
            for (k, tc), r in zip(jobs, out):
                for arm in ARMS:
                    c = r[arm]["true"]
                    raw_ch.append({"Rung": name, "Bath": N_TRAIN + k, "T_clock": tc, "Arm": arm, "Check_time": c["tau"],
                                   "p_erase": c["p_erase"], "p_x": c["p_x"], "p_y": c["p_y"], "p_z": c["p_z"]})
            np.savez_compressed(os.path.join(outdir, f"raw_signals_and_spectra_{tag}.npz"),
                                time=np.linspace(0, m.T_MAX, m.N_POINTS), parameter_names=np.array(m.PARAM_NAMES),
                                true_parameters=P_all, signal_xyz=S.astype(np.float32),
                                fft_features=X.astype(np.float32),
                                fft_features_of_exact_signals=X_exact.astype(np.float32),
                                feature=np.array(FEATURE))
        print(f"  {name} (shots {n_shots}, readout {readout}, prep {prep}, drift {drift}): done ({time.time() - t0:.0f} s)")

    df1, df2, df3 = pd.DataFrame(rows1), pd.DataFrame(rows2), pd.DataFrame(rows3)
    print("\nTable L1. Bath parameters recovered on every rung (unseen baths)")
    print(df1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nTable L2. One round per data qubit on every rung")
    print(df2.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nTable L3. Surface code logical error rate on every rung")
    print(df3.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    df1.to_csv(os.path.join(outdir, "tableL1_inference_on_ladder.csv"), index=False)
    df2.to_csv(os.path.join(outdir, "tableL2_round_on_ladder.csv"), index=False)
    df3.to_csv(os.path.join(outdir, "tableL3_surface_code_on_ladder.csv"), index=False)
    print(f"\nSaved tables L1 to L3 in {outdir}/")
    if SAVE_RAW:
        pd.concat(raw_par).to_csv(os.path.join(outdir, "raw_parameters_per_bath.csv"), index=False)
        pd.DataFrame(raw_ch).to_csv(os.path.join(outdir, "raw_round_channels_per_bath.csv"), index=False)
        print(f"Saved the raw data of all {N_TRAIN + N_TEST} baths on every rung in {outdir}/ "
              f"(raw_parameters_per_bath.csv, raw_round_channels_per_bath.csv, raw_signals_and_spectra_<rung>.npz)")
    return df1, df3

def plot(frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, surface = "#0b0b0b", "#52514e", "#ffffff"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface})
    names = [r[0] for r in LADDER]
    ticks = [f"{r[0]}\n{'exact' if r[1] is None else r[1]} shots\nreadout {100 * r[2]:g}%\nprep {r[3]:g}\ndrift {100 * r[4]:g}%"
             for r in LADDER]
    x = np.arange(len(names))

    # Figure L1: logical error rate on the ladder
    tc, dmax = T_CLOCK_LIST[0], max(m.DISTANCES)
    style = {ARM_CLOCK: ("#eb6834", "s", "--", "Clock, frame calibrated"),
             ARM_ML: ("#8a5cd0", "^", ":", "Structured bath, ML inferred"),
             ARM_FIT: ("#2a78d6", "o", "-", "Structured bath, ML then fit"),
             ARM_EXACT: ("#1baf7a", "D", "-.", "Structured bath, exact at calibration (bound)")}
    fig, axes = plt.subplots(len(frames), 2, figsize=(11, 4.2 * len(frames)), squeeze=False, sharex=True)
    for r, (gamma_sys, (_, df)) in enumerate(frames.items()):
        d = df[(df["T_clock"] == tc) & (df["Distance"] == dmax)]
        for c, (basis, title) in enumerate((("Z", "Bit flip memory"), ("X", "Phase flip memory"))):
            ax, sub = axes[r, c], d[d["Memory_basis"] == basis]
            floor = 0.5 / sub["Shots"].iloc[0]
            for arm in ARMS:
                col, mk, ls, lab = style[arm]
                y = [max(float(sub[(sub["Arm"] == arm) & (sub["Rung"] == n)]["LER"].iloc[0]), floor) for n in names]
                ax.plot(x, y, ls, color=col, marker=mk, ms=6, lw=2, markeredgecolor=surface, label=lab)
            ax.set_yscale("log")
            ax.set_ylabel("Logical error rate")
            ax.set_title(f"{title}, rail dephasing {'off' if gamma_sys == 0 else 'on'}", fontsize=10.5)
            ax.grid(True, which="major", color="#e4e3df", lw=0.8)
            ax.set_axisbelow(True)
            ax.set_xticks(x)
            ax.set_xticklabels(ticks, fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"Distance {dmax}, clock period {tc:g}, balanced check: logical error rate on the noise ladder", fontsize=12)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig("figure_L1_ladder_logical_error.png", dpi=200)
    print("Saved figure_L1_ladder_logical_error.png")

    # Figure L2: parameter recovery on the ladder
    methods = (("ML, trained on exact signals", "#9a9892", "s", "--"), ("ML, trained on noisy signals", "#8a5cd0", "^", ":"),
               (LABEL_FIT, "#2a78d6", "o", "-"))
    fig, axes = plt.subplots(len(frames), len(m.PARAM_NAMES), figsize=(3.0 * len(m.PARAM_NAMES), 3.4 * len(frames)),
                             squeeze=False, sharex=True, sharey=True)
    for r, (gamma_sys, (df, _)) in enumerate(frames.items()):
        for c, pname in enumerate(m.PARAM_NAMES):
            ax, sub = axes[r, c], df[df["Parameter"] == pname]
            for label, col, mk, ls in methods:
                y = [max(float(sub[(sub["Method"] == label) & (sub["Rung"] == n)]["R2"].iloc[0]), -0.5) for n in names]
                ax.plot(x, y, ls, color=col, marker=mk, ms=5, lw=1.8, markeredgecolor=surface, label=label)
            ax.axhline(0.0, color=muted, lw=0.6)
            ax.set_ylim(-0.55, 1.05)
            ax.set_title(f"{pname}, dephasing {'off' if gamma_sys == 0 else 'on'}", fontsize=9.5)
            ax.grid(True, which="major", color="#e4e3df", lw=0.8)
            ax.set_axisbelow(True)
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace("rung ", "") for n in names], fontsize=8)
            if c == 0:
                ax.set_ylabel("R2 on unseen baths (floor of the axis: -0.5)")
            if r == len(frames) - 1:
                ax.set_xlabel("Rung of the noise ladder")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Bath parameters recovered on the noise ladder", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("figure_L2_ladder_parameters.png", dpi=200)
    print("Saved figure_L2_ladder_parameters.png")

if __name__ == "__main__":
    plot({g: run_case(g) for g in GAMMA_SYS_LIST})
