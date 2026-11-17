#!/usr/bin/env python
"""
free_link_check_placement.py

Part 2, the test for the controller: the erasure check placed from the map
with every link free, tested exactly as in Part 1.

Everything that defines the test is taken from dual_rail_structured_bath.py
without change: the operators of the 7 qubit model and its sector, the
channel of one round (round_channel), the balanced check rule (window
T_clock to T_clock * (1 + WINDOW_WIDE), cost = unflagged Pauli error +
ERASURE_COST * erasure), the correcting rotation, the surface code with
heralded erasure and its decoder (logical_errors), the clock periods, the
distances, the shots and the seed.

What is new: the Hamiltonian has one coupling per link (the eleven
parameters of free_link_training.py), and the bath of every data qubit is
read from raw_parameters_per_bath.csv of a free link training run (true
values, regression, ML then fit, for the balanced input and for the three
inputs). No training is repeated here.

Arms, at each clock period:
  Clock, frame calibrated                  check at T_clock, true rotation,
                                           decoder knows the true rates
  Structured bath, ML then fit, three inputs      the full map
  Structured bath, ML then fit, balanced input    the full map, one input
  Structured bath, ML alone, three inputs         regression without the fit
  Full map without the layer 2 bonds       fit (three inputs) with the two
                                           layer 2 bonds set to mid range
  Vertical links only                      fit (three inputs) with all three
                                           horizontal bonds set to mid range
  Structured bath, exact (bound)           the true bath

Usage:
  python free_link_check_placement.py --results results_free_links_trapped_gamma_sys_0.0 --gamma_sys 0.0

AI (LLM) assistance (Claude, Anthropic) was used in preparing code and text.
The framework, calculations and analysis are those of R. Sakidja, who takes
full responsibility for the scientific content.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.linalg import expm, polar

import dual_rail_structured_bath as m
import free_link_training as f

# ---- sector operators of the free link model, from the operators of m ----
_H0 = m._Hs[0]
_HB = [m._proj(m._heis(a, b)) for _, a, b, _, _ in f.BONDS]
N_BOND = len(f.BONDS)
HORIZONTAL_L2 = [f.NAMES.index("J_L2_34"), f.NAMES.index("J_L2_45")]
HORIZONTAL_ALL = [f.NAMES.index("J_L1_12")] + HORIZONTAL_L2
MID = 0.5 * (np.array([b[4][0] for b in f.BONDS]) + np.array([b[4][1] for b in f.BONDS]))


def sector_model(p, gamma_sys):
    """As m.sector_model, with one coupling per link. p = the eleven
    parameters in the order of free_link_training.NAMES."""
    H = _H0 + sum(p[k] * _HB[k] for k in range(N_BOND))
    g1, g2 = p[N_BOND], p[N_BOND + 1]
    cs = [np.sqrt(gamma_sys) * d for d in m._DEPH]
    cs += [np.sqrt((g1 if k < 2 else g2) * (1 + m.N_TH)) * m._DECAY[k] for k in range(5)]
    return H, cs


def liouvillian(p, gamma_sys):
    """As m.liouvillian."""
    H, cs = sector_model(p, gamma_sys)
    L = -1j * (np.kron(H, m._EYE) - np.kron(m._EYE, H.T))
    for c in cs:
        cdc = c.conj().T @ c
        L += np.kron(c, c.conj()) - 0.5 * np.kron(cdc, m._EYE) - 0.5 * np.kron(m._EYE, cdc.T)
    return L


def correction(p, gamma_sys, tau):
    """As m.correction."""
    H, cs = sector_model(p, gamma_sys)
    H_eff = H - 0.5j * sum(c.conj().T @ c for c in cs)
    return polar(expm(-1j * H_eff * tau)[m.I_L:m.I_R + 1, m.I_L:m.I_R + 1])[0]


def best_check_time(p, gamma_sys, t_clock):
    """The balanced rule of m.best_check_time."""
    L = liouvillian(p, gamma_sys)
    taus = np.linspace(t_clock, t_clock * (1 + m.WINDOW_WIDE), m.N_TAU_WIDE)
    cost = []
    for t in taus:
        c = m.round_channel(L, t, correction(p, gamma_sys, t))
        cost.append(c["p_x"] + c["p_y"] + c["p_z"] + m.ERASURE_COST * c["p_erase"])
    return float(taus[int(np.argmin(cost))])


ARM_CLOCK = "Clock, frame calibrated"
ARM_FIT3 = "Structured bath, ML then fit, three inputs"
ARM_FIT1 = "Structured bath, ML then fit, balanced input"
ARM_ML3 = "Structured bath, ML alone, three inputs"
ARM_NO_L2 = "Full map without the layer 2 bonds"
ARM_VERT = "Vertical links only"
ARM_EXACT = "Structured bath, exact (bound)"
ARMS = (ARM_CLOCK, ARM_FIT3, ARM_FIT1, ARM_ML3, ARM_NO_L2, ARM_VERT, ARM_EXACT)


def arms_job(models, gamma_sys, t_clock):
    """One data qubit at one clock period. models: dict arm -> eleven
    parameters; models[ARM_EXACT] is the true bath. Every arm returns the
    channel that really acts and the channel its decoder believes."""
    p_true = models[ARM_EXACT]
    L_true = liouvillian(p_true, gamma_sys)
    clock = dict(m.round_channel(L_true, t_clock, correction(p_true, gamma_sys, t_clock)), tau=t_clock)
    out = {ARM_CLOCK: {"true": clock, "prior": clock}}
    for arm, p in models.items():
        t = best_check_time(p, gamma_sys, t_clock)
        U = correction(p, gamma_sys, t)
        out[arm] = {"true": dict(m.round_channel(L_true, t, U), tau=t),
                    "prior": dict(m.round_channel(liouvillian(p, gamma_sys), t, U), tau=t)}
    return out


def consistency_checks(gamma_sys):
    """1. With the links tied as in Part 1, the free link generator equals
    m.liouvillian. 2. It equals the generator of free_link_training.py up to
    the thermal absorption that m drops (rate gamma * n_th)."""
    J_s, J_1, J_i, J_2, g1, g2 = [0.5 * (lo + hi) for lo, hi in m.PARAM_RANGES]
    a = m.SOURCE_ASYMMETRY
    tied = np.array([(1 - a) * J_s, (1 + a) * J_s, J_1, J_i, J_i, J_i, J_i, J_2, J_2, g1, g2])
    d1 = np.abs(liouvillian(tied, gamma_sys) - m.liouvillian((J_s, J_1, J_i, J_2, g1, g2, gamma_sys))).max()
    p = f.draw(np.random.default_rng(3), 1)[0]
    d2 = np.abs(liouvillian(p, gamma_sys) - f.liouvillian(p, gamma_sys)).max()
    return d1, d2


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder of a free link training run")
    ap.add_argument("--gamma_sys", type=float, required=True)
    ap.add_argument("--shots", type=int, default=m.SHOTS)
    ap.add_argument("--t_clock", type=float, nargs="+", default=list(m.T_CLOCK_LIST))
    ap.add_argument("--distances", type=int, nargs="+", default=list(m.DISTANCES))
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    outdir = a.outdir or a.results.rstrip("/") + "_check_placement"
    os.makedirs(outdir, exist_ok=True)
    assert m.ATTACH == "split", "run with the split attachment (SQB_ATTACH unset or split)"
    d1, d2 = consistency_checks(a.gamma_sys)
    print(f"Check placement with every link free. gamma_sys = {a.gamma_sys}, baths from {a.results}")
    print(f"Consistency: free link generator against Part 1 with tied links: {d1:.2e}; "
          f"against free_link_training.py: {d2:.2e}")
    assert d1 < 1e-12 and d2 < 1e-6

    raw = pd.read_csv(os.path.join(a.results, "raw_parameters_per_bath.csv"))
    get = lambda arm, kind: raw[raw.Arm == arm].sort_values("Bath")[[f"{kind}_{n}" for n in f.NAMES]].to_numpy()
    P_true, P_fit3, P_fit1 = get("three_inputs", "true"), get("three_inputs", "fit"), get("balanced", "fit")
    P_ml3 = get("three_inputs", "regression")
    assert np.allclose(P_true, get("balanced", "true"))
    n_bath = len(P_true)
    P_no_l2, P_vert = P_fit3.copy(), P_fit3.copy()
    P_no_l2[:, HORIZONTAL_L2] = MID[HORIZONTAL_L2]
    P_vert[:, HORIZONTAL_ALL] = MID[HORIZONTAL_ALL]
    models = [{ARM_FIT3: P_fit3[k], ARM_FIT1: P_fit1[k], ARM_ML3: P_ml3[k], ARM_NO_L2: P_no_l2[k],
               ARM_VERT: P_vert[k], ARM_EXACT: P_true[k]} for k in range(n_bath)]
    outcome = raw[raw.Arm == "three_inputs"].sort_values("Bath")["outcome"].tolist() if "outcome" in raw else None
    print(f"{n_bath} unseen baths; clock periods {a.t_clock}; distances {a.distances}; {a.shots} shots per point")

    # ---- one round per bath ----
    t0 = time.time()
    jobs = [(k, tc) for tc in a.t_clock for k in range(n_bath)]
    res = Parallel(n_jobs=-1)(delayed(arms_job)(models[k], a.gamma_sys, tc) for k, tc in jobs)
    true_ch = {(tc, arm): [] for tc in a.t_clock for arm in ARMS}
    prior_ch = {(tc, arm): [] for tc in a.t_clock for arm in ARMS}
    rows_raw = []
    for (k, tc), r in zip(jobs, res):
        for arm in ARMS:
            true_ch[(tc, arm)].append(r[arm]["true"])
            prior_ch[(tc, arm)].append(r[arm]["prior"])
            c = r[arm]["true"]
            rows_raw.append({"Bath": k, "Fit_outcome_three_inputs": outcome[k] if outcome else "", "T_clock": tc,
                             "Arm": arm, "Check_time": c["tau"], "p_erase": c["p_erase"], "p_x": c["p_x"],
                             "p_y": c["p_y"], "p_z": c["p_z"], "Unflagged_process_fidelity": c["chi_I"]})
    pd.DataFrame(rows_raw).to_csv(os.path.join(outdir, "raw_round_channels_per_bath.csv"), index=False)
    rows1 = []
    for (tc, arm), chans in true_ch.items():
        rows1.append({"T_clock": tc, "Arm": arm, "Mean_check_time": np.mean([c["tau"] for c in chans]),
                      "p_erase": np.mean([c["p_erase"] for c in chans]),
                      "p_pauli_unflagged": np.mean([c["p_x"] + c["p_y"] + c["p_z"] for c in chans])})
    df1 = pd.DataFrame(rows1)
    print(f"\nTable P1. One round per data qubit with every link free, mean over {n_bath} unseen baths "
          f"({time.time() - t0:.0f} s)")
    print(df1.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    df1.to_csv(os.path.join(outdir, "tableP1_round_free_links.csv"), index=False)

    # ---- surface code ----
    t0 = time.time()
    pts = [(tc, arm, b, d) for tc in a.t_clock for arm in ARMS for b in m.BASES for d in a.distances]
    errs = Parallel(n_jobs=-1)(delayed(m.logical_errors)(d, b, true_ch[(tc, arm)], prior_ch[(tc, arm)], a.shots,
                                                         m.SEED + i) for i, (tc, arm, b, d) in enumerate(pts))
    shots = (a.shots // m.N_DRAWS) * m.N_DRAWS
    rows2 = []
    for (tc, arm, b, d), e in zip(pts, errs):
        tau = np.mean([c["tau"] for c in true_ch[(tc, arm)]])
        ler = e / shots
        rows2.append({"T_clock": tc, "Arm": arm, "Memory_basis": b.upper(), "Distance": d, "Logical_errors": e,
                      "Shots": shots, "LER": ler,
                      "LER_per_unit_time": 1 - (1 - min(ler, 0.5)) ** (1.0 / (d * tau))})
    df2 = pd.DataFrame(rows2)
    print(f"\nTable P2. Surface code logical error rate with every link free, {shots} shots per point over "
          f"{m.N_DRAWS} bath placements ({time.time() - t0:.0f} s)")
    print(df2.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    df2.to_csv(os.path.join(outdir, "tableP2_surface_code_free_links.csv"), index=False)

    dmax = max(a.distances)
    piv = df2[df2.Distance == dmax].pivot_table(index=["T_clock", "Memory_basis"], columns="Arm", values="LER")[list(ARMS)]
    print(f"\nTable P3. Logical error rate at distance {dmax}, arms side by side")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(piv.rename(columns={ARM_CLOCK: "Clock", ARM_FIT3: "Fit, 3 inputs", ARM_FIT1: "Fit, balanced",
                                  ARM_ML3: "ML alone", ARM_NO_L2: "No L2 bonds", ARM_VERT: "Vertical only",
                                  ARM_EXACT: "Exact"}).to_string(float_format=lambda v: f"{v:.5f}"))
    piv.to_csv(os.path.join(outdir, "tableP3_distance_%d_side_by_side.csv" % dmax))
    print(f"\nSaved tables P1 to P3 and raw_round_channels_per_bath.csv in {outdir}/")


if __name__ == "__main__":
    main()
