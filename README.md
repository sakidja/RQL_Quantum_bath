# Erasure Check Timing from an Inferred Structured Bath

Code, result tables and figures for the paper

**R. Sakidja, "Erasure Check Timing from an Inferred Structured Bath: Dual Rail Qubits with Near Field Memory"**

The near field of a dual rail superconducting qubit is kept as a finite quantum bath (two rails, five bath qubits, one excitation). The bath is inferred from the calibration record of the qubit itself, before any erasure check is made: Fourier features, principal components and gradient boosted regression give a first estimate, and a least squares fit of the bath model to the record refines it. The inferred bath predicts when the excitation is in the near field and when it has returned to the rails. The erasure check, a correcting rotation and the decoder weights are set from that prediction and tested in a rotated surface code.

Relation to Reciprocal Quantum Logic (RQL): the method is complementary to the RQL control of dual rail qubits, not an alternative to it. The controller supplies the clock and the flux pulses, and the map of the near field supplies the times at which they act. No hardware is added and no check is added.

This repository holds the scripts, the result tables of the runs, the raw per bath tables behind them, and the figures. Every table and figure of the paper is listed below with the script and the file that produce it.

## Layout

| Folder | Content |
|---|---|
| `part1/` | Part 1 of the paper: the layered bath with tied links, both ways of attaching the rails, and the noise ladder |
| `part2/` | Part 2 of the paper: every link of the near field free |
| `figures/` | The figures of the paper, as drawn by the scripts |

Each result folder sits next to the script that writes it and keeps the name the script gives it, so every script runs here without any change of path.

## Part 1

| File | Purpose |
|---|---|
| `dual_rail_structured_bath.py` | Model, calibration record, regression, fit, channel of one round, check placement, surface code. Writes Tables 1 to 3 of a run and the raw tables |
| `noise_ladder.py` | The same pipeline with a noisy calibration record (shots, readout error, preparation error, drift). Writes Tables L1 to L3 and the raw tables |
| `plot_results_one_case.py` | Figures 4, 5, 6 and 9 from the tables of a run. Computes nothing new |
| `plot_ladder_final.py` | Figures 7 and 8 from the tables of the noise ladder. Computes nothing new |
| `plot_bath_schematic.py` | Figure 1, the schematic of the bath for both attachments. Draws only |
| `run_part1.sh`, `run_ladder.sh` | Batch scripts (SLURM). Put your own account in the header |

Runs:

```
SQB_ATTACH=one   python dual_rail_structured_bath.py     # first attachment
SQB_ATTACH=split python dual_rail_structured_bath.py     # split attachment
SQB_ATTACH=split python noise_ladder.py                  # noise ladder
SQB_RESULTS=results_v3_final8_one_gamma_sys_0.0   SQB_ATTACH=one   python plot_results_one_case.py
SQB_RESULTS=results_v3_final8_split_gamma_sys_0.0 SQB_ATTACH=split python plot_results_one_case.py
python plot_ladder_final.py
python plot_bath_schematic.py
```

Settings of the model in Part 1, all in the head of `dual_rail_structured_bath.py`:

| Quantity | Value |
|---|---|
| Rail to layer 1, J_sb | 0.1 to 0.5 |
| Inside layer 1, J_L1 | 0.5 to 0.99 |
| Layer 1 to layer 2, J_L12 | 0.1 to 0.5, four links at 0.8, 1.1, 0.9, 1.2 times J_L12 |
| Inside layer 2, J_L2 | 0.5 to 0.99, two bonds at 0.5 and 1.5 times J_L2 |
| Far field rates of layer 1 and layer 2 | 0.0001 to 0.002 |
| Calibration inputs | balanced state, excitation on rail L, excitation on rail R |
| Record | 5000 points to t = 1500, 9 rows |
| Training baths, unseen baths | 4000, 49 |
| Regression | 30 principal components of the log spectrum, plus decay envelope and line features |
| Fit | bounded least squares, staged window, up to 6 starting points |
| Clock periods | 1, 2, 3, 4, 6, 12, 25, 50 |
| Surface code | rotated, distances 3, 5, 7, 20000 shots per point |

In the noise ladder the fit has three more free numbers next to the six bath parameters: the contrast of the signal (readout error), and the tilt and the phase of the preparation. It reads the whole record and scans the two horizontal bonds before it refines. The rungs are exact, 10000 shots per point, and two rungs of 1000 shots per point.

## Part 2

| File | Purpose |
|---|---|
| `free_link_training.py` | Eleven free parameters (nine couplings, two rates), bounds on every coupling, regression, fit from up to 16 starting points with a staged start. Writes Tables F0 to F3 and the raw tables |
| `free_link_check_placement.py` | Check placement and surface code from the baths of a training run. Writes Tables P1 to P3 and the raw table. No training is repeated |
| `plot_free_links.py`, `plot_final_figures.py` | Figures 2, 10, 11 and 12. Compute nothing new |
| `dual_rail_structured_bath.py` | Library copy used by the two Part 2 scripts for the operators, the channel of one round and the surface code |
| `run_part2_training.sh`, `run_part2_check.sh` | Batch scripts (SLURM). Put your own account in the header |

Runs:

```
python free_link_training.py --gamma_sys 0.0 --n_train 4000 --n_test 49 --n_pc 30 --workers 120 \
       --full_window --n_starts 16 --vertical_range 0.1 0.5 --outdir results_free_links_final16h_gamma_sys_0.0
python free_link_check_placement.py --gamma_sys 0.0 --results results_free_links_final16h_gamma_sys_0.0 \
       --t_clock 1 2 3 4 6 12 25 50
python plot_final_figures.py
```

## Tables of the paper

| Table | Script | File |
|---|---|---|
| 1 | `part1/dual_rail_structured_bath.py` | settings in the head of the script |
| 2 | not in this repository | test on the unchanged model of the earlier work, Google Colab run of the author |
| 3 | `part1/dual_rail_structured_bath.py` | `part1/results_v3_final8_one_gamma_sys_0.0/table1_inference.csv` |
| 4 | same | `.../table2_round_channels.csv` |
| 5, 6 | same | `.../table3_surface_code.csv` |
| 7 | `part1/noise_ladder.py` | settings in the head of the script |
| 8 | `part1/noise_ladder.py` | `part1/results_v3_ladder_fair_split_gamma_sys_0.0/tableL1_inference_on_ladder.csv` |
| 9 | same | `.../tableL3_surface_code_on_ladder.csv` (one round: `tableL2_round_on_ladder.csv`) |
| 10 | `part1/dual_rail_structured_bath.py` | `part1/results_v3_final8_split_gamma_sys_0.0/table1_inference.csv` |
| 11 | same | `.../table3_surface_code.csv` |
| 12 | same | column `LER_per_unit_time` of `table3_surface_code.csv`, both attachments |
| 13 | `part2/free_link_training.py` | settings in the head of the script |
| 14 | `part2/free_link_training.py` | `part2/results_free_links_final16h_gamma_sys_0.0/tableF0_preflight_occupancy.csv` |
| 15 | same | `.../tableF1_preflight_identifiability.csv` |
| 16 | same | `.../tableF3_fit_outcomes.csv` and `raw_fit_starts_per_bath.csv` |
| 17 | same | `.../tableF2_r2_free_links.csv` |
| 18 | `part2/free_link_check_placement.py` | `part2/results_free_links_final16h_gamma_sys_0.0_check_placement/tableP1_round_free_links.csv` |
| 19 | same | `.../tableP2_surface_code_free_links.csv` |

## Figures of the paper

| Figure | Script | File in `figures/` |
|---|---|---|
| 1 | `part1/plot_bath_schematic.py` | `figure_bath_schematic.png` |
| 2 | `part2/plot_final_figures.py` | `figure_1_signal_and_fingerprint.png` |
| 3 | not in this repository | Google Colab run of the author on the model of the earlier work |
| 4 | `part1/plot_results_one_case.py` | `figure_round_channels.png` |
| 5 | same | `figure_logical_error_vs_clock.png` |
| 6 | same | `figure_logical_error_vs_distance.png` |
| 7 | `part1/plot_ladder_final.py` | `figure_6_ladder_parameters.png` |
| 8 | same | `figure_7_ladder_logical_error.png` |
| 9 | `part1/plot_results_one_case.py` | `figure_logical_error_vs_clock_split.png` |
| 10 | `part2/plot_final_figures.py` | `figure_9_bound_on_every_coupling.png` |
| 11 | same | `figure_10_round_free_links.png` |
| 12 | same | `figure_11_logical_error_free_links.png` |

The first column is the number of the figure in the paper. Some file names carry a number from before the schematic of the bath was added as Figure 1, so the number in a file name is one lower than the number of the figure.

## Raw data

Every run saves the raw data behind its tables next to them:

* `raw_parameters_per_bath.csv`: for every bath, the true parameters, the regression estimate and the fitted parameters.
* `raw_round_channels_per_bath.csv`: for every bath, arm and clock period, the check time and the channel of one round.
* `raw_fit_starts_per_bath.csv` (Part 2): every starting point of the fit, its kind, its final cost and its outcome.

The signals and spectra of every bath (`raw_signals_and_spectra*.npz`, `raw_signals_test.npz`) are written by the same runs. They are several hundred megabytes and are not in this repository; running the scripts writes them again. Figure 2 is drawn from `part1/results_v3_final8_one_gamma_sys_0.0/raw_signals_and_spectra.npz`.

## Requirements

Python 3 with the packages of `requirements.txt`. The runs of the paper used one CPU node; the Part 1 runs take about ten minutes each with the training set saved, the noise ladder about two hours, the Part 2 training about ninety minutes.

## What the paper does not cover

* After a flagged erasure the model empties the near field. In hardware an excitation left there stays until it decays or returns.
* The noise ladder is run on Part 1 only.
* Dynamical decoupling timed from the map is named as the next step and is not simulated here.

## Acknowledgement

This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility, as part of the QIS@Perlmutter program. AI (LLM) assistance (Claude, Anthropic) was used in preparing code and text. The framework, calculations and analysis are those of R. Sakidja, who takes full responsibility for the scientific content.
