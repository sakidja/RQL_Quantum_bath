# Erasure Check Timing from an Inferred Structured Bath: Dual Rail Qubits with Near Field Memory

Code, result tables and figures for the paper of the same title by Ridwan Sakidja (Missouri State University).

The method of the structured quantum bath (R. Sakidja, arXiv:2508.17514) is applied to a dual rail erasure qubit. The bath is a layered network of two level systems in the near field of the qubit: the qubit couples to layer 1 (two nodes), and layer 1 couples to layer 2 (three nodes). The six bath parameters J_sb, J_L1, J_L12, J_L2, γ_L1 and γ_L2 are inferred from the calibration signal of the qubit itself (Fourier spectrum, principal components, XGBoost regression, then a fit of the signal). The inferred bath predicts when the excitation is exchanged with the near field, and the erasure check and the correction are placed at the time the map gives, in place of a fixed clock. The logical error rate of a rotated surface code with heralded erasures (distance 3, 5 and 7) is then compared between the clocked check and the check timed from the inferred bath. Part 2 of the paper (Section 4) frees every link of the bath: nine couplings and two rates are drawn independently and inferred from the calibration signals of one and of three input states, and the check placement and the surface code are run on that model.

## Contents

| File | Purpose |
|---|---|
| `dual_rail_structured_bath.py` | Main run: model, calibration signals, regression, fit, one round channel per data qubit, surface code |
| `noise_ladder.py` | The same chain on calibration signals with shot noise, readout error, preparation error and drift (five rungs) |
| `training_size_study.py` | Bath parameters recovered with 400 and with 2000 training baths |
| `table15_per_unit_time.py` | Lowest logical error rate per unit time, clocked check against timed check |
| `plot_results.py`, `plot_ladder.py`, `plot_fft.py` | Figures from the saved tables |
| `run_perlmutter.sh`, `run_perlmutter_first_attachment.sh`, `run_noise_ladder.sh`, `run_training_size.sh` | Slurm scripts (NERSC Perlmutter, 32 cores) |
| `free_link_training.py` | Part 2: model with every link free, preflight (bounds and layer populations), regression, fit from several starting points |
| `free_link_check_placement.py` | Part 2: check placement, round channel and surface code on the model with every link free |
| `free_link_analysis.py` | Part 2: numbers of the text read from the raw files (rank correlations, counts of baths recovered) |
| `plot_free_links.py` | Figures 9 to 12 from the saved tables |
| `run_free_link_training.sh`, `run_free_link_control.sh`, `run_free_link_check_placement.sh`, `run_free_link_check_oldrates.sh` | Slurm scripts of Part 2 |
| `noise_ladder_dsp.py`, `table_dsp_comparison.py`, `run_noise_ladder_dsp.sh`, `run_noise_ladder_fixed_floor.sh` | Noise ladder with the fixed floor and the multitaper spectrum, and the comparison of the four spectra |
| `colab/structured_bath_prediction_test.ipynb` | Google Colab notebook: model of arXiv:2508.17514, prediction of unseen input states on unseen baths |
| `results_*/` | Result tables of the runs behind the paper (CSV) |
| `preflight_300_*/` | Preflight runs of Part 2 (300 baths each) |
| `logs/` | Printed output of the runs of Part 2 and of the noise ladder with the other spectra |
| `figures/` | Figures 1 to 12 of the paper |

## Two attachments of the bath to the qubit

The attachment is selected with the environment variable `SQB_ATTACH`.

| Value | Geometry | Paper |
|---|---|---|
| `one` | Both nodes of layer 1 couple to rail R with strength J_sb | Tables 3 to 6, Figures 3 to 5 |
| `split` (default) | Rail L couples to node Q1 with 0.8 J_sb, rail R couples to node Q2 with 1.2 J_sb | Tables 7 to 14, Figures 6 to 8 |

## Which run produces which table and figure

| Paper | Script | Result file |
|---|---|---|
| Table 1 (parameters of the model) | settings block of `dual_rail_structured_bath.py` | |
| Table 2, Figure 2 | `colab/structured_bath_prediction_test.ipynb` | printed by the notebook |
| Figure 1 | `SQB_ATTACH=one python plot_fft.py` | `figures/figure1_...png` |
| Table 3 | `dual_rail_structured_bath.py`, `SQB_ATTACH=one` | `results_v2_gamma_sys_*/table1_inference.csv` |
| Table 4, Figure 3 | same run | `results_v2_gamma_sys_*/table2_round_channels.csv` |
| Tables 5 and 6, Figures 4 and 5 | same run | `results_v2_gamma_sys_*/table3_surface_code.csv` |
| Table 7 (definition of the ladder) | `LADDER` in `noise_ladder.py` | |
| Tables 8 and 10, Figure 6 | `noise_ladder.py` | `results_v3_ladder_split_gamma_sys_*/tableL1_inference_on_ladder.csv` |
| Tables 9 and 11, Figure 7 | `noise_ladder.py` | `results_v3_ladder_split_gamma_sys_*/tableL3_surface_code_on_ladder.csv` |
| Table 12 | `dual_rail_structured_bath.py`, `SQB_ATTACH=split` | `results_v3_split_gamma_sys_*/table1_inference.csv` |
| Table 13, Figure 8 | same run, `plot_results.py` | `results_v3_split_gamma_sys_*/table3_surface_code.csv` |
| Table 14 | `training_size_study.py` | `results_v3_training_size_split_gamma_sys_*/tableS1_inference_vs_training_size.csv` |
| Table 15 | `table15_per_unit_time.py` | `table15_per_unit_time.csv` (written by the script from the two surface code tables) |

The folder name ends with the direct rail dephasing of the run: `gamma_sys_0.0` or `gamma_sys_0.005`.

## Runs behind the paper

All numbers of the paper come from the author's runs on NERSC Perlmutter and, for Table 2 and Figure 2, from the author's Google Colab run.

| Run | Folders |
|---|---|
| Main run, first attachment | `results_v2_gamma_sys_*` |
| Main run, split attachment | `results_v3_split_gamma_sys_*` |
| Noise ladder, split attachment | `results_v3_ladder_split_gamma_sys_*` |
| Training size study, split attachment | `results_v3_training_size_split_gamma_sys_*` |

The main run with the first attachment used the earlier version of the main script, which printed the tables and did not store the data of each bath. The script in this repository stores them. A rerun of the first attachment with `sbatch run_perlmutter_first_attachment.sh` writes `results_v3_one_gamma_sys_*` with the raw data; `table15_per_unit_time.py` and `plot_results.py` read that folder when it is present.

## Raw data

With `SAVE_RAW = True` (the default) every run writes, next to its tables, the data behind them:

| File | Content |
|---|---|
| `raw_parameters_per_bath.csv` | True, regressed and fitted parameters of each of the 49 unseen baths |
| `raw_round_channels_per_bath.csv` | Erasure and Pauli error probabilities of the round for each unseen bath, each clock period and each arm |
| `raw_signals_and_spectra*.npz` | Calibration signals and Fourier features of every bath (training and unseen) |
| `raw_bath_parameters.csv`, `raw_parameters_per_unseen_bath.csv` | Training size study |

The CSV tables in this repository are the summary tables. The raw files of the three runs with the split attachment are on Perlmutter; the CSV files are small and can be added to the result folders, and the `npz` files are large and are better placed in a data archive (Zenodo, for example) that the paper cites.

## How to run

One time setup on Perlmutter:

```bash
module load python
python -m venv $SCRATCH/sqb-env
source $SCRATCH/sqb-env/bin/activate
pip install qutip stim pymatching xgboost scikit-learn joblib pandas scipy matplotlib
```

The Slurm scripts carry the line `#SBATCH -A YOUR_PROJECT` or `#SBATCH --account=YOUR_PROJECT`; put your own project there.

Runs:

```bash
sbatch run_perlmutter.sh                    # main run, split attachment
sbatch run_perlmutter_first_attachment.sh   # main run, first attachment
sbatch run_noise_ladder.sh                  # noise ladder
sbatch run_training_size.sh                 # training size study
sacct -j <jobid> --format=JobID,State,Elapsed
```

After the runs:

```bash
python table15_per_unit_time.py
python plot_results.py                      # SQB_ATTACH=one python plot_results.py for the first attachment
python plot_ladder.py                       # Figures 6 and 7 from the ladder tables
python plot_fft.py
```

On a workstation the same scripts run with `python dual_rail_structured_bath.py`; the simulated dataset is cached in an `npz` file, so a second run skips the simulations.

## Settings of the runs

400 training baths and 49 unseen baths (seed in the script); parameter ranges J_sb 0.1 to 0.5, J_L1 0.5 to 0.99, J_L12 0.01 to 0.1, J_L2 0.5 to 0.99, γ_L1 and γ_L2 0.001 to 0.01; direct rail dephasing γ_sys 0 and 0.005; βω = 12; clock periods 1, 2, 3, 4 and 6; surface code distance 3, 5 and 7; 20000 shots per point over 5 bath placements.

## Statement on AI assistance

AI (LLM) assistance (Claude, Anthropic) was used in preparing code and text. The framework, the calculations and the analysis are the author's, and the author takes full responsibility for the content.

## Citation

R. Sakidja, "Erasure Check Timing from an Inferred Structured Bath: Dual Rail Qubits with Near Field Memory" (2026).

R. Sakidja, arXiv:2508.17514.

---

## Part 2: the near field with every link free (Section 4)

Model: `free_link_training.py`. Nine couplings and two rates, each drawn independently (Table 16). Two
calibration arms (balanced input; balanced, rail L and rail R inputs), 4000 training baths, 49 unseen baths,
exact signals, fit from up to six starting points. The check placement and the surface code are those of
`dual_rail_structured_bath.py`, called by `free_link_check_placement.py`. `free_link_analysis.py` prints the
numbers of the text that are read from the raw files.

| Paper | Script | Run | File |
| --- | --- | --- | --- |
| Table 16 | `free_link_training.py` | BONDS and RATES blocks of the script | |
| Table 17 | `free_link_training.py` | `--preflight_only --skip_validation --n_preflight 300 --outdir preflight_300_trapped` and the same with `--gamma_range 0.001 0.01 --outdir preflight_300_old_rates` | `tableF0_preflight_occupancy.csv` in both folders |
| Table 18 | same | same two runs | `tableF1_preflight_identifiability.csv` in both folders |
| Text under Table 18 (rank correlations, four boxes) | `free_link_analysis.py` | `python free_link_analysis.py tilt preflight_300_gamma_sys_0.0`; that folder is the run `--preflight_only --skip_validation --n_preflight 300 --gamma_range 0.001 0.01 --rate_draw uniform --outdir preflight_300_gamma_sys_0.0` | `raw_preflight_per_bath.csv` |
| Table 19, rows with 1 starting point | `free_link_training.py`, `free_link_analysis.py` | `--gamma_range 0.001 0.01 --rate_draw uniform --n_starts 1 --full_window`, both values of gamma_sys; counts by `python free_link_analysis.py outcomes results_free_links_gamma_sys_0.0 results_free_links_gamma_sys_0.005` | `raw_parameters_per_bath.csv` |
| Table 19, rates of Part 1, up to 6 starting points | `free_link_training.py` | `run_free_link_control.sh` (`--gamma_range 0.001 0.01 --n_starts 6 --full_window`) | `results_free_links_oldrates_multistart_gamma_sys_*/tableF3_fit_outcomes.csv` |
| Table 19, trapped rates | `free_link_training.py` | `run_free_link_training.sh` | `results_free_links_trapped_gamma_sys_*/tableF3_fit_outcomes.csv` |
| Tables 20 and 21 | same | same run | `results_free_links_trapped_gamma_sys_*/tableF2_r2_free_links.csv` |
| Baths not recovered | `free_link_analysis.py` | `python free_link_analysis.py outcomes results_free_links_trapped_gamma_sys_0.0 results_free_links_trapped_gamma_sys_0.005` | `raw_parameters_per_bath.csv`, `raw_fit_starts_per_bath.csv` |
| Tables 22 and 23 | `free_link_check_placement.py` | `run_free_link_check_placement.sh` | `results_free_links_trapped_gamma_sys_0.0_check_placement/tableP1_round_free_links.csv`, `tableP3_distance_7_side_by_side.csv` |
| Table 24 | same | same run | `results_free_links_trapped_gamma_sys_0.005_check_placement/tableP3_distance_7_side_by_side.csv` |
| Figures 9 to 12 | `plot_free_links.py` | `python plot_free_links.py`; reads the tables of the rows above and of Table 25 and computes nothing new; copies of the figures are in `figures/` | `figure_9_bound_on_every_coupling.png`, `figure_10_round_free_links.png`, `figure_11_logical_error_free_links.png`, `figure_12_far_field_rates.png` |
| Table 25 | same as Tables 22 and 23 | `run_free_link_check_placement.sh` for the trapped rates, `run_free_link_check_oldrates.sh` for the rates of Part 1 (`--results results_free_links_oldrates_multistart_gamma_sys_0.0`) | `tableP1_round_free_links.csv` and `tableP3_distance_7_side_by_side.csv` of the two check placement folders at gamma_sys = 0 |

The run with 1 starting point and the folder `preflight_300_gamma_sys_0.0` were run with the version of
`free_link_training.py` of that date, which drew the far field rates uniformly on 0.001 to 0.01, started the
fit from the regression estimate only, and wrote no Table F3. The options `--rate_draw uniform` and
`--n_starts 1` of the present script reproduce that setting.

Raw data of Part 2: `raw_training_parameters.csv` (the 4000 training baths), `raw_parameters_per_bath.csv`
(true, regression and fitted values of every unseen bath in both arms, fit cost, starting points used,
outcome), `raw_fit_starts_per_bath.csv` (every starting point of every bath), `raw_signals_test.npz` (signals
of the unseen baths for the three inputs), `raw_preflight_per_bath.csv` and
`raw_preflight_occupancy_per_bath.csv` (bounds, sensitivities and layer populations of every preflight bath),
`raw_round_channels_per_bath.csv` (check time, erasure and Pauli probabilities of every bath, clock period and
arm, with the outcome of the fit of that bath), `tableP2_surface_code_free_links.csv` (logical error rate at
distances 3, 5 and 7 and per unit time).

### Notes on Part 2

The two 300 bath preflight folders `preflight_300_trapped` and `preflight_300_old_rates`, the folder
`preflight_300_gamma_sys_0.0` and the result folders named `results_free_links_*` were produced in the working
folder of the author on Perlmutter and are copied here unchanged. `preflight_300_vertical_0.3` is a further
preflight with the links from layer 1 to layer 2 drawn on 0.01 to 0.3; it is not used in the paper. The printed
output of the Part 2 jobs is in `logs/`.

The plot scripts of this repository draw on a white background. The figures of the paper were drawn with these
scripts; Figure 1 with `SQB_ATTACH=one python plot_fft.py`.

The check placement run at the trapped rates was made twice with the same settings. The second run reproduced every table of the first to the last digit; the printed output of both is in `logs/` (`log_check_placement_trapped_rates_run_1.out` and `log_check_placement_trapped_rates_run_2.out`).

## Signal processing of the spectrum on the noise ladder (Section 3, paragraph "Fixed floor of the spectrum")

The noise ladder of Tables 8 to 11 is repeated with three other definitions of the spectrum. `noise_ladder_dsp.py`
is `noise_ladder.py` with the fixed floor and the two multitaper modes added; the mode is selected with the environment variable `SQB_FEATURE`.

| Spectrum | `SQB_FEATURE` | Regression that drives the fit and the check | Result folders |
| --- | --- | --- | --- |
| Published (Tables 8 to 11) | `published` | trained on noisy signals | `results_v3_ladder_split_gamma_sys_*` |
| Fixed floor | `fixed_floor` | trained on exact signals only | `results_v3_ladder_fixedfloor_split_gamma_sys_*` |
| Multitaper (DPSS, 3 tapers, half bandwidth 2) | `multitaper` | trained on noisy signals | `results_v3_ladder_multitaper_split_gamma_sys_*` |
| Multitaper with fixed floor | `multitaper_fixed_floor` | trained on exact signals only | `results_v3_ladder_multitaper_fixedfloor_split_gamma_sys_*` |

Fixed floor: the log spectrum is normalized to its peak and cut at a floor fixed at the noise amplitude of the
measurement, 2 / sqrt(points x shots), which is known from the number of shots. Multitaper: the spectral estimate
with discrete prolate spheroidal sequences proposed as the signal processing step of the Fourier analysis in
arXiv:2508.17514.

| Paper | Script | Run | File |
| --- | --- | --- | --- |
| Fixed floor, R2 and logical error rate | `noise_ladder_dsp.py` | `sbatch run_noise_ladder_fixed_floor.sh` | `tableL1_inference_on_ladder.csv`, `tableL3_surface_code_on_ladder.csv` of the fixed floor folders |
| Multitaper | `noise_ladder_dsp.py` | `sbatch run_noise_ladder_dsp.sh multitaper_fixed_floor` and `sbatch run_noise_ladder_dsp.sh multitaper` | the same two tables of the multitaper folders |
| Ranges and counts of the paragraph (17 of 20, 10 of 10 rungs, 36 of 40, 18 of 20) | `table_dsp_comparison.py` | `python table_dsp_comparison.py`; reads the tables of the four spectra, computes nothing new, and prints the counts under "Counts quoted in the paper" | `tableD1_regression_by_feature.csv`, `tableD2_logical_error_by_feature.csv` |

Every one of these folders holds Tables L1 to L3 and the raw data of every bath on every rung
(`raw_parameters_per_bath.csv`, `raw_round_channels_per_bath.csv`). The signals and spectra
(`raw_signals_and_spectra_<rung>.npz`) are written by the runs and are not in the repository because of their size.
The printed output of the three runs is in `logs/`.
