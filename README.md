# Predicting quay crane productivity per shift — reproducibility package

Code and synthetic data that accompany the paper

> A. Cerdán, A.Galarreta, and A. Melgar, "A Reproducible Procedure for Predicting Quay Crane Productivity per Shift in a Container Terminal," submitted to CI2A 2027.

The paper develops and evaluates models that predict crane moves per hour (CMPH) for each vessel call and eight-hour shift. The terminal's data are confidential. This repository therefore publishes a **synthetic dataset** with the same schema and runs the paper's final model on it, so that the pipeline can be inspected and run end to end without the original data.

## Contents

| Path | What it is |
|---|---|
| `data/synthetic/synthetic_shifts.csv` | Synthetic dataset: 1,984 call–shift rows from 963 vessel calls (one row per vessel call and shift) |
| `src/experimentos.py` | Shared pipeline: temporal partition by call, baselines, 30-day and service recent levels, preprocessing, temporal cross-validation, paired bootstrap by call |
| `reports/experimentos/e1_hiperparametros.json` | Tuned hyperparameters (Ridge `alpha = 46.4`; XGBoost; LightGBM) |
| `reproducibility/final_model.ipynb` | Final model (Ridge, 35 numerical + 5 categorical predictors) on the synthetic data: cross-validation, validation, test, per shift and per call, and a real-vs-synthetic table |
| `reproducibility/evaluacion.py` | Evaluation functions used by the notebook |
| `scripts/compare_algorithms.py` | Comparison of algorithms (Table III of the paper) on the synthetic data |
| `reproducibility/generador.py` | Code of the generator, for transparency. Its fitted parameters were estimated on the terminal's data and **are not released**, so it cannot be re-run here |

Code comments and some identifiers are in Spanish; the data dictionary below translates every column.

## How to run

```bash
pip install -r requirements.txt
cd reproducibility && jupyter nbconvert --to notebook --execute final_model.ipynb   # or open it in Jupyter
cd .. && python scripts/compare_algorithms.py . sintetico                          # ~5–10 min
```

Expected results on the published synthetic dataset (Python 3.11–3.12, scikit-learn 1.7–1.9):

| | Synthetic | Real data (paper) |
|---|---|---|
| Ridge MAE, temporal cross-validation | 2.071 | 2.067 |
| Ridge MAE, test (per shift) | 1.941 | 1.963 |
| Reduction in MAE vs. 30-day berth mean, test | 18.4% [10.5, 25.6] | 13.3% [5.3, 20.5] |
| Best algorithm in cross-validation | Ridge (XGBoost and SVR within its CI) | Ridge |

The generator was calibrated on cross-validation and validation errors only, with a tolerance of ±5% on the MAE of the final model fixed before generating. Averaged over ten generator seeds, the MAE differs from the real one by −1.6% in cross-validation, −0.2% in validation, and 0.0% in test. The single published seed can fall outside the tolerance on one split (validation here); the criterion applies to the ten-seed average. The simple baselines are weaker on the synthetic data than on the real data, so improvements and R² are larger. The synthetic terminal plan (`plan_terminal`) is only a noisy linear function of each call's CMPH and does not reproduce the real plan.

## How the synthetic data were generated

1. Vessel calls are drawn in sequence with the real gaps between calls; dates are shifted to 2030–2032. Categories are anonymized (`servicio_01`, `operador_linea_01`, …).
2. Call-level numerical variables come from a Gaussian copula with empirical marginals; shift-level variables from a copula conditioned on the call, the position of the shift and the shift of day, with AR(1) persistence within the call.
3. CMPH uses the paper's definition: latent productivity = trend + service effect + k·g(X) + call noise + shift noise, where g is a gradient-boosting model fitted to the residuals of the real data (not Ridge, so the comparison of algorithms is not circular); moves ~ Poisson(productivity × crane-hours); CMPH = moves / crane-hours. Shifts removed by the quality filters are simulated and removed, as in the real data.
4. Privacy: no synthetic row copies a real one. The 5th percentile of the standardized distance from each synthetic row to the nearest real row is 2.67, against 2.08 between real rows; no category name, call identifier or date matches the real data.

## Data dictionary (`synthetic_shifts.csv`)

One row per vessel call (`visita_id`, `seq`) and shift. "Proxy" marks variables measured from the executed operation as a proxy for the shift plan (Section IV-A of the paper); they are excluded from the restricted variant.

| Column | Meaning |
|---|---|
| `visita_id`, `seq` | Vessel call identifier (synthetic, sequential) |
| `fecha_turno`, `inicio_turno`, `turno` | Date, start time and shift of day (T1 07:00, T2 15:00, T3 23:00) |
| `cmph` | **Target**: crane moves per hour = `movs` / `horas_grua` |
| `movs`, `horas_grua` | Quay crane moves and effective crane-hours in the shift |
| `n_gruas`, `qc_1`…`qc_5` | Number of quay cranes and presence of each crane (proxy) |
| `n_tractos`, `n_rtg`, `n_reach_stacker`, `n_empty_handler`, `n_tractos_por_grua` | Yard equipment assigned and tractors per crane (proxy) |
| `pct_reefer`, `pct_hazmat`, `pct_oog`, `pct_twin`, `pct_sobre_cubierta`, `n_bahias` | Container mix of the shift: share of reefer, hazardous, out-of-gauge, twin-lift and on-deck moves; bays worked (proxy) |
| `pts_descarga`, `pts_embarque`, `pts_max_grua` | Yard discharge and loading points served; maximum per crane (proxy) |
| `adjacent_berth_active` | Whether the adjacent berth operated during the shift (proxy) |
| `cit_ofertadas_imp`, `cit_ofertadas_exp` | External truck appointments offered, import and export |
| `pct_carga_pendiente` | Share of the declared cargo of the call still pending at the start of the shift (from previous shifts), in [0, 1] |
| `esc_total_ctn`, `esc_transbordo`, `esc_reestiba`, `esc_tapas` | Declared cargo of the call: total containers, transshipments, restows, hatch covers |
| `esc_pct_40`, `esc_pct_vacio`, `esc_pct_descarga` | Declared cargo shares: 40-ft, empty, discharge |
| `atraso_itinerario_h`, `horas_cutoff_a_inicio` | Hours of delay against the published schedule; hours from the cargo cutoff to the start of the shift |
| `servicio`, `operador_linea`, `clase_buque`, `pais` | Liner service, line operator, vessel class, flag (anonymized) |
| `plan_terminal` | Synthetic stand-in for the terminal's planned CMPH, one value per call |

The 30-day berth mean (`nivel_reciente_30d`) and the service mean of the last five calls (`nivel_servicio_reciente`) are computed by the pipeline (`agregar_e5` in `src/experimentos.py`) using only shifts that ended before the one predicted.

## License

Code: MIT (see `LICENSE`). Synthetic dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this code or data, please cite the paper above (full reference to be added after publication).
