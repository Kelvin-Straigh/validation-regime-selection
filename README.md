# Validation-Regime Selection and Deep-Learning Fragility in Volatility Forecasting

This repository holds the code and supporting data for the paper:

**Silent Failure: Validation-Regime Selection as One Source of Deep-Learning Fragility in Volatility Forecasting.**

Author: Kelvin Afriyie Apenteng
ORCID: 0009-0000-5856-3082
Preprint: [SSRN link to add]

## What the paper shows

The paper tests one question. Does the volatility of the validation window change how a tuned deep-learning model forecasts in a high-volatility regime?

The design varies one thing and holds the rest fixed. The training data is the same. The test period is the same and frozen. The hyperparameter search space is the same. The only difference is the validation window.

- The calm arm validates on 2013 to 2014.
- The stress arm validates on 2015 to 2016.

The finding. The model tuned on the calm window forecasts variances that are too small in the high-volatility part of the test set. The model tuned on the stress window does not. The penalty is larger for the higher-capacity model, the Temporal Fusion Transformer, than for the smaller LSTM.

The arms differ only in the validation window. So the effect can act only through hyperparameter selection. That is the mechanism the paper isolates.

## Data

The study uses daily S&P 500 prices, ticker ^GSPC, from August 2000 to January 2025.

The data is public. It is pulled at run time from Yahoo Finance through the `yfinance` package. Raw prices are not stored in this repository.

The realized variance proxy is Garman-Klass. It is computed from the daily open, high, low, and close.

The four GARCH baselines are fitted in R. Their test-set forecasts are provided here as `garch_forecasts.csv` so the full comparison runs without setting up R.

## How to run

1. Install the dependencies.

   ```
   pip install -r requirements.txt
   ```

2. Open the notebook `validation_regime_experiment.ipynb`.

3. Run the cells from top to bottom.

The first experiment cell pulls the data, computes Garman-Klass variance, runs the Optuna search for both arms, trains the six deep-learning models, and saves `calm_vs_stress_results.csv`.

The later cells load the GARCH forecasts, build the combined ten-model panel, run the Diebold-Mariano tests, and save the figures.

A GPU is recommended for the Temporal Fusion Transformer. The notebook also runs on CPU, but training is slower.

## Reproducibility

The Optuna sampler is seeded with seed 42. The search selects the same configurations on each run, and the result is stable.

Two sources of small numerical drift remain. GPU training is not fully deterministic. Yahoo Finance can revise historical prices over time. The file `calm_vs_stress_full.pkl` holds the exact forecasts and selected hyperparameters behind the reported numbers, so the published values can be checked without re-running the search.

## Files

- `validation_regime_experiment.ipynb` — the main experiment, the evaluation, and the figures.
- `tft_training_window_robustness.ipynb` — TFT training-length robustness sweep, re-selected per window. Re-runs the Optuna search for the TFT at three training-window lengths and recomputes the calm-versus-stress high-volatility QLIKE gap at each.
- `tft_training_window_robustness_fixed_config.ipynb` — the fixed-configuration version of the TFT sweep, behind Table 4. Holds the hyperparameters fixed at the main-experiment selections and varies only the training window. Seed 42.
- `lstm_training_window_robustness.ipynb` — LSTM training-length robustness sweep. The counterpart to the TFT sweep. Includes the floor check used to choose the shortest training arm, then the re-selection sweep across the three lengths.
- `garch_forecasts.R` — fits the four GARCH baselines in R (rugarch). Reads `sp500_for_garch.csv` and writes `garch_forecasts.csv`.
- `garch_forecasts.csv` — GARCH baseline forecasts, the output of the R script. Provided so the combined panel runs without setting up R.
- `sp500_for_garch.csv` — date, returns, and Garman-Klass variance, written by the main notebook and read by the R script. The two pipelines share one S&P 500 series.
- `calm_vs_stress_full.pkl` — banked forecasts and selected hyperparameters.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT.

## License

MIT. See `LICENSE`.
