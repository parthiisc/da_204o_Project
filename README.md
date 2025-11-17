# DA_204o_Project
Final Project for DA 204o : Data Science in Practice

## Overview

This repository contains Jupyter notebooks and helper code to preprocess per-state soil-moisture CSV files (2018 & 2020), extract date and seasonal features, run exploratory data analysis, create ML-ready datasets, and train / compare models.

Key capabilities:
- Per-file preprocessing: normalize columns, parse date columns, add `Year` / `Month` / `Day`.
- Merge per-year CSVs and produce `merged_final.csv` for EDA and modeling.
- Notebooks for EDA, feature engineering and model training (including optional XGBoost / LightGBM).

## Prerequisites

- Python 3.9+ (recommended)
- A virtual environment (recommended)

Dependencies are listed in `requirements.txt`.

## Quick install

```bash
# create virtualenv (macOS / Linux)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

If you run into install issues for `xgboost` or `lightgbm`, those are optional for the modeling notebooks and can be omitted.

## Notebooks (recommended order)

- `0_merge.ipynb` — Preprocess raw CSV files under `Soil_data/2018` and `Soil_data/2020`. Adds `Year`, `Month`, `Day` and writes per-year merged files and a combined `merged_final.csv`.
- `1_EDA_master.ipynb` — Master EDA (uses helpers in `utils.py`) and saves plots to `eda_outputs/`.
- `2_Feature_Engineering.ipynb` — Creates ML-ready features and saves `soil_ml_ready.csv`.
- `3_Modeling.ipynb` — Trains baseline models (Linear, RandomForest) and optionally XGBoost / LightGBM; saves results to `model_outputs/`.
- `model_comparison.ipynb` — Aggregates model metrics from saved outputs for comparison.


## Typical workflow

1. Activate virtual environment and install deps (see Quick install).
2. Run `0_merge.ipynb` (execute top-to-bottom) to preprocess raw CSVs and produce:
	- `merged_soil_data_2018.csv` (per-year merged)
	- `merged_soil_data_2020.csv`
	- `merged_final.csv` (combined, cleaned)
3. Run `1_EDA_master.ipynb` to inspect the merged dataset and save EDA outputs.
4. Run `2_Feature_Engineering.ipynb` to produce `soil_ml_ready.csv`.
5. Run `3_Modeling.ipynb` to train models and save metrics.

## File / folder overview

- `Soil_data/` — raw per-state CSVs (2018 and 2020 subfolders).
- `merge.ipynb` — preprocessing and merging logic.
- `utils.py` — shared helper functions used by EDA / feature engineering notebooks.
- `requirements.txt` — Python dependencies.
- `eda_outputs/` — default output folder for EDA plots and tables (created by notebooks).
- `soil_ml_ready.csv` — output of feature engineering used for modeling.
- `model_outputs/` — saved models and metrics.

## Troubleshooting

- Date parsing: the preprocessing in `0_merge.ipynb` uses `pandas.to_datetime(..., errors='coerce', dayfirst=True)` and will coerce unparseable values to `NaT`. If many `NaT` values remain, inspect a few raw date strings and consider adding specific parsing rules.
- Encoding errors reading CSVs: notebooks use a utf-8 → latin1 fallback when reading files.
- If a notebook complains about missing columns, run upstream notebooks in order (merge → EDA → feature engineering).


---
Last updated: 2025-11-18
