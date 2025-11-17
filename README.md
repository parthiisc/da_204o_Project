# DA_204o_Project
Final Project for DA 204o : Data Science in Practice

## Overview
This repository contains notebooks and scripts to process soil moisture CSVs (2018 & 2020), extract date features, and produce a merged dataset used for EDA and modeling.

## Quick Setup
1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Notebooks
- `merge.ipynb` — scans `Soil_data/2018` and `Soil_data/2020`, preprocesses each CSV (normalizes columns, robustly parses `Date` into `Date_parsed`, adds `Year`/`Month`/`Day`, converts numeric-like columns), then writes per-year final CSVs and combined `merged_final.csv`.
- `eda.ipynb` — exploratory analysis using the merged dataset, includes season mapping by state.

## How to run the preprocessing (recommended)
Open a terminal, activate the virtual environment, then run the `merge.ipynb` notebook in JupyterLab or run the equivalent script cell. After processing, the outputs will be written to:

- `merged_2018_final.csv`
- `merged_2020_final.csv`
- `merged_final.csv`

If you prefer a quick script-based check (no notebook UI), run a short Python snippet that imports `merge.ipynb` logic or use the `process_csv` function in the notebook cells directly.

## Troubleshooting date parsing
- The preprocessing tries multiple common formats (`YYYY/MM/DD`, `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`, and common textual formats). If you still see missing `Date_parsed` entries, inspect the problematic raw `Date` strings and report examples — I can add custom parsing rules.

## Contact / Next steps
If you'd like, I can:
- Run the merge cell and report how many rows still have missing `Date_parsed` values and show sample problematic date strings.
- Add a small script `scripts/check_dates.py` to summarize parsing failures automatically.

---
Generated/updated on: 2025-11-17
