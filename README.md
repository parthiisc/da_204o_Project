# Soil Moisture Prediction for Agricultural Planning

## Team Members

**Note:** Team member details:
- **Parth, Patel** - parthpatel@iisc.ac.in
- **Payal Dey** - payaldey@iisc.ac.in
- **Inderjit Singh Chauhan** -  inderjits@iisc.ac.in

---

## Problem Statement

Agriculture in India is heavily dependent on soil moisture levels, which directly impact crop yield, irrigation planning, and water resource management. Traditional methods of soil moisture assessment are time-consuming, expensive, and often lack spatial and temporal coverage. This project aims to develop a machine learning-based system to predict soil moisture levels at 15cm depth for different states and districts across India, enabling farmers and agricultural planners to make data-driven decisions about crop selection, irrigation scheduling, and resource allocation.

**Key Objectives:**
- Predict soil moisture levels at 15cm depth for any state-district combination
- Provide district-level rankings within states for comparative analysis
- Offer state-aware crop recommendations based on predicted moisture levels
- Enable 3-month forward projections for planning purposes
- Create an interactive web application for easy access and visualization

---

## Dataset Description

### Data Sources

The project uses soil moisture data from multiple sources:

1. **State-wise Soil Moisture Data (2018 & 2020)**
   - Location: `data/raw/2018/` and `data/raw/2020/` folders
   - Contains CSV files for each Indian state (e.g., `sm_gujarat_2018.csv`, `sm_maharashtra_2020.csv`)
   - Each file contains district-level soil moisture measurements
   - **Total States Covered:** 36 states/union territories
   - **Temporal Coverage:** 2018 and 2020 (monthly data)

2. **Merged Dataset**
   - **File:** `data/processed/merged_final.csv`
   - **Size:** ~287,288 rows × 11 columns
   - Contains consolidated data from all states for both years
   - Includes: state, district, year, month, day, and soil moisture measurements

3. **ML-Ready Dataset**
   - **File:** `data/processed/soil_ml_ready.csv`
   - **Size:** ~287,288 rows × 25 columns
   - Preprocessed dataset with engineered features
   - Includes: temporal features, lag features, rolling statistics, and encoded variables


### Key Variables

- **Target Variable:** `average_soilmoisture_level_(at_15cm)` - Soil moisture at 15cm depth
- **Features:**
  - Temporal: `Month_num`, `month_sin`, `month_cos`, `Season_num`
  - Geographic: `state_freq`, `district_id`
  - Lag Features: `lag_1`, `lag_7` (1-day and 7-day lags)
  - Rolling Statistics: `rolling_3`, `rolling_6` (3-day and 6-day rolling means)

### Data Characteristics

- **Training Data:** 2018 (137,388 rows)
- **Test Data:** 2020 (149,900 rows)
- **Geographic Coverage:** All major Indian states and union territories
- **Temporal Resolution:** Daily measurements aggregated to monthly patterns

---

## High-Level Approach and Methods

### 1. Data Collection and Preprocessing
## File Size Notes & Model Hosting

Some data and model files are large. Recommended approaches:

- Keep large binary model files out of git history when possible. Use Git LFS for smaller teams and quota-friendly usage.
- Prefer hosting model artifacts externally (GitHub Releases, S3, GDrive) and let the app download them at runtime. This repo already includes a manifest (`models_manifest.json`) and a downloader helper located at `streamlit_app/model_downloader.py`.

How the app handles models (current setup)

- `models_manifest.json` maps model filenames to direct-download URLs (we use a GitHub Release `v1.0-models` in this repo).
- At runtime the app will attempt to load models from `model_outputs/` and `CNN/`. If models are missing, the downloader will fetch them from the manifest URLs.
- A helper script is provided: `scripts/download_models.py` (run with `PYTHONPATH=. python scripts/download_models.py`) to pre-download models into the correct folders locally.

This avoids committing large `.pth` or `.joblib` artifacts directly to the repository while still allowing the app to fetch them automatically.

## Requirements (recommended additions)

Make sure `requirements.txt` contains the runtime packages required by the app. At minimum we recommend adding:

```
streamlit
requests
joblib
xgboost
lightgbm
torch
torchvision
```

Install locally with:

```bash
pip install -r requirements.txt
```

## Quick Run (local)

From the repository root you can run the app with the convenience wrapper we added:

```bash
streamlit run streamlit_app/Home.py
```



## License

See `LICENSE` file for details.

---

## Contributing

This is a course project. For questions or issues, please contact the team members listed above.

---

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## Acknowledgments

- Data sources: [Specify your data sources]
- Libraries and tools: pandas, numpy, scikit-learn, xgboost, lightgbm, streamlit, plotly
- Course: DA_204o - Data Analytics

---

**Last Updated:** 2025-12-02
### Application Features

**Predictions:** Real-time soil moisture predictions for any state-district combination  
**Rankings:** District-level comparisons within states  
**Projections:** 3-month forward-looking forecasts  
**Recommendations:** State-specific crop suggestions based on moisture levels  
**Visualizations:** Interactive heatmaps and time series plots  
**Export:** Downloadable reports in CSV and PNG formats  

---

## Repository Structure

Top-level layout (important files and folders):

```
DA_204o_Project/
├── .devcontainer/                # VS Code devcontainer (optional)
├── .dist/                        # build/dist artifacts (optional)
├── .env                          # local env overrides (not committed)
├── .gitattributes                # Git LFS / attributes
├── .gitignore
├── .venv/                        # local virtual environment (not committed)
├── CNN/                          # CNN model Inference for Streamlit App (pytorch .pth files)
├── CNN_model/                    # CNN Trained models (if present)
├── Course project proposal.pdf
├── Final_Academic_Presentation_v3.pptx
├── LICENSE
├── README.md
├── config.yaml                   # Application configuration
├── data/                         # Raw and processed datasets
├── model_outputs/                # Trained model artifacts (joblib)
├── models_manifest.json          # Manifest mapping model filenames -> download URLs
├── requirements.txt
├── scripts/                      # Utility scripts (download, maintenance)
├── streamlit_app/                # Streamlit application package
│   ├── Home.py                   # App landing page (module)
│   ├── app.py                    # older app entry (kept for reference)
│   ├── model_utils.py            # model loading & predictions
│   ├── model_downloader.py       # helper to fetch models from manifest
│   ├── pages/                    # Streamlit multipage components
│   └── ...
├── streamlit_app.py              # legacy / alternative entry (optional)
└── scripts/download_models.py    # convenience script to pre-download models

```

Notes:
- Keep large binaries (models) outside of git history when possible; use Releases or S3 and the provided `models_manifest.json`.
- The app expects model files under `model_outputs/` (regression `.joblib`) and `CNN/` (`.pth` weights). Use `scripts/download_models.py` to populate these folders locally before running the app if desired.


---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DA_204o_Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data files:**
   ```bash
   ls data/processed/merged_final.csv data/processed/soil_ml_ready.csv
   ls model_outputs/*.joblib
   ```

4. **Run the application from DA_204o_Project folder:**
   ```bash
   streamlit run streamlit_app/Home.py
   ```


## Usage

### Running Predictions

1. Open the Streamlit app in your browser
2. Select a **State** from the dropdown
3. Select a **District** from the filtered list
4. Choose a **Prediction Date** (today or future)
5. Click **"Predict & Analyze"**

### Features Available

- **Predictions:** View XGBoost and LightGBM predictions
- **District Ranking:** See how the selected district ranks within the state
- **3-Month Projection:** View forward-looking forecasts
- **Crop Recommendations:** Get state-specific crop suggestions
- **Visualizations:** Explore heatmaps and time series plots
- **Export Data:** Download results as CSV or PNG

---

## File Size Notes

Some data files may exceed 100 MB. For large files:
- Use Git LFS (Git Large File Storage) if available
- Or provide download links in this README
- Contact repository maintainers for data access

## Git LFS — Managing Large Models

If your project contains large model files (for example `.pth` or `.joblib`), Git LFS (Large File Storage) is recommended. Below are concise instructions and best practices you can follow.

- **Install Git LFS**
   ```bash
   # macOS (Homebrew)
   brew install git-lfs
   git lfs install

   # or Debian/Ubuntu
   curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   sudo apt-get install git-lfs
   git lfs install
   ```

- **Track files**
   ```bash
   # Track model patterns used in this repo
   git lfs track "model_outputs/*.joblib"
   git lfs track "CNN/*.pth"
   git add .gitattributes
   git commit -m "Track model artifacts with Git LFS"
   ```

- **Push and pull LFS objects**
   - After committing LFS-tracked files, push as usual: `git push origin main` (Git LFS will upload the large objects automatically).
   - To fetch LFS objects after cloning: `git lfs pull` (or `git lfs fetch && git lfs checkout`).

- **Migrate existing history to LFS (advanced)**
   - If you already committed large binaries and want to move them to LFS, you can rewrite history with `git lfs migrate import`.
   - **WARNING:** this rewrites commits and requires a force-push to update the remote. Create a backup branch first:
      ```bash
      git branch backup-before-lfs
      git lfs migrate import --include="model_outputs/*.joblib,CNN/*.pth" --include-ref=refs/heads/main --yes
      git push origin main --force
      ```
   - Use migration only when necessary and when you understand the impact on collaborators and CI systems.

- **GitHub LFS limits & alternatives**
   - GitHub repositories using LFS must respect storage and bandwidth quotas. If you hit limits or want a simpler approach, host model archives externally (GitHub Releases, S3, or other) and set `STREAMLIT_MODEL_URL` (or update `config.yaml`) so the app downloads models at startup.

- **Troubleshooting**
   - If `git push` fails after migrating, check remote quotas or try pushing via SSH. You can also upload model artifacts to a Release and use a URL.
   - If collaborators cannot access LFS files after cloning, ensure they run `git lfs install` and `git lfs pull`.

---

## License

See `LICENSE` file for details.

---

## Contributing

This is a course project. For questions or issues, please contact the team members listed above.

---

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## Acknowledgments

- Data sources: [Specify your data sources]
- Libraries and tools: pandas, numpy, scikit-learn, xgboost, lightgbm, streamlit, plotly
- Course: DA_204o - Data Analytics

---

**Last Updated:** [Current Date]

