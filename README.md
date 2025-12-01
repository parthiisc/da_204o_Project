# Soil Moisture Prediction for Agricultural Planning

## Team Members

**Note:** Team member details:
- **Parth, Patel** - Roll Number 1 - Email 1
- **Payal Dey** - Roll Number 2 - Email 2
- **Inderjit Singh Chauhan** - Roll Number 3 - Email 3

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
- **Notebook:** `0_merge.ipynb`
- Merged state-wise CSV files into a unified dataset
- Standardized column names and data formats
- Handled missing values and data inconsistencies

### 2. Exploratory Data Analysis (EDA)
- **Notebook:** `1_EDA_master.ipynb`
- Analyzed distributions, correlations, and temporal patterns
- Identified seasonal variations and state-wise differences
- Generated visualizations for key insights
- Outputs stored in `eda_outputs/` folder

### 3. Feature Engineering
- **Notebook:** `2_Feature_Engineering.ipynb`
- Created cyclic encoding for months (sin/cos transformations)
- Generated lag features (1-day, 7-day) for temporal dependencies
- Computed rolling statistics (3-day, 6-day means)
- Encoded categorical variables (state, district)
- Produced ML-ready dataset: `soil_ml_ready.csv`

### 4. Modeling
- **Notebook:** `3_Modeling.ipynb`
- **Models Used:**
  - Linear Regression (baseline)
  - Random Forest
  - XGBoost
  - LightGBM
- **Evaluation Strategy:**
  - Temporal split: 2018 (train) / 2020 (test)
  - Metrics: RMSE, MAE, R²
  - Cross-validation for hyperparameter tuning
- **Preprocessing:**
  - Median imputation for missing values
  - StandardScaler for feature normalization
- **Model Persistence:** Trained models saved in `model_outputs/` as `.joblib` files

### 5. Application Development
- **Technology:** Streamlit web framework
- **Location:** `streamlit_app/` package
- **Features:**
  - Interactive state/district selection
  - Real-time predictions with XGBoost and LightGBM
  - District rankings within states
  - 3-month forward projections
  - State-aware crop recommendations
  - Interactive visualizations (heatmaps, time series plots)
  - Downloadable reports (CSV, PNG)

### 6. Model Architecture

```
Input Features (10 features)
    ↓
[Preprocessing: Imputation + Scaling]
    ↓
[Model Ensemble: XGBoost + LightGBM]
    ↓
[Average Prediction]
    ↓
Output: Soil Moisture Prediction + Analysis
```

---

## Summary of Results

### Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | [Value] | [Value] | [Value] |
| Random Forest | [Value] | [Value] | [Value] |
| XGBoost | [Value] | [Value] | [Value] |
| LightGBM | [Value] | [Value] | [Value] |
| Ensemble (XGBoost + LightGBM) | [Value] | [Value] | [Value] |

*Note: Update with actual performance metrics from your model evaluation*

### Key Insights

1. **Temporal Patterns:** Strong seasonal variations observed, with peak moisture during monsoon months (June-September)
2. **Geographic Variations:** Significant differences across states, with coastal regions showing higher average moisture
3. **Model Performance:** Ensemble approach (XGBoost + LightGBM) provides best balance of accuracy and robustness
4. **Feature Importance:** Temporal features (month, season) and lag features are most predictive

### Application Features

**Predictions:** Real-time soil moisture predictions for any state-district combination  
**Rankings:** District-level comparisons within states  
**Projections:** 3-month forward-looking forecasts  
**Recommendations:** State-specific crop suggestions based on moisture levels  
**Visualizations:** Interactive heatmaps and time series plots  
**Export:** Downloadable reports in CSV and PNG formats  

---

## Repository Structure

```
DA_204o_Project/
├── README.md                          # This file
├── LICENSE                            # License file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Application configuration
│
├── data/                              # Data files
│   ├── raw/                          # Original data files
│   │   ├── 2018/                    # 2018 state-wise CSVs
│   │   └── 2020/                     # 2020 state-wise CSVs
│   ├── processed/                    # Processed datasets
│   │   ├── merged_final.csv          # Merged dataset
│   │   └── soil_ml_ready.csv        # ML-ready dataset
│   └── Soil_data/                    # Additional soil data
│
├── code/                              # Code and notebooks
│   ├── 0_merge.ipynb                 # Data merging
│   ├── 1_EDA_master.ipynb            # Exploratory Data Analysis
│   ├── 2_Feature_Engineering.ipynb  # Feature engineering
│   ├── 3_Modeling.ipynb              # Model training
│   ├── model_comparison.ipynb        # Model comparison
│   ├── retrain_pipeline.py           # Retraining script
│   └── utils.py                      # Utility functions
│
├── streamlit_app/                     # Streamlit application
│   ├── app.py                        # Main application
│   ├── config_loader.py              # Configuration loader
│   ├── data_loader.py                # Data loading module
│   ├── model_utils.py                # Model management
│   ├── feature_engineering.py        # Feature building
│   └── recommendations.py            # Crop recommendations
│
├── model_outputs/                     # Trained models
│   ├── LightGBM_reg.joblib
│   ├── XGBoost_reg.joblib
│   ├── RandomForest_reg.joblib
│   └── Linear_reg.joblib
│
├── eda_outputs/                       # EDA visualizations
│   ├── correlation_matrix.png
│   ├── monthly_by_year.png
│   └── ...
│
└── docs/                              # Documentation
    ├── QUICK_START.md
    ├── SETUP_AND_RUN_GUIDE.md
    ├── IMPLEMENTATION_SUMMARY.md
    └── CODE_REVIEW_AND_IMPROVEMENTS.md
```

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

4. **Run the application:**
   ```bash
   streamlit run streamlit_app/app.py
   ```

The application will open in your browser at `http://localhost:8501`

### Quick Start

For detailed setup instructions, see:
- `QUICK_START.md` - Quick reference guide
- `SETUP_AND_RUN_GUIDE.md` - Comprehensive setup guide

---

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

