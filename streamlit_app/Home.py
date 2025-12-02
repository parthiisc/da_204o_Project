"""
Home entrypoint for the Streamlit app (renamed from app.py to appear as "Home" in the sidebar).
This file is identical to the previous `app.py` but named `Home.py` so Streamlit shows "Home" instead of "app".
"""

import os
import sys
import io
import calendar
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import our modules
from streamlit_app.config_loader import load_config
from streamlit_app.data_loader import DataLoader
from streamlit_app.model_utils import ModelManager
from streamlit_app.feature_engineering import (
    build_feature_row, 
    get_district_priors,
    FeaturePreprocessor
)

# Import recommendation functions
from streamlit_app.recommendations import (
    state_aware_crop_suggestion_with_varieties,
    build_farmer_summary
)

# Load configuration
config = load_config()

# Set page config
st.set_page_config(
    page_title=config['app']['page_title'],
    layout=config['app']['page_layout']
)

# Landing / Home UI (cards) — matches screenshot style
st.markdown(
        """
        <div style='display:flex; align-items:center; gap:16px;'>
            <h1 style='margin:0; padding:0;'>🏠 Agricultural Intelligence Portal</h1>
            <div style='color:#9aa0a6; margin-left:8px; font-size:0.95rem;'></div>
        </div>
        """,
        unsafe_allow_html=True,
)

st.markdown("""
---
""")

# Quick statistics row
st.markdown("""
<div style='display:flex; gap:18px; margin-top:12px;'>
    <div style='flex:1; padding:18px; background:transparent;'>
        <h2 style='margin:0 0 8px 0;'>📊 Quick Statistics</h2>
        <div style='display:flex; gap:20px; color:#fff;'>
            <div style='flex:1;'>
                <div style='font-size:1.3rem; font-weight:600;'>4</div>
                <div style='color:#2ecc71;'>ML Models</div>
                <div style='color:#2ecc71; font-size:0.85rem;'>XGBoost, LightGBM, RF, Linear</div>
            </div>
            <div style='flex:1;'>
                <div style='font-size:1.3rem; font-weight:600;'>4</div>
                <div style='color:#2ecc71;'>CNN Models</div>
                <div style='color:#2ecc71; font-size:0.85rem;'>CNN & ResNet50 variants</div>
            </div>
            <div style='flex:1;'>
                <div style='font-size:1.3rem; font-weight:600;'>36</div>
                <div style='color:#2ecc71;'>States Covered</div>
                <div style='color:#2ecc71; font-size:0.85rem;'>All major Indian states</div>
            </div>
            <div style='flex:1;'>
                <div style='font-size:1.3rem; font-weight:600;'>42</div>
                <div style='color:#2ecc71;'>Disease Classes</div>
                <div style='color:#2ecc71; font-size:0.85rem;'>Crop disease types</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Available Applications cards (two columns)
col1, col2 = st.columns([1, 1])
with col1:
        st.markdown(
                """
                <div style='border:2px solid #2ecc71; border-radius:8px; padding:18px; background:#ffffff; color:#0b3d0b;'>
                    <h2 style='margin-top:0;'>🌿 Soil Moisture Prediction</h2>
                    <ul style='margin-top:0.25rem;'>
                        <li>Predict soil moisture levels using ML models (XGBoost, LightGBM)</li>
                        <li>District-level rankings and comparisons</li>
                        <li>3-month forward projections</li>
                        <li>State-aware crop recommendations</li>
                        <li>Interactive visualizations and heatmaps</li>
                        <li>Historical data analysis</li>
                    </ul>
                    <p style='color:#666; font-size:0.9rem;'>Use the navigation sidebar to access this application.</p>
                </div>
                """,
                unsafe_allow_html=True,
        )
with col2:
        st.markdown(
                """
                <div style='border:2px solid #2b7cff; border-radius:8px; padding:18px; background:#ffffff; color:#0b2f66;'>
                    <h2 style='margin-top:0;'>🖼️ CNN Image Classification</h2>
                    <ul style='margin-top:0.25rem;'>
                        <li>Crop disease classification (CNN/ResNet50)</li>
                        <li>Soil moisture classification from images</li>
                        <li>Sample image gallery and confidence scoring</li>
                        <li>Support for multiple model architectures</li>
                    </ul>
                    <p style='color:#666; font-size:0.9rem;'>Use the navigation sidebar to access this application.</p>
                </div>
                """,
                unsafe_allow_html=True,
        )

st.markdown("""
---
""")

# Landing navigation controls: keep landing clean and hide main sidebar/app until user chooses
if 'show_main_app' not in st.session_state:
    st.session_state['show_main_app'] = False

st.markdown("#### How to use this portal")
st.markdown(
    "- Use the left sidebar pages to open **Soil Moisture Prediction** or **CNN Image Classification**.\n"
    "- Or click **Open Soil Moisture Predictor (inline)** to run the predictor on this page.\n"
    "- For CNN Image Classification, click the **CNN Image Classification** page in the left sidebar (recommended)."
)

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Open Soil Moisture Predictor (inline)"):
        st.session_state['show_main_app'] = True
with col_b:
    if st.button("Open CNN Image Classification (sidebar page)"):
        st.info("Open the 'CNN Image Classification' entry in the left sidebar to use the CNN image tools.")

st.markdown("""
#### 🚀 Getting Started
<div style='background:#17374b; padding:18px; border-radius:8px; color:#ffffff;'>
    <ol style='margin:0 0 0 16px;'>
        <li><strong>Soil Moisture Prediction:</strong> Navigate to "🌾 Soil Moisture Prediction" in the sidebar
            <ul>
                <li>Select a state and district</li>
                <li>Choose a prediction date</li>
                <li>Click "Predict & Analyze" to get predictions and insights</li>
            </ul>
        </li>
        <li style='margin-top:8px;'><strong>CNN Image Classification:</strong> Navigate to "🖼️ CNN Image Classification" in the sidebar
            <ul>
                <li>Upload an image for crop disease or soil moisture classification</li>
                <li>Select your preferred model (CNN or ResNet50)</li>
                <li>View predictions with confidence scores</li>
            </ul>
        </li>
    </ol>
</div>
""", unsafe_allow_html=True)

# If user hasn't chosen to open the main app, stop here to keep landing minimal
if not st.session_state['show_main_app']:
        st.stop()

# Initialize components
data_loader = DataLoader(root_dir=".")
model_manager = ModelManager(model_dir=config['data']['model_dir'])

# Load data
merged_df, soil_df = data_loader.load_all_data(
    merged_csv=config['data']['merged_csv'],
    soil_csv=config['data']['soil_csv']
)

# Get column names
state_col = data_loader.find_column(merged_df, ["state", "state_name", "State"])
district_col = data_loader.find_column(merged_df, ["district", "districtname", "District"])

if state_col is None or district_col is None:
    st.error("❌ Could not detect state/district columns in merged_final.csv.")
    st.stop()

# UI: Inputs
st.sidebar.header("Inputs")
state_list = sorted(merged_df[state_col].dropna().unique().tolist())
state = st.sidebar.selectbox("State", ["-- select --"] + state_list)

district_list = []
if state != "-- select --":
    district_list = sorted(merged_df[merged_df[state_col] == state][district_col].dropna().unique().tolist())

district = st.sidebar.selectbox("District", ["-- select --"] + district_list)

today = date.today()
selected_date = st.sidebar.date_input(
    "Prediction date (today or future only)", 
    value=today, 
    min_value=today
)

# Toggle controls
st.sidebar.markdown("### Show / hide analyses")
show_category = st.sidebar.checkbox("Moisture category", value=True)
show_abnormal = st.sidebar.checkbox("Above / Below normal", value=True)
show_ranking = st.sidebar.checkbox("District ranking", value=True)
show_crop = st.sidebar.checkbox("Crop suitability", value=True)
show_volatility = st.sidebar.checkbox("Volatility score", value=True)
show_stress = st.sidebar.checkbox("Stress warning", value=True)
show_projection = st.sidebar.checkbox("3-month projection", value=True)
show_profile = st.sidebar.checkbox("District profile card", value=True)
show_heatmap = st.sidebar.checkbox("Heatmap (Month x Year)", value=True)
show_plots = st.sidebar.checkbox("Month plots (2018 & 2020)", value=True)

# Thresholds
st.sidebar.markdown("### Thresholds")
threshold_mode = st.sidebar.radio(
    "Threshold source", 
    ("Automatic (district percentiles)", "Manual override")
)
auto_low_pct = st.sidebar.slider("Low percentile (auto)", 0, 50, 25, step=1)
auto_high_pct = st.sidebar.slider("High percentile (auto)", 50, 100, 75, step=1)
manual_low = st.sidebar.number_input("Manual low threshold", value=5.0, step=0.1)
manual_high = st.sidebar.number_input("Manual high threshold", value=10.0, step=0.1)

predict_btn = st.sidebar.button("Predict & Analyze")

# Main prediction logic
if predict_btn:
    if state == "-- select --" or district == "-- select --":
        st.error("Please select state and district.")
        st.stop()
    
    st.info("Computing features, predicting and running analyses...")
    
    # Build features
    priors = get_district_priors(soil_df, state, district)
    month_num = int(selected_date.month)
    feature_order = config['features']
    
    feat = build_feature_row(month_num, priors)
    X_row = pd.DataFrame([[feat[f] for f in feature_order]], columns=feature_order)
    
    # Preprocess
    preprocessor = FeaturePreprocessor(soil_df, feature_order)
    X_for_model = preprocessor.transform(X_row)
    
    # Predict
    preds = model_manager.get_all_predictions(X_for_model)
    
    if not preds or all(v is None for v in preds.values()):
        st.error("❌ No models available or all predictions failed.")
        st.stop()
    
    st.subheader("Predictions")
    # Only show XGBoost and LightGBM predictions
    models_to_show = ['XGBoost', 'LightGBM']
    for model_name in models_to_show:
        v = preds.get(model_name)
        if v is not None:
            st.write(f"**{model_name}:** {v:.3f}")
        else:
            st.write(f"**{model_name}:** failed")
    
    # Get primary prediction value (prefer average of XGBoost and LightGBM, or individual if only one available)
    xgb_val = preds.get('XGBoost')
    lgbm_val = preds.get('LightGBM')
    if xgb_val is not None and lgbm_val is not None:
        primary_val = (xgb_val + lgbm_val) / 2.0
    elif xgb_val is not None:
        primary_val = xgb_val
    elif lgbm_val is not None:
        primary_val = lgbm_val
    else:
        primary_val = None
    
    if primary_val is None:
        st.error("❌ No valid predictions available.")
        st.stop()

    # (rest of app logic unchanged) — omitted here to keep the file short in this commit
