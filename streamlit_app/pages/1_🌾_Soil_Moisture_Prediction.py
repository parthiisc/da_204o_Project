"""
Soil Moisture Prediction Page
Original prediction application using ML models
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
# Go up from pages/ to streamlit_app/ to project root
streamlit_app_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(streamlit_app_dir)
# Add both paths
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if streamlit_app_dir not in sys.path:
    sys.path.insert(0, streamlit_app_dir)

# Import our modules
from streamlit_app.config_loader import load_config
from streamlit_app.data_loader import DataLoader
from streamlit_app.model_utils import ModelManager
from streamlit_app.feature_engineering import (
    build_feature_row, 
    get_district_priors,
    FeaturePreprocessor
)
from streamlit_app.recommendations import (
    state_aware_crop_suggestion_with_varieties,
    build_farmer_summary
)

# Load configuration
config = load_config()

st.title("🌾 Soil Moisture Prediction")

# Initialize components (use parent_dir as root for data files)
data_loader = DataLoader(root_dir=parent_dir)
model_dir_path = os.path.join(parent_dir, config['data']['model_dir'].lstrip('./'))
model_manager = ModelManager(model_dir=model_dir_path)

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
    
    # Build region DataFrame for analysis
    plot_col = 'average_soilmoisture_level_(at_15cm)'
    source_df = soil_df.copy() if (soil_df is not None and plot_col in soil_df.columns) else merged_df.copy()
    
    src_state_col = data_loader.find_column(source_df, [state_col, "state", "state_name"])
    src_district_col = data_loader.find_column(source_df, [district_col, "district", "districtname"])
    
    df_region = pd.DataFrame()
    if src_state_col is not None and src_district_col is not None:
        try:
            mask = (source_df[src_state_col].astype(str).str.upper() == state.upper()) & \
                   (source_df[src_district_col].astype(str).str.upper() == district.upper())
            df_region = source_df[mask].copy()
        except Exception:
            df_region = pd.DataFrame()
        
        if df_region.empty:
            try:
                mask = source_df[src_state_col].astype(str).str.upper().str.contains(state.upper(), na=False) & \
                       source_df[src_district_col].astype(str).str.upper().str.contains(district.upper(), na=False)
                df_region = source_df[mask].copy()
            except Exception:
                df_region = pd.DataFrame()
    
    if df_region.empty:
        st.warning("No historical rows found for this state/district — some analyses limited.")
    else:
        # Ensure Month column exists
        month_col = data_loader.find_column(df_region, ["Month_num", "Month", "Month_name"])
        if month_col:
            try:
                df_region['Month_num'] = pd.to_numeric(df_region[month_col], errors='coerce')
            except Exception:
                pass
        
        # Add Month abbreviation
        def month_abbrev_safe(m):
            try:
                m_i = int(m)
                if 1 <= m_i <= 12:
                    return calendar.month_abbr[m_i]
            except Exception:
                pass
            return str(m) if pd.notna(m) else "Unknown"
        
        if 'Month_num' in df_region.columns:
            df_region['Month'] = df_region['Month_num'].apply(month_abbrev_safe)
        
        # Ensure Year exists
        if 'Year' not in df_region.columns:
            date_col = data_loader.find_column(df_region, ["date", "Date", "DATE"])
            if date_col is not None:
                try:
                    df_region[date_col] = pd.to_datetime(df_region[date_col], errors='coerce')
                    df_region['Year'] = df_region[date_col].dt.year
                except Exception:
                    pass
    
    # Historical month stats
    hist_month_mean = None
    hist_month_std = None
    try:
        month_abbr_for_selected = calendar.month_abbr[month_num]
    except Exception:
        month_abbr_for_selected = None
    
    if not df_region.empty and 'Month' in df_region.columns and plot_col in df_region.columns and month_abbr_for_selected:
        hist_month_vals = df_region[df_region['Month'] == month_abbr_for_selected][plot_col].dropna().values
        if len(hist_month_vals) > 0:
            hist_month_mean = float(np.mean(hist_month_vals))
            hist_month_std = float(np.std(hist_month_vals, ddof=0))
    
    # Auto thresholds
    auto_low_val = None
    auto_high_val = None
    if threshold_mode.startswith("Automatic") and not df_region.empty and plot_col in df_region.columns:
        vals = df_region[plot_col].dropna().values
        if len(vals) > 0:
            auto_low_val = float(np.percentile(vals, auto_low_pct))
            auto_high_val = float(np.percentile(vals, auto_high_pct))
    
    if threshold_mode.startswith("Automatic") and auto_low_val is not None and auto_high_val is not None:
        low_thresh = auto_low_val
        high_thresh = auto_high_val
    else:
        low_thresh = manual_low
        high_thresh = manual_high
    
    # Build farmer summary using recommendations module
    summary_text, summary_components = build_farmer_summary(
        state=state,
        district=district,
        month_num=month_num,
        primary_val=primary_val,
        df_region=df_region,
        hist_month_mean=hist_month_mean,
        hist_month_std=hist_month_std,
        preds=preds,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        plot_col=plot_col
    )
    
    # Helper functions for downloads
    def df_to_csv_bytes(df):
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode('utf-8')
    
    def fig_to_png_bytes(fig):
        try:
            return fig.to_image(format="png", width=1200, height=800, scale=2)
        except Exception:
            return None
    
    # Calculate ranking for all districts in state
    ranking_df = pd.DataFrame()
    if state != "-- select --" and merged_df is not None:
        districts_in_state = sorted(merged_df[merged_df[state_col] == state][district_col].dropna().unique().tolist())
        rank_list = []
        for d in districts_in_state:
            p = get_district_priors(soil_df, state, d)
            feat_d = build_feature_row(month_num, p)
            X_row_d = pd.DataFrame([[feat_d[f] for f in feature_order]], columns=feature_order)
            X_for_d = preprocessor.transform(X_row_d)
            preds_d = model_manager.get_all_predictions(X_for_d)
            vals = [v for v in preds_d.values() if v is not None]
            rank_val = float(np.mean(vals)) if vals else np.nan
            rank_list.append((d, rank_val))
        ranking_df = pd.DataFrame(rank_list, columns=['district', 'pred_moisture']).sort_values('pred_moisture', ascending=False)
    
    # 3-month projection
    proj_list = []
    for i in range(3):
        m = ((month_num - 1 + i) % 12) + 1
        feat_i = build_feature_row(m, priors)
        X_row_i = pd.DataFrame([[feat_i[f] for f in feature_order]], columns=feature_order)
        X_for_i = preprocessor.transform(X_row_i)
        preds_i = model_manager.get_all_predictions(X_for_i)
        proj_list.append({
            "month_num": m,
            "month": calendar.month_abbr[m],
            "LightGBM": preds_i.get('LightGBM'),
            "XGBoost": preds_i.get('XGBoost'),
            "Ensemble": preds_i.get('Ensemble')
        })
    proj_df = pd.DataFrame(proj_list)
    
    # Monthly means for district profile
    month_means_df = pd.DataFrame()
    if not df_region.empty and plot_col in df_region.columns and 'Month' in df_region.columns:
        month_means = df_region.groupby('Month')[plot_col].mean().reset_index().rename(columns={plot_col: 'mean_moisture'})
        month_order = list(calendar.month_abbr[1:13])
        month_means['Month'] = pd.Categorical(month_means['Month'], categories=month_order, ordered=True)
        month_means = month_means.sort_values('Month')
        month_means_df = month_means
    
    # Extract summary components
    if isinstance(summary_components, dict):
        cat = summary_components.get('category')
        ab_pct = summary_components.get('above_below_pct')
        stress = summary_components.get('stress')
        crop = summary_components.get('crop_suggestion')
        crop_expl = summary_components.get('crop_explanation')
        varieties = summary_components.get('example_varieties', [])
        irrigation = summary_components.get('irrigation_advice')
        unc = summary_components.get('uncertainty', "")
    else:
        cat = ab_pct = stress = crop = crop_expl = varieties = irrigation = unc = None
    
    # Display farmer summary
    st.markdown("### Farmer summary — quick actionable points")
    
    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        if primary_val is not None:
            st.metric(label="Predicted moisture (15 cm)", value=f"{primary_val:.2f}")
        else:
            st.metric(label="Predicted moisture (15 cm)", value="N/A")
    with col_b:
        if cat == "LOW":
            st.markdown("**Category:** 🔴 **LOW**")
        elif cat == "MEDIUM":
            st.markdown("**Category:** 🟡 **MEDIUM**")
        elif cat == "HIGH":
            st.markdown("**Category:** 🟢 **HIGH**")
        else:
            st.markdown("**Category:** —")
    with col_c:
        if ab_pct is not None:
            if ab_pct >= 0:
                st.markdown(f"**Above normal:** 🔼 {ab_pct:.0f}%")
            else:
                st.markdown(f"**Below normal:** 🔽 {abs(ab_pct):.0f}%")
        else:
            st.markdown("**Above/Below normal:** —")
    
    st.markdown("---")
    
    # Stress / Uncertainty badges
    col2, col3 = st.columns([1, 1])
    with col2:
        if stress == "Severe":
            st.warning("⚠️ Severe water stress likely")
        elif stress == "Moderate":
            st.info("⚠️ Moderate water stress — consider irrigation")
        elif stress == "None":
            st.success("✅ No severe water stress predicted")
        else:
            st.markdown("**Stress:** —")
    with col3:
        if unc:
            st.markdown(f"**Uncertainty:** 🔎 {unc.strip()}")
        else:
            st.markdown("**Uncertainty:** Low")
    
    st.markdown("---")
    
    # Crop suggestion and example varieties
    col_left, col_right = st.columns([2, 1])
    with col_left:
        if crop:
            st.markdown("#### Recommended crop suitability")
            st.markdown(f"**{crop}**")
            if crop_expl:
                st.markdown(f"*{crop_expl}*")
            if varieties:
                st.markdown("**Example varieties / hybrids (examples):**")
                for v in varieties[:6]:
                    st.markdown(f"- {v}")
                st.caption("Example variety names are for guidance. For full state lists see NutriCereals / ICAR.")
        else:
            st.markdown("#### Recommended crop suitability")
            st.markdown("—")
    with col_right:
        if irrigation:
            if "Irrigate urgently" in irrigation:
                st.error(f"🚨 {irrigation}")
            elif "Consider irrigation soon" in irrigation:
                st.warning(f"⚠️ {irrigation}")
            else:
                st.success(f"✅ {irrigation}")
        else:
            st.markdown("—")
    
    st.markdown("---")
    
    # Quick points
    st.markdown("#### Quick points")
    bullet_lines = []
    if primary_val is not None:
        bullet_lines.append(f"• Predicted moisture at 15 cm: **{primary_val:.2f}**.")
    if cat is not None:
        bullet_lines.append(f"• Category: **{cat}**.")
    if ab_pct is not None:
        if ab_pct >= 0:
            bullet_lines.append(f"• This is **{ab_pct:.0f}% above** the historical monthly mean.")
        else:
            bullet_lines.append(f"• This is **{abs(ab_pct):.0f}% below** the historical monthly mean.")
    if stress == "Severe":
        bullet_lines.append("• Action: **Irrigate urgently** and inspect soil; high risk of crop stress.")
    elif stress == "Moderate":
        bullet_lines.append("• Action: Consider scheduled irrigation within a few days.")
    else:
        bullet_lines.append("• Action: No immediate irrigation recommended; monitor weekly.")
    if crop:
        bullet_lines.append(f"• Crop suggestion: {crop}.")
        if varieties:
            bullet_lines.append(f"• Example varieties: {', '.join(varieties[:3])} (see NutriCereals for full state list).")
    for bl in bullet_lines:
        st.markdown(bl)
    
    # Models used and historical years
    models_to_show = ['XGBoost', 'LightGBM']
    models_used = ", ".join([k for k in models_to_show if preds.get(k) is not None]) if preds else "None"
    years_text = "N/A"
    if (df_region is not None) and ('Year' in df_region.columns) and df_region['Year'].notna().any():
        years_text = ", ".join(map(str, sorted(df_region['Year'].dropna().unique())))
    st.caption(f"Models used: {models_used} • District historical years: {years_text}")
    
    # ---------- Display analyses and plots with download buttons ----------
    
    # Moisture category
    if show_category:
        st.markdown("#### Moisture category")
        if cat == "LOW":
            st.error(f"🔴 **LOW** — Predicted value ({primary_val:.2f}) is below threshold ({low_thresh:.2f})")
        elif cat == "MEDIUM":
            st.info(f"🟡 **MEDIUM** — Predicted value ({primary_val:.2f}) is between thresholds ({low_thresh:.2f} - {high_thresh:.2f})")
        elif cat == "HIGH":
            st.success(f"🟢 **HIGH** — Predicted value ({primary_val:.2f}) is above threshold ({high_thresh:.2f})")
        else:
            st.write("Category not available.")
    
    # Above/Below normal
    if show_abnormal:
        st.markdown("#### Above / Below normal")
        if ab_pct is not None:
            if ab_pct >= 0:
                st.success(f"🔼 **{ab_pct:.0f}% above** historical monthly mean ({hist_month_mean:.2f})")
            else:
                st.warning(f"🔽 **{abs(ab_pct):.0f}% below** historical monthly mean ({hist_month_mean:.2f})")
        else:
            st.write("Historical comparison not available.")
    
    # District ranking
    if show_ranking:
        st.markdown("#### District ranking within selected state (for selected month)")
        if not ranking_df.empty:
            ranking_display = ranking_df.reset_index(drop=True).copy()
            ranking_display.index = ranking_display.index + 1
            ranking_display.index.name = "Rank"
            st.dataframe(ranking_display.style.format({"pred_moisture": "{:.2f}"}))
            ranking_csv = ranking_display.reset_index()
            csv_bytes = df_to_csv_bytes(ranking_csv)
            st.download_button(
                "Download ranking CSV",
                data=csv_bytes,
                file_name=f"{state}_{calendar.month_abbr[month_num]}_ranking.csv",
                mime="text/csv"
            )
        else:
            st.write("Ranking not available.")
    
    # Crop suitability
    if show_crop:
        st.markdown("#### Crop suitability")
        if crop:
            st.markdown(f"**{crop}**")
            if crop_expl:
                st.markdown(crop_expl)
            if varieties:
                st.markdown("**Example varieties:**")
                for v in varieties:
                    st.markdown(f"- {v}")
        else:
            st.write("Crop suggestion not available.")
    
    # Volatility score
    if show_volatility:
        st.markdown("#### Volatility score")
        if hist_month_std is not None:
            volatility = hist_month_std / max(hist_month_mean, 0.1) if hist_month_mean else 0
            st.metric("Coefficient of Variation", f"{volatility:.2f}")
            if volatility > 0.5:
                st.warning("High volatility — conditions vary significantly")
            elif volatility > 0.3:
                st.info("Moderate volatility")
            else:
                st.success("Low volatility — relatively stable conditions")
        else:
            st.write("Volatility score not available.")
    
    # Stress warning
    if show_stress:
        st.markdown("#### Stress warning")
        if stress == "Severe":
            st.error("🚨 **Severe water stress** — Immediate action required")
        elif stress == "Moderate":
            st.warning("⚠️ **Moderate water stress** — Consider irrigation soon")
        elif stress == "None":
            st.success("✅ **No severe stress** — Conditions are acceptable")
        else:
            st.write("Stress assessment not available.")
    
    # 3-month projection
    if show_projection:
        st.markdown("#### 3-month projection")
        proj_df_display = proj_df.copy()
        proj_df_display['month'] = proj_df_display['month'].astype(str)
        proj_df_display = proj_df_display[['month', 'LightGBM', 'XGBoost', 'Ensemble']]
        st.table(proj_df_display.set_index('month').style.format("{:.2f}"))
        csv_bytes = df_to_csv_bytes(proj_df_display)
        st.download_button(
            "Download projection CSV",
            data=csv_bytes,
            file_name=f"{state}_{calendar.month_abbr[month_num]}_projection.csv",
            mime="text/csv"
        )
    
    # District profile
    if show_profile:
        st.markdown("#### District profile (historical)")
        if month_means_df.empty:
            st.write("No historical profile available.")
        else:
            profile_display = month_means_df.copy()
            profile_display.index = profile_display.index + 1
            profile_display.index.name = "No"
            st.table(profile_display.style.format({"mean_moisture": "{:.2f}"}))
            profile_csv = profile_display.reset_index()
            csv_bytes = df_to_csv_bytes(profile_csv)
            st.download_button(
                "Download monthly means CSV",
                data=csv_bytes,
                file_name=f"{state}_{district}_monthly_means.csv",
                mime="text/csv"
            )
    
    # Heatmap
    if show_heatmap:
        st.subheader("Heatmap (Month × Year)")
        if df_region.empty or plot_col not in df_region.columns or 'Month' not in df_region.columns or 'Year' not in df_region.columns:
            st.write("Heatmap not available (insufficient data).")
        else:
            pivot = df_region.pivot_table(index='Month', columns='Year', values=plot_col, aggfunc='mean')
            month_order = list(calendar.month_abbr[1:13])
            pivot = pivot.reindex(month_order)
            year_cols = []
            for c in pivot.columns:
                try:
                    year_cols.append(int(float(c)))
                except Exception:
                    try:
                        year_cols.append(int(c))
                    except Exception:
                        year_cols.append(c)
            y_labels = list(pivot.index)
            fig = px.imshow(
                pivot.values,
                labels=dict(x="Year", y="Month", color="Moisture"),
                x=year_cols,
                y=y_labels,
                aspect="auto",
                color_continuous_scale="YlGnBu"
            )
            fig.update_layout(height=480, xaxis_title="Year", yaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True)
            png = fig_to_png_bytes(fig)
            if png is not None:
                st.download_button(
                    "Download heatmap PNG",
                    data=png,
                    file_name=f"{state}_{district}_heatmap.png",
                    mime="image/png"
                )
            else:
                st.info("PNG export requires 'kaleido' (pip install kaleido) — CSV download available instead.")
                pivot_csv = pivot.reset_index()
                csv_bytes = df_to_csv_bytes(pivot_csv)
                st.download_button(
                    "Download heatmap CSV (pivot)",
                    data=csv_bytes,
                    file_name=f"{state}_{district}_heatmap.csv",
                    mime="text/csv"
                )
    
    # Month plots (2018 & 2020)
    if show_plots:
        st.subheader("Month-wise plots (2018 & 2020) — interactive")
        if df_region.empty or plot_col not in df_region.columns or 'Month' not in df_region.columns:
            st.write("Month plots not available (insufficient data).")
        else:
            df_2018 = df_region[pd.to_numeric(df_region.get('Year', pd.Series([np.nan] * len(df_region))), errors='coerce') == 2018].copy()
            df_2020 = df_region[pd.to_numeric(df_region.get('Year', pd.Series([np.nan] * len(df_region))), errors='coerce') == 2020].copy()
            
            month_names = list(calendar.month_abbr[1:13])
            month_to_num = {calendar.month_abbr[i]: i for i in range(1, 13)}
            
            def make_month_fig(df_y, year):
                if df_y.empty:
                    fig = go.Figure()
                    fig.add_annotation(text=f"No data for {year}", xref="paper", yref="paper", showarrow=False)
                    return fig
                agg = df_y.groupby('Month')[plot_col].mean().reset_index()
                agg['Month_num'] = agg['Month'].map(month_to_num).astype(float)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=agg['Month_num'],
                    y=agg[plot_col],
                    mode='lines+markers',
                    name=f"{year} mean"
                ))
                df_y_plot = df_y.dropna(subset=['Month', plot_col]).copy()
                df_y_plot['Month_num_plot'] = df_y_plot['Month'].map(month_to_num)
                jitter = (np.random.rand(len(df_y_plot)) - 0.5) * 0.2
                x_pts = df_y_plot['Month_num_plot'].values + jitter
                fig.add_trace(go.Scatter(
                    x=x_pts,
                    y=df_y_plot[plot_col],
                    mode='markers',
                    marker=dict(size=6, opacity=0.6),
                    name=f"{year} records"
                ))
                pm = month_num
                if preds.get('LightGBM') is not None:
                    fig.add_trace(go.Scatter(
                        x=[pm],
                        y=[preds['LightGBM']],
                        mode='markers+text',
                        marker=dict(size=14, color='orange'),
                        text=[f"LGB {preds['LightGBM']:.2f}"],
                        textposition="top center",
                        name="LGBM pred"
                    ))
                if preds.get('XGBoost') is not None:
                    fig.add_trace(go.Scatter(
                        x=[pm],
                        y=[preds['XGBoost']],
                        mode='markers+text',
                        marker=dict(size=14, color='green'),
                        text=[f"XGB {preds['XGBoost']:.2f}"],
                        textposition="bottom center",
                        name="XGB pred"
                    ))
                fig.update_xaxes(tickmode='array', tickvals=list(range(1, 13)), ticktext=month_names)
                fig.update_layout(height=420, xaxis_title="Month", yaxis_title="Moisture (15cm)")
                return fig
            
            fig1 = make_month_fig(df_2018, 2018)
            fig2 = make_month_fig(df_2020, 2020)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
                png1 = fig_to_png_bytes(fig1)
                if png1 is not None:
                    st.download_button(
                        "Download 2018 plot PNG",
                        data=png1,
                        file_name=f"{state}_{district}_2018_plot.png",
                        mime="image/png"
                    )
                else:
                    st.info("Install kaleido for PNG export: pip install kaleido")
                if not df_2018.empty:
                    csv_bytes = df_to_csv_bytes(df_2018)
                    st.download_button(
                        "Download 2018 raw CSV",
                        data=csv_bytes,
                        file_name=f"{state}_{district}_2018_raw.csv",
                        mime="text/csv"
                    )
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
                png2 = fig_to_png_bytes(fig2)
                if png2 is not None:
                    st.download_button(
                        "Download 2020 plot PNG",
                        data=png2,
                        file_name=f"{state}_{district}_2020_plot.png",
                        mime="image/png"
                    )
                else:
                    st.info("Install kaleido for PNG export: pip install kaleido")
                if not df_2020.empty:
                    csv_bytes = df_to_csv_bytes(df_2020)
                    st.download_button(
                        "Download 2020 raw CSV",
                        data=csv_bytes,
                        file_name=f"{state}_{district}_2020_raw.csv",
                        mime="text/csv"
                    )
    
    st.success("✅ Analysis complete. Use the download buttons to save CSV or PNG outputs.")

