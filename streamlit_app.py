# app.py
"""
Soil Moisture Predictor — Month names (Jan..Dec)
Changes in this version:
 - Trend feature removed (no trend computation, no UI toggle, no display)
 - Heatmap X-axis (Year) displays integers (no floats like 2018.0)
 - Rest of functionality retained: predictions, state-aware crop suggestions, farmer summary, plots, downloads
Place in project root with:
 - merged_final.csv
 - soil_ml_ready.csv (optional)
 - model_outputs/ (LightGBM_reg.joblib and/or XGBoost_reg.joblib)

Run:
    pip install streamlit pandas numpy scikit-learn joblib plotly kaleido
    streamlit run app.py
"""

import os
import io
import calendar
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Soil Moisture Predictor", layout="wide")
st.title("🌾 Soil Moisture Predictor")

# ---------- Paths ----------
ROOT = "."
MODEL_DIR = os.path.join(ROOT, "model_outputs")
MERGED_CSV = os.path.join(ROOT, "merged_final.csv")
SOIL_CSV = os.path.join(ROOT, "soil_ml_ready.csv")

# ---------- Helpers ----------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None

def find_col(df, candidates):
    if df is None:
        return None
    cols = {c.lower(): c for c in df.columns}
    for n in candidates:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

@st.cache_data
def load_models_cached(model_dir):
    lgbm = None; xgb = None
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith(".joblib"):
                lf = f.lower()
                p = os.path.join(model_dir, f)
                try:
                    if ("light" in lf or "lgbm" in lf) and lgbm is None:
                        lgbm = joblib.load(p)
                    if ("xgb" in lf or "xgboost" in lf) and xgb is None:
                        xgb = joblib.load(p)
                except Exception:
                    continue
    return lgbm, xgb

def season_from_month(m):
    if m in (12,1,2): return 1
    if m in (3,4,5): return 2
    if m in (6,7,8,9): return 3
    return 4

def get_district_priors(soil_df, state, district):
    priors = {}
    if soil_df is None:
        return priors
    sc = find_col(soil_df, ["state","state_name"])
    dc = find_col(soil_df, ["district","districtname"])
    if sc is None or dc is None:
        return priors
    mask = (soil_df[sc].astype(str).str.upper() == str(state).upper()) & \
           (soil_df[dc].astype(str).str.upper() == str(district).upper())
    rows = soil_df[mask].copy()
    if rows.empty:
        mask = soil_df[sc].astype(str).str.upper().str.contains(str(state).upper(), na=False) & \
               soil_df[dc].astype(str).str.upper().str.contains(str(district).upper(), na=False)
        rows = soil_df[mask].copy()
    if rows.empty:
        return priors
    if 'Year' in rows.columns:
        try:
            maxy = int(pd.to_numeric(rows['Year'], errors='coerce').dropna().max())
            rows = rows[pd.to_numeric(rows['Year'], errors='coerce') == maxy]
        except Exception:
            pass
    for key in ['state_freq','district_id','lag_1','lag_7','rolling_3','rolling_6']:
        if key in rows.columns:
            vals = rows[key].dropna()
            if len(vals)>0:
                priors[key] = float(vals.iloc[-1])
    return priors

def build_feature_row(month_num, priors):
    feat = {
        "Month_num": month_num,
        "month_sin": np.sin(2 * np.pi * month_num / 12.0),
        "month_cos": np.cos(2 * np.pi * month_num / 12.0),
        "Season_num": season_from_month(month_num)
    }
    for k in ['state_freq','district_id','lag_1','lag_7','rolling_3','rolling_6']:
        feat[k] = priors.get(k, 0.0)
    return feat

def preprocess_row(X_row, soil_df, feature_order):
    if soil_df is not None and 'Year' in soil_df.columns:
        try:
            train_df = soil_df[pd.to_numeric(soil_df['Year'], errors='coerce') == 2018]
            X_train = pd.DataFrame(columns=feature_order)
            for f in feature_order:
                X_train[f] = train_df[f] if f in train_df.columns else np.nan
            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_train_imp = imp.fit_transform(X_train.values.astype(float))
            scaler.fit(X_train_imp)
            X_row_imp = imp.transform(X_row.values.astype(float))
            X_row_scaled = scaler.transform(X_row_imp)
            return X_row_scaled
        except Exception:
            return X_row.values.astype(float)
    else:
        return X_row.values.astype(float)

def predict_for_row(model, X_for_model):
    try:
        return float(model.predict(X_for_model)[0])
    except Exception:
        return None

def ensemble_pred(preds):
    vals = [v for v in preds.values() if v is not None]
    return float(np.mean(vals)) if len(vals)>0 else None

def fig_to_png_bytes(fig):
    """Try to convert Plotly fig to PNG bytes. Requires kaleido."""
    try:
        img_bytes = fig.to_image(format="png", width=900, height=540, scale=2)
        return img_bytes
    except Exception:
        return None

def df_to_csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

# ---------- State-aware crop suggestion with example varieties ----------
def state_aware_crop_suggestion_with_varieties(state, primary_val):
    s = str(state).strip().upper() if state else ""
    if primary_val is None:
        return ("Unknown", "Prediction not available.", [])
    if primary_val < 4.0:
        moisture_group = "very_low"
    elif primary_val < 8.0:
        moisture_group = "low_moderate"
    elif primary_val < 12.0:
        moisture_group = "moderate_high"
    else:
        moisture_group = "high"

    state_varieties = {
        "RAJASTHAN": ["Pearl millet (bajra) — local hybrids", "Kodo/little millets (local)"],
        "GUJARAT": ["Pearl millet (bajra) hybrids", "Sorghum (local hybrids)"],
        "MAHARASHTRA": ["Sorghum (jowar) hybrids", "Pearl millet (bajra) hybrids"],
        "KARNATAKA": ["Finger millet (ragi) varieties", "Sorghum (jowar)"],
        "TELANGANA": ["Sorghum (jowar) hybrids", "Pearl millet hybrids"],
        "ANDHRA PRADESH": ["Sorghum, pearl millet hybrids"],
        "HARYANA": ["Pearl millet (bajra) hybrids"],
        "UTTAR PRADESH": ["Pearl millet (bajra) in drier zones"],
        "ODISHA": ["Finger millet (ragi) varieties", "Small millets"],
        "CHHATTISGARH": ["Small millets", "Sorghum"],
        "MADHYA PRADESH": ["Sorghum, pearl millet, small millets"],
        "TAMIL NADU": ["Finger millet (ragi) varieties"],
        "DEFAULT": ["Pearl millet (bajra) hybrids", "Sorghum (jowar) hybrids", "Finger millet (ragi) varieties"]
    }

    preferred = state_varieties.get(s, state_varieties["DEFAULT"])

    if moisture_group == "very_low":
        label = "Drought-tolerant millets"
        explanation = f"{', '.join(preferred[:2])} — recommended in very low soil moisture conditions."
    elif moisture_group == "low_moderate":
        label = "Moderate-moisture crops / millets"
        explanation = f"{', '.join(preferred[:2])}; consider pulses and oilseeds where local practice supports them."
    elif moisture_group == "moderate_high":
        label = "Moderate–high moisture crops"
        explanation = f"{', '.join(preferred[:2])}; if irrigation available consider higher-value crops but millets remain safe."
    else:
        label = "High moisture crops possible"
        explanation = "Conditions may support higher-water crops (paddy/sugarcane) if irrigation is available; millets/sorghum still safe where rain uncertain."

    return label, explanation, preferred

# ---------- Improved farmer summary builder (trend removed) ----------
def build_farmer_summary(state, district, month_num, primary_val,
                         df_region, hist_month_mean, hist_month_std,
                         preds, low_thresh, high_thresh, plot_col):
    summary_components = {}
    if primary_val is None:
        return "Prediction not available", summary_components

    # month percentiles (based on Month abbreviations)
    month_vals = []
    if (df_region is not None) and ('Month' in df_region.columns) and (plot_col in df_region.columns):
        try:
            month_abbr = calendar.month_abbr[int(month_num)]
            month_vals = df_region[df_region['Month'] == month_abbr][plot_col].dropna().values
        except Exception:
            month_vals = []
    pct10 = pct25 = pct50 = pct75 = pct90 = None
    if len(month_vals) > 0:
        pct10 = float(np.percentile(month_vals, 10))
        pct25 = float(np.percentile(month_vals, 25))
        pct50 = float(np.percentile(month_vals, 50))
        pct75 = float(np.percentile(month_vals, 75))
        pct90 = float(np.percentile(month_vals, 90))

    # Category
    if pct25 is not None and pct75 is not None:
        if primary_val < pct25:
            category = "LOW"
        elif primary_val > pct75:
            category = "HIGH"
        else:
            category = "MEDIUM"
    else:
        if primary_val < low_thresh:
            category = "LOW"
        elif primary_val > high_thresh:
            category = "HIGH"
        else:
            category = "MEDIUM"
    summary_components['category'] = category

    # Above/below normal using hist_month_mean
    abnorm_pct = None
    if hist_month_mean is not None and hist_month_mean != 0:
        abnorm_pct = (primary_val - hist_month_mean) / hist_month_mean * 100.0
    summary_components['above_below_pct'] = abnorm_pct

    # Stress rules
    stress_level = "None"
    if pct10 is not None:
        if primary_val < pct10:
            stress_level = "Severe"
        elif primary_val < pct25:
            stress_level = "Moderate"
        else:
            stress_level = "None"
    else:
        if hist_month_mean is not None:
            if primary_val < 0.7 * hist_month_mean:
                stress_level = "Severe"
            elif (hist_month_mean - primary_val) / hist_month_mean > 0.30:
                stress_level = "Moderate"
            else:
                stress_level = "None"
    summary_components['stress'] = stress_level

    # State-aware crop suggestion
    label, explanation, varieties = state_aware_crop_suggestion_with_varieties(state, primary_val)
    summary_components['crop_suggestion'] = label
    summary_components['crop_explanation'] = explanation
    summary_components['example_varieties'] = varieties

    # Irrigation advice
    if stress_level == "Severe":
        irrigation = "Irrigate urgently"
    elif stress_level == "Moderate":
        irrigation = "Consider irrigation soon"
    else:
        irrigation = "No immediate irrigation required"
    summary_components['irrigation_advice'] = irrigation

    # Uncertainty
    uncertainty = ""
    if hist_month_std is not None and hist_month_mean is not None:
        if hist_month_std > max(0.1, 0.3 * max(1.0, abs(hist_month_mean))):
            uncertainty = " (high uncertainty)"
        else:
            uncertainty = ""
    summary_components['uncertainty'] = uncertainty

    # Assemble one-liner (trend intentionally omitted)
    parts = []
    parts.append(f"Predicted moisture: {primary_val:.2f}")
    parts.append(f"Category: {category}")
    if abnorm_pct is not None:
        parts.append(f"{abs(abnorm_pct):.0f}% {'above' if abnorm_pct>0 else 'below'} normal")
    if stress_level == "Severe":
        parts.append("⚠️ Severe water stress likely")
    elif stress_level == "Moderate":
        parts.append("⚠️ Moderate water stress likely")
    else:
        parts.append("No severe water stress predicted")
    parts.append(f"{label}: {explanation}{uncertainty}")

    summary_text = " • ".join(parts)
    return summary_text, summary_components

# ---------- Load Data & Models ----------
merged_df = load_csv(MERGED_CSV)
soil_df = load_csv(SOIL_CSV)
lgbm_model, xgb_model = load_models_cached(MODEL_DIR)

if merged_df is None:
    st.error("merged_final.csv not found in root.")
    st.stop()

state_col = find_col(merged_df, ["state","state_name","State"])
district_col = find_col(merged_df, ["district","districtname","District"])
if state_col is None or district_col is None:
    st.error("Could not detect state/district columns in merged_final.csv.")
    st.stop()

# ---------- UI: inputs, toggles ----------
st.sidebar.header("Inputs")
state_list = sorted(merged_df[state_col].dropna().unique().tolist())
state = st.sidebar.selectbox("State", ["-- select --"] + state_list)
district_list = []
if state != "-- select --":
    district_list = sorted(merged_df[merged_df[state_col] == state][district_col].dropna().unique().tolist())
district = st.sidebar.selectbox("District", ["-- select --"] + district_list)

today = date.today()
selected_date = st.sidebar.date_input("Prediction date (today or future only)", value=today, min_value=today)

# Toggle controls for analyses (trend removed from UI)
st.sidebar.markdown("### Show / hide analyses")
show_category = st.sidebar.checkbox("Moisture category", value=True)
show_abnormal = st.sidebar.checkbox("Above / Below normal", value=True)
# show_trend removed
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
threshold_mode = st.sidebar.radio("Threshold source", ("Automatic (district percentiles)", "Manual override"))
auto_low_pct = st.sidebar.slider("Low percentile (auto)", 0, 50, 25, step=1)
auto_high_pct = st.sidebar.slider("High percentile (auto)", 50, 100, 75, step=1)
manual_low = st.sidebar.number_input("Manual low threshold", value=5.0, step=0.1)
manual_high = st.sidebar.number_input("Manual high threshold", value=10.0, step=0.1)

predict_btn = st.sidebar.button("Predict & Analyze")

# ---------- Main ----------
if predict_btn:
    if state == "-- select --" or district == "-- select --":
        st.error("Please select state and district.")
        st.stop()

    st.info("Computing features, predicting and running analyses...")

    # Build features & predict
    priors = get_district_priors(soil_df, state, district)
    month_num = int(selected_date.month)
    feature_order = ['Month_num','month_sin','month_cos','Season_num','state_freq','district_id','lag_1','lag_7','rolling_3','rolling_6']
    feat = build_feature_row(month_num, priors)
    X_row = pd.DataFrame([[feat[f] for f in feature_order]], columns=feature_order)
    X_for_model = preprocess_row(X_row, soil_df, feature_order)

    preds = {}
    if lgbm_model is not None:
        preds['LightGBM'] = predict_for_row(lgbm_model, X_for_model)
    if xgb_model is not None:
        preds['XGBoost'] = predict_for_row(xgb_model, X_for_model)
    ensemble = ensemble_pred(preds)
    if ensemble is not None:
        preds['Ensemble'] = ensemble

    st.subheader("Predictions")
    if preds:
        for k,v in preds.items():
            st.write(f"**{k}:** {v:.3f}" if v is not None else f"**{k}:** failed")
    else:
        st.error("No models available.")

    # Build df_region
    plot_col = 'average_soilmoisture_level_(at_15cm)'
    source_df = soil_df.copy() if (soil_df is not None and plot_col in soil_df.columns) else merged_df.copy()

    src_state_col = find_col(source_df, [state_col, "state","state_name"])
    src_district_col = find_col(source_df, [district_col, "district","districtname"])
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
        # Ensure Month_num and Month (Jan/Feb...) exist
        month_col = find_col(df_region, ["Month_num","Month","Month_name","Month_name"])
        if month_col is None:
            st.warning("Month column not found — month-based analyses may be limited.")
        try:
            df_region['Month_num'] = pd.to_numeric(df_region.get(month_col, df_region.get('Month_num', df_region.get('Month', pd.Series([np.nan]*len(df_region))))), errors='coerce')
        except Exception:
            df_region['Month_num'] = pd.to_numeric(df_region.get('Month_num', pd.Series([np.nan]*len(df_region))), errors='coerce')

        # Add Month abbreviation column (Jan, Feb, ...)
        def month_abbrev_safe(m):
            try:
                m_i = int(m)
                if 1 <= m_i <= 12:
                    return calendar.month_abbr[m_i]   # Jan, Feb, ...
            except Exception:
                pass
            try:
                s = str(m).strip()
                for i in range(1,13):
                    if s.upper() in (calendar.month_name[i].upper(), calendar.month_abbr[i].upper()):
                        return calendar.month_abbr[i]
            except Exception:
                pass
            return str(m)
        df_region['Month'] = df_region['Month_num'].apply(month_abbrev_safe)

        # Ensure Year exists (or build from date if available)
        if 'Year' not in df_region.columns:
            date_col = find_col(df_region, ["date","Date","DATE"])
            if date_col is not None:
                try:
                    df_region[date_col] = pd.to_datetime(df_region[date_col], errors='coerce')
                    df_region['Year'] = df_region[date_col].dt.year
                except Exception:
                    df_region['Year'] = pd.to_numeric(df_region.get('Year', pd.Series([np.nan]*len(df_region))), errors='coerce')

    # Historical month stats (using Month abbreviation)
    hist_month_vals = []
    hist_month_mean = None; hist_month_std = None
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
    auto_low_val = None; auto_high_val = None
    if threshold_mode.startswith("Automatic") and not df_region.empty and plot_col in df_region.columns:
        vals = df_region[plot_col].dropna().values
        if len(vals) > 0:
            auto_low_val = float(np.percentile(vals, auto_low_pct))
            auto_high_val = float(np.percentile(vals, auto_high_pct))

    if threshold_mode.startswith("Automatic") and auto_low_val is not None and auto_high_val is not None:
        low_thresh = auto_low_val; high_thresh = auto_high_val
    else:
        low_thresh = manual_low; high_thresh = manual_high

    primary_val = preds.get('Ensemble') if preds.get('Ensemble') is not None else (preds.get('LightGBM') or preds.get('XGBoost') or None)

    # ---------- Build analytical artifacts for downloads ----------
    # Ranking DF
    ranking_df = pd.DataFrame()
    if merged_df is not None and state is not None and state != "-- select --":
        districts_in_state = sorted(merged_df[merged_df[state_col] == state][district_col].dropna().unique().tolist())
        rank_list = []
        for d in districts_in_state:
            p = get_district_priors(soil_df, state, d)
            feat_d = build_feature_row(month_num, p)
            X_row_d = pd.DataFrame([[feat_d[f] for f in feature_order]], columns=feature_order)
            X_for_d = preprocess_row(X_row_d, soil_df, feature_order)
            vals = []
            if lgbm_model is not None:
                v = predict_for_row(lgbm_model, X_for_d); vals.append(v)
            if xgb_model is not None:
                v = predict_for_row(xgb_model, X_for_d); vals.append(v)
            vals = [v for v in vals if v is not None]
            rank_val = float(np.mean(vals)) if vals else np.nan
            rank_list.append((d, rank_val))
        ranking_df = pd.DataFrame(rank_list, columns=['district','pred_moisture']).sort_values('pred_moisture', ascending=False)

    # 3-month projection df
    proj_list = []
    for i in range(3):
        m = ((month_num - 1 + i) % 12) + 1
        feat_i = build_feature_row(m, priors)
        X_row_i = pd.DataFrame([[feat_i[f] for f in feature_order]], columns=feature_order)
        X_for_i = preprocess_row(X_row_i, soil_df, feature_order)
        preds_i = {}
        if lgbm_model is not None:
            preds_i['LightGBM'] = predict_for_row(lgbm_model, X_for_i)
        if xgb_model is not None:
            preds_i['XGBoost'] = predict_for_row(xgb_model, X_for_i)
        preds_i['Ensemble'] = ensemble_pred(preds_i)
        proj_list.append({"month_num": m, "month": calendar.month_abbr[m], "LightGBM": preds_i.get('LightGBM'), "XGBoost": preds_i.get('XGBoost'), "Ensemble": preds_i.get('Ensemble')})
    proj_df = pd.DataFrame(proj_list)

    # monthly means DF for district (Month as Jan..Dec)
    month_means_df = pd.DataFrame()
    if not df_region.empty and plot_col in df_region.columns and 'Month' in df_region.columns:
        month_means = df_region.groupby('Month')[plot_col].mean().reset_index().rename(columns={plot_col:'mean_moisture'})
        # ensure month order Jan..Dec
        month_order = list(calendar.month_abbr[1:13])
        month_means['Month'] = pd.Categorical(month_means['Month'], categories=month_order, ordered=True)
        month_means = month_means.sort_values('Month')
        month_means_df = month_means

    # features DF
    features_df = X_row.copy()

    # ---------- Farmer summary (one-liner) using improved rules ----------
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

    # ---------------- Attractive, point-wise farmer summary display ----------------
    st.markdown("### Farmer summary — quick actionable points")

    # Extract structured values safely
    if isinstance(summary_components, dict):
        cat = summary_components.get('category')
        ab_pct = summary_components.get('above_below_pct')
        # trend removed
        stress = summary_components.get('stress')
        crop = summary_components.get('crop_suggestion')
        crop_expl = summary_components.get('crop_explanation')
        varieties = summary_components.get('example_varieties', [])
        irrigation = summary_components.get('irrigation_advice')
        unc = summary_components.get('uncertainty', "")
    else:
        cat = ab_pct = stress = crop = crop_expl = varieties = irrigation = unc = None

    # Top KPIs
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

    # Stress / Uncertainty badges (trend removed)
    col2, col3 = st.columns([1,1])
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
    col_left, col_right = st.columns([2,1])
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

    # Plain-language bullet points (trend omitted)
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

    # Small caption: models used and available historical years
    models_used = ", ".join([k for k,v in preds.items() if v is not None]) if preds else "None"
    years_text = "N/A"
    if (df_region is not None) and ('Year' in df_region.columns) and df_region['Year'].notna().any():
        years_text = ", ".join(map(str, sorted(df_region['Year'].dropna().unique())))
    st.caption(f"Models used: {models_used} • District historical years: {years_text}")

    # ---------- Display analyses and plots with download buttons ----------
    # Ranking display & download (INDEX STARTS FROM 1)
    if show_ranking:
        st.markdown("#### District ranking within selected state (for selected month)")
        if not ranking_df.empty:
            ranking_display = ranking_df.reset_index(drop=True).copy()
            ranking_display.index = ranking_display.index + 1
            ranking_display.index.name = "Rank"
            st.dataframe(ranking_display.style.format({"pred_moisture":"{:.2f}"}))
            ranking_csv = ranking_display.reset_index()
            csv_bytes = df_to_csv_bytes(ranking_csv)
            st.download_button("Download ranking CSV", data=csv_bytes, file_name=f"{state}_{calendar.month_abbr[month_num]}_ranking.csv", mime="text/csv")
        else:
            st.write("Ranking not available.")

    # Projection download
    if show_projection:
        st.markdown("#### 3-month projection")
        proj_df_display = proj_df.copy()
        proj_df_display['month'] = proj_df_display['month'].astype(str)
        proj_df_display = proj_df_display[['month','LightGBM','XGBoost','Ensemble']]
        st.table(proj_df_display.set_index('month').style.format("{:.2f}"))
        csv_bytes = df_to_csv_bytes(proj_df_display)
        st.download_button("Download projection CSV", data=csv_bytes, file_name=f"{state}_{calendar.month_abbr[month_num]}_projection.csv", mime="text/csv")

    # District profile (monthly means) display & download — Month names shown
    if show_profile:
        st.markdown("#### District profile (historical)")
        if month_means_df.empty:
            st.write("No historical profile available.")
        else:
            profile_display = month_means_df.copy()
            profile_display.index = profile_display.index + 1
            profile_display.index.name = "No"
            st.table(profile_display.style.format({"mean_moisture":"{:.2f}"}))
            profile_csv = profile_display.reset_index()
            csv_bytes = df_to_csv_bytes(profile_csv)
            st.download_button("Download monthly means CSV", data=csv_bytes, file_name=f"{state}_{district}_monthly_means.csv", mime="text/csv")

    # Plots: heatmap (Month names as Y) — Year axis as integers
    if show_heatmap:
        st.subheader("Heatmap (Month × Year)")
        if df_region.empty or plot_col not in df_region.columns or 'Month' not in df_region.columns or 'Year' not in df_region.columns:
            st.write("Heatmap not available (insufficient data).")
        else:
            pivot = df_region.pivot_table(index='Month', columns='Year', values=plot_col, aggfunc='mean')
            # reorder as Jan..Dec
            month_order = list(calendar.month_abbr[1:13])
            pivot = pivot.reindex(month_order)
            # ensure year column labels are integers (convert floats to ints)
            year_cols = []
            for c in pivot.columns:
                try:
                    # convert float-like to int
                    year_cols.append(int(float(c)))
                except Exception:
                    # fallback: keep as string
                    try:
                        year_cols.append(int(c))
                    except Exception:
                        year_cols.append(c)
            # plot with integer year labels
            y_labels = list(pivot.index)
            fig = px.imshow(pivot.values,
                            labels=dict(x="Year", y="Month", color="Moisture"),
                            x=year_cols,
                            y=y_labels,
                            aspect="auto",
                            color_continuous_scale="YlGnBu")
            fig.update_layout(height=480, xaxis_title="Year", yaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True)
            png = fig_to_png_bytes(fig)
            if png is not None:
                st.download_button("Download heatmap PNG", data=png, file_name=f"{state}_{district}_heatmap.png", mime="image/png")
            else:
                st.info("PNG export requires 'kaleido' (pip install kaleido) — CSV download available instead.")
                pivot_csv = pivot.reset_index()
                csv_bytes = df_to_csv_bytes(pivot_csv)
                st.download_button("Download heatmap CSV (pivot)", data=csv_bytes, file_name=f"{state}_{district}_heatmap.csv", mime="text/csv")

    # Month plots (2018 & 2020) — x-axis shows month names
    if show_plots:
        st.subheader("Month-wise plots (2018 & 2020) — interactive")
        if df_region.empty or plot_col not in df_region.columns or 'Month' not in df_region.columns:
            st.write("Month plots not available (insufficient data).")
        else:
            df_2018 = df_region[pd.to_numeric(df_region.get('Year', pd.Series([np.nan]*len(df_region))), errors='coerce') == 2018].copy()
            df_2020 = df_region[pd.to_numeric(df_region.get('Year', pd.Series([np.nan]*len(df_region))), errors='coerce') == 2020].copy()

            # month names and tick mapping
            month_names = list(calendar.month_abbr[1:13])
            month_to_num = {calendar.month_abbr[i]: i for i in range(1,13)}

            def make_month_fig(df_y, year):
                if df_y.empty:
                    fig = go.Figure()
                    fig.add_annotation(text=f"No data for {year}", xref="paper", yref="paper", showarrow=False)
                    return fig
                # aggregate mean by month abbreviation but keep numeric for plotting
                agg = df_y.groupby('Month')[plot_col].mean().reset_index()
                agg['Month_num'] = agg['Month'].map(month_to_num).astype(float)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=agg['Month_num'], y=agg[plot_col],
                                         mode='lines+markers', name=f"{year} mean"))
                # raw points jittered
                df_y_plot = df_y.dropna(subset=['Month', plot_col]).copy()
                df_y_plot['Month_num_plot'] = df_y_plot['Month'].map(month_to_num)
                jitter = (np.random.rand(len(df_y_plot)) - 0.5) * 0.2
                x_pts = df_y_plot['Month_num_plot'].values + jitter
                fig.add_trace(go.Scatter(x=x_pts, y=df_y_plot[plot_col], mode='markers',
                                         marker=dict(size=6, opacity=0.6), name=f"{year} records"))
                pm = month_num
                if preds.get('LightGBM') is not None:
                    fig.add_trace(go.Scatter(x=[pm], y=[preds['LightGBM']],
                                             mode='markers+text', marker=dict(size=14, color='orange'),
                                             text=[f"LGB {preds['LightGBM']:.2f}"], textposition="top center",
                                             name="LGBM pred"))
                if preds.get('XGBoost') is not None:
                    fig.add_trace(go.Scatter(x=[pm], y=[preds['XGBoost']],
                                             mode='markers+text', marker=dict(size=14, color='green'),
                                             text=[f"XGB {preds['XGBoost']:.2f}"], textposition="bottom center",
                                             name="XGB pred"))
                fig.update_xaxes(tickmode='array', tickvals=list(range(1,13)), ticktext=month_names)
                fig.update_layout(height=420, xaxis_title="Month", yaxis_title="Moisture (15cm)")
                return fig

            fig1 = make_month_fig(df_2018, 2018)
            fig2 = make_month_fig(df_2020, 2020)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
                png1 = fig_to_png_bytes(fig1)
                if png1 is not None:
                    st.download_button("Download 2018 plot PNG", data=png1, file_name=f"{state}_{district}_2018_plot.png", mime="image/png")
                else:
                    st.info("Install kaleido for PNG export: pip install kaleido")
                if not df_2018.empty:
                    csv_bytes = df_to_csv_bytes(df_2018)
                    st.download_button("Download 2018 raw CSV", data=csv_bytes, file_name=f"{state}_{district}_2018_raw.csv", mime="text/csv")
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
                png2 = fig_to_png_bytes(fig2)
                if png2 is not None:
                    st.download_button("Download 2020 plot PNG", data=png2, file_name=f"{state}_{district}_2020_plot.png", mime="image/png")
                else:
                    st.info("Install kaleido for PNG export: pip install kaleido")
                if not df_2020.empty:
                    csv_bytes = df_to_csv_bytes(df_2020)
                    st.download_button("Download 2020 raw CSV", data=csv_bytes, file_name=f"{state}_{district}_2020_raw.csv", mime="text/csv")


    st.success("Analysis complete. Use the download buttons to save CSV or PNG outputs.")