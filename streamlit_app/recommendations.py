"""
Crop recommendations and farmer summary utilities.
"""

import calendar
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def state_aware_crop_suggestion_with_varieties(state: str, primary_val: Optional[float]) -> Tuple[str, str, List[str]]:
    """
    Provide state-aware crop suggestions with example varieties.
    
    Args:
        state: State name
        primary_val: Predicted moisture value
        
    Returns:
        (label, explanation, varieties) tuple
    """
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


def build_farmer_summary(
    state: str,
    district: str,
    month_num: int,
    primary_val: Optional[float],
    df_region: Optional[pd.DataFrame],
    hist_month_mean: Optional[float],
    hist_month_std: Optional[float],
    preds: Dict[str, Optional[float]],
    low_thresh: float,
    high_thresh: float,
    plot_col: str
) -> Tuple[str, Dict]:
    """
    Build comprehensive farmer summary.
    
    Args:
        state: State name
        district: District name
        month_num: Month number (1-12)
        primary_val: Primary prediction value
        df_region: Regional historical data
        hist_month_mean: Historical month mean
        hist_month_std: Historical month std
        preds: All predictions dictionary
        low_thresh: Low threshold
        high_thresh: High threshold
        plot_col: Column name for plotting
        
    Returns:
        (summary_text, summary_components) tuple
    """
    summary_components = {}
    if primary_val is None:
        return "Prediction not available", summary_components

    # Month percentiles
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

    # Above/below normal
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

    # Assemble summary
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

