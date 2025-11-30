"""
Feature engineering utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def season_from_month(month_num: int) -> int:
    """
    Convert month number to season.
    
    Args:
        month_num: Month number (1-12)
        
    Returns:
        Season number: 1=Winter, 2=Summer, 3=Monsoon, 4=Post-Monsoon
    """
    if month_num in (12, 1, 2):
        return 1  # Winter
    if month_num in (3, 4, 5):
        return 2  # Summer
    if month_num in (6, 7, 8, 9):
        return 3  # Monsoon
    return 4  # Post-Monsoon


def get_district_priors(soil_df: Optional[pd.DataFrame], state: str, district: str) -> Dict[str, float]:
    """
    Get prior features for a specific district.
    
    Args:
        soil_df: DataFrame containing soil data
        state: State name (case-insensitive)
        district: District name (case-insensitive)
        
    Returns:
        Dictionary with prior features
    """
    priors = {}
    if soil_df is None:
        return priors
    
    # Find state and district columns
    state_col = None
    district_col = None
    
    for col in soil_df.columns:
        if 'state' in col.lower():
            state_col = col
        if 'district' in col.lower():
            district_col = col
    
    if state_col is None or district_col is None:
        return priors
    
    # Try exact match
    mask = (soil_df[state_col].astype(str).str.upper() == str(state).upper()) & \
           (soil_df[district_col].astype(str).str.upper() == str(district).upper())
    rows = soil_df[mask].copy()
    
    # Try partial match if exact fails
    if rows.empty:
        mask = (soil_df[state_col].astype(str).str.upper().str.contains(str(state).upper(), na=False)) & \
               (soil_df[district_col].astype(str).str.upper().str.contains(str(district).upper(), na=False))
        rows = soil_df[mask].copy()
    
    if rows.empty:
        return priors
    
    # Get most recent year's data
    if 'Year' in rows.columns:
        try:
            max_year = int(pd.to_numeric(rows['Year'], errors='coerce').dropna().max())
            rows = rows[pd.to_numeric(rows['Year'], errors='coerce') == max_year]
        except Exception:
            pass
    
    # Extract prior features
    for key in ['state_freq', 'district_id', 'lag_1', 'lag_7', 'rolling_3', 'rolling_6']:
        if key in rows.columns:
            vals = rows[key].dropna()
            if len(vals) > 0:
                priors[key] = float(vals.iloc[-1])
    
    return priors


def build_feature_row(month_num: int, priors: Dict[str, float]) -> Dict[str, float]:
    """
    Build feature row for prediction.
    
    Args:
        month_num: Month number (1-12)
        priors: Dictionary of prior features
        
    Returns:
        Dictionary of features
    """
    feat = {
        "Month_num": float(month_num),
        "month_sin": np.sin(2 * np.pi * month_num / 12.0),
        "month_cos": np.cos(2 * np.pi * month_num / 12.0),
        "Season_num": float(season_from_month(month_num))
    }
    
    # Add prior features with defaults
    for k in ['state_freq', 'district_id', 'lag_1', 'lag_7', 'rolling_3', 'rolling_6']:
        feat[k] = priors.get(k, 0.0)
    
    return feat


class FeaturePreprocessor:
    """Handles feature preprocessing (imputation and scaling)."""
    
    def __init__(self, soil_df: Optional[pd.DataFrame], feature_order: List[str]):
        """
        Initialize preprocessor.
        
        Args:
            soil_df: Training data DataFrame
            feature_order: List of feature names in order
        """
        self.feature_order = feature_order
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self._fit(soil_df)
    
    def _fit(self, soil_df: Optional[pd.DataFrame]) -> None:
        """Fit imputer and scaler on training data."""
        if soil_df is None or 'Year' not in soil_df.columns:
            logger.warning("No soil data for preprocessing - will use raw values")
            return
        
        try:
            train_df = soil_df[pd.to_numeric(soil_df['Year'], errors='coerce') == 2018]
            if train_df.empty:
                logger.warning("No 2018 data for preprocessing")
                return
            
            X_train = pd.DataFrame(columns=self.feature_order)
            for f in self.feature_order:
                if f in train_df.columns:
                    X_train[f] = train_df[f]
                else:
                    X_train[f] = np.nan
            
            X_train_imp = self.imputer.fit_transform(X_train.values.astype(float))
            self.scaler.fit(X_train_imp)
            logger.info("Preprocessor fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting preprocessor: {str(e)}")
    
    def transform(self, X_row: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted imputer and scaler.
        
        Args:
            X_row: DataFrame with features
            
        Returns:
            Transformed feature array
        """
        try:
            if self.imputer is None or self.scaler is None:
                return X_row.values.astype(float)
            
            X_imp = self.imputer.transform(X_row.values.astype(float))
            return self.scaler.transform(X_imp)
        except Exception as e:
            logger.warning(f"Preprocessing failed, using raw values: {str(e)}")
            return X_row.values.astype(float)

