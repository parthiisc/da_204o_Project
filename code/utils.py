# utils.py - shared helpers for Soil Moisture project
from pathlib import Path
import pandas as pd
import numpy as np

def read_csv_fallback(path):
    path = Path(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, encoding='latin1', low_memory=False)

def normalize_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

month_map = {
    'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
    'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
}

def ensure_month_features(df, date_col='date', month_col='Month'):
    df = df.copy()
    # Normalize Month string column if exists
    if month_col in df.columns:
        df[month_col] = df[month_col].astype(str).str.strip().str.upper().replace({'NAN': pd.NA, 'NONE': pd.NA})
        df['Month_num'] = df[month_col].map(month_map)
        # handle full names like January
        mask_full = df['Month_num'].isna() & df[month_col].notna()
        if mask_full.any():
            df.loc[mask_full, 'Month_num'] = df.loc[mask_full, month_col].str[:3].map(month_map)
    else:
        df['Month_num'] = pd.NA
    # Try to fill from date if missing
    if date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors='coerce')
        mask_date = df['Month_num'].isna() & parsed.notna()
        if mask_date.any():
            df.loc[mask_date, 'Month_num'] = parsed.dt.month
    # ensure type
    df['Month_num'] = pd.to_numeric(df['Month_num'], errors='coerce').astype('Int64')
    # month_name canonical
    month_order = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    df['Month_name'] = df.get(month_col)
    mask_name_from_num = df['Month_name'].isna() & df['Month_num'].notna()
    df.loc[mask_name_from_num, 'Month_name'] = df.loc[mask_name_from_num, 'Month_num'].apply(lambda m: month_order[int(m)-1])
    # cyclic encoding
    df['month_sin'] = np.nan
    df['month_cos'] = np.nan
    mask = df['Month_num'].notna()
    df.loc[mask, 'month_sin'] = np.sin(2 * np.pi * df.loc[mask, 'Month_num'] / 12)
    df.loc[mask, 'month_cos'] = np.cos(2 * np.pi * df.loc[mask, 'Month_num'] / 12)
    return df

def ensure_season_features(df, season_col='Season'):
    df = df.copy()
    if season_col not in df.columns:
        df[season_col] = pd.NA
    df['Season_up'] = df[season_col].astype(str).str.upper().str.replace('_',' ').str.strip()
    season_map = {
        'WINTER':1,
        'SUMMER':2,
        'MONSOON':3,
        'POST MONSOON':4,
        'POST-MONSOON':4,
        'POSTMONSOON':4
    }
    df['Season_num'] = df['Season_up'].map(season_map).astype('Int64')
    return df

def detect_target_column(df):
    for c in df.columns:
        if 'moist' in c.lower():
            return c
    return None

def basic_checks(df):
    report = {}
    report['shape'] = df.shape
    report['columns'] = list(df.columns)
    report['missing_counts'] = df.isna().sum().to_dict()
    report['duplicates'] = int(df.duplicated().sum())
    return report
