"""
Complete retraining pipeline: Merge → Feature Engineering → Model Training
Run this script to regenerate merged_final.csv, soil_ml_ready.csv, and retrain models.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import (
    read_csv_fallback, 
    normalize_columns, 
    ensure_month_features, 
    ensure_season_features, 
    detect_target_column
)

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Optional: XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - will skip XGBoost model")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available - will skip LightGBM model")

ROOT = Path.cwd()
print("=" * 70)
print("SOIL MOISTURE PREDICTION - COMPLETE RETRAINING PIPELINE")
print("=" * 70)


# ============================================================================
# STEP 1: MERGE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: MERGING DATA FILES")
print("=" * 70)

def merge_soil_data():
    """Merge all state CSV files from 2018 and 2020 folders."""
    base_folder_2018 = ROOT / 'data' / 'raw' / 'Soil_data' / '2018'
    base_folder_2020 = ROOT / 'data' / 'raw' / 'Soil_data' / '2020'
    
    all_dfs = []
    
    # Process 2018 files
    if base_folder_2018.exists():
        print(f"\n📁 Processing 2018 data from: {base_folder_2018}")
        csv_files_2018 = list(base_folder_2018.glob('*.csv'))
        print(f"   Found {len(csv_files_2018)} CSV files")
        
        for csv_file in csv_files_2018:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                
                # Add Year if not present
                if 'year' not in df.columns:
                    df['Year'] = 2018
                else:
                    df['Year'] = 2018  # Override to ensure consistency
                
                # Detect and parse date column
                date_cols = [c for c in df.columns if "date" in c]
                if date_cols:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
                    df['Month'] = df[date_col].dt.month
                    df['Day'] = df[date_col].dt.day
                
                all_dfs.append(df)
                print(f"   ✅ {csv_file.name}: {len(df)} rows")
            except Exception as e:
                print(f"   ❌ Error processing {csv_file.name}: {str(e)}")
    
    # Process 2020 files
    if base_folder_2020.exists():
        print(f"\n📁 Processing 2020 data from: {base_folder_2020}")
        csv_files_2020 = list(base_folder_2020.glob('*.csv'))
        print(f"   Found {len(csv_files_2020)} CSV files")
        
        for csv_file in csv_files_2020:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                
                # Add Year if not present
                if 'year' not in df.columns:
                    df['Year'] = 2020
                else:
                    df['Year'] = 2020  # Override to ensure consistency
                
                # Detect and parse date column
                date_cols = [c for c in df.columns if "date" in c]
                if date_cols:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
                    df['Month'] = df[date_col].dt.month
                    df['Day'] = df[date_col].dt.day
                
                all_dfs.append(df)
                print(f"   ✅ {csv_file.name}: {len(df)} rows")
            except Exception as e:
                print(f"   ❌ Error processing {csv_file.name}: {str(e)}")
    
    if not all_dfs:
        raise FileNotFoundError("No CSV files found in data/raw/Soil_data/2018 or data/raw/Soil_data/2020")
    
    # Merge all DataFrames
    print(f"\n📊 Merging {len(all_dfs)} DataFrames...")
    merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    print(f"   ✅ Merged shape: {merged_df.shape}")
    
    # Create output directory if it doesn't exist
    output_dir = ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged file
    output_path = output_dir / 'merged_final.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"   💾 Saved to: {output_path}")
    
    return merged_df

# Run merge
try:
    merged_df = merge_soil_data()
    print(f"\n✅ STEP 1 COMPLETE: Merged {len(merged_df)} rows")
except Exception as e:
    print(f"\n❌ STEP 1 FAILED: {str(e)}")
    sys.exit(1)


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

def create_ml_ready_features(df):
    """Create ML-ready features from merged data."""
    print(f"\n📊 Starting with {len(df)} rows")
    
    # Normalize columns
    df = normalize_columns(df)
    
    # Ensure month & season features
    df = ensure_month_features(df)
    df = ensure_season_features(df)
    
    # Identify target
    target = detect_target_column(df)
    if target is None:
        raise ValueError("Target column (moisture) not found")
    print(f"   ✅ Target column: {target}")
    
    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"   ✅ Removed {initial_len - len(df)} duplicates")
    
    # Sort for lag features
    df = df.sort_values(['state_name', 'districtname', 'Year', 'Month_num', 'Day']).reset_index(drop=True)
    
    # Create lag and rolling features per state,district
    print("   🔄 Creating lag and rolling features...")
    grp = df.groupby(['state_name', 'districtname'])
    df['lag_1'] = grp[target].shift(1)
    df['lag_7'] = grp[target].shift(7)
    df['rolling_3'] = grp[target].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['rolling_6'] = grp[target].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    
    # Simple encodings
    print("   🔄 Creating encodings...")
    df['state_freq'] = df['state_name'].map(df['state_name'].value_counts(normalize=True))
    df['district_id'] = df['districtname'].astype('category').cat.codes
    
    # Target transform
    df['target_log1p'] = np.log1p(df[target].clip(lower=0))
    
    print(f"   ✅ Final shape: {df.shape}")
    return df

# Run feature engineering
try:
    ml_ready_df = create_ml_ready_features(merged_df)
    
    # Create output directory if it doesn't exist
    output_dir = ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ML-ready file
    output_path = output_dir / 'soil_ml_ready.csv'
    ml_ready_df.to_csv(output_path, index=False)
    print(f"\n   💾 Saved to: {output_path}")
    print(f"✅ STEP 2 COMPLETE: Created ML-ready dataset with {len(ml_ready_df)} rows")
except Exception as e:
    print(f"\n❌ STEP 2 FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# STEP 3: MODEL TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: MODEL TRAINING")
print("=" * 70)

def compute_rmse(y_true, y_pred):
    """Robust RMSE computation."""
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return (mean_squared_error(y_true, y_pred)) ** 0.5

def train_models(df):
    """Train all models and save them."""
    # Identify target
    target_candidates = [c for c in df.columns if 'average_soilmoisture_level' in c.lower()]
    if not target_candidates:
        raise KeyError('Target column not found')
    target = target_candidates[0]
    print(f"   ✅ Target: {target}")
    
    # Prepare features
    features = ['Month_num', 'month_sin', 'month_cos', 'Season_num', 
                'state_freq', 'district_id', 'lag_1', 'lag_7', 
                'rolling_3', 'rolling_6']
    features = [f for f in features if f in df.columns]
    print(f"   ✅ Features ({len(features)}): {features}")
    
    # Temporal split: 2018 = train, 2020 = test
    train = df[df['Year'] == 2018].copy()
    test = df[df['Year'] == 2020].copy()
    print(f"\n   📊 Train: {len(train)} rows (2018)")
    print(f"   📊 Test: {len(test)} rows (2020)")
    
    # Prepare X and y - remove rows with NaN targets
    train_clean = train.dropna(subset=[target]).copy()
    test_clean = test.dropna(subset=[target]).copy()
    
    X_train = train_clean[features].copy()
    y_train = train_clean[target].copy()
    X_test = test_clean[features].copy()
    y_test = test_clean[target].copy()
    
    print(f"   📊 After removing NaN targets:")
    print(f"      Train: {len(X_train)} rows")
    print(f"      Test: {len(X_test)} rows")
    
    # Impute and scale
    print("\n   🔄 Preprocessing...")
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Create output directory
    output_dir = ROOT / 'model_outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Store results
    results = {}
    
    # Train Linear Regression
    print("\n   🤖 Training Linear Regression...")
    try:
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        y_pred_lr = lr_model.predict(X_test_scaled)
        
        results['Linear'] = {
            'RMSE': compute_rmse(y_test, y_pred_lr),
            'MAE': mean_absolute_error(y_test, y_pred_lr),
            'R2': r2_score(y_test, y_pred_lr)
        }
        
        # Save model
        joblib.dump(lr_model, output_dir / 'Linear_reg.joblib')
        print(f"      ✅ RMSE: {results['Linear']['RMSE']:.4f}")
    except Exception as e:
        print(f"      ❌ Error: {str(e)}")
    
    # Train Random Forest
    print("\n   🤖 Training Random Forest...")
    try:
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        y_pred_rf = rf_model.predict(X_test_scaled)
        
        results['RandomForest'] = {
            'RMSE': compute_rmse(y_test, y_pred_rf),
            'MAE': mean_absolute_error(y_test, y_pred_rf),
            'R2': r2_score(y_test, y_pred_rf)
        }
        
        # Save model
        joblib.dump(rf_model, output_dir / 'RandomForest_reg.joblib')
        print(f"      ✅ RMSE: {results['RandomForest']['RMSE']:.4f}")
    except Exception as e:
        print(f"      ❌ Error: {str(e)}")
    
    # Train XGBoost
    if XGBOOST_AVAILABLE:
        print("\n   🤖 Training XGBoost...")
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            y_pred_xgb = xgb_model.predict(X_test_scaled)
            
            results['XGBoost'] = {
                'RMSE': compute_rmse(y_test, y_pred_xgb),
                'MAE': mean_absolute_error(y_test, y_pred_xgb),
                'R2': r2_score(y_test, y_pred_xgb)
            }
            
            # Save model
            joblib.dump(xgb_model, output_dir / 'XGBoost_reg.joblib')
            print(f"      ✅ RMSE: {results['XGBoost']['RMSE']:.4f}")
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
    
    # Train LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n   🤖 Training LightGBM...")
        try:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            y_pred_lgb = lgb_model.predict(X_test_scaled)
            
            results['LightGBM'] = {
                'RMSE': compute_rmse(y_test, y_pred_lgb),
                'MAE': mean_absolute_error(y_test, y_pred_lgb),
                'R2': r2_score(y_test, y_pred_lgb)
            }
            
            # Save model
            joblib.dump(lgb_model, output_dir / 'LightGBM_reg.joblib')
            print(f"      ✅ RMSE: {results['LightGBM']['RMSE']:.4f}")
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
    
    # Save results
    import json
    results_path = output_dir / 'regression_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   💾 Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R2':<12}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} {metrics['R2']:<12.4f}")
    print("=" * 70)
    
    return results

# Run model training
try:
    results = train_models(ml_ready_df)
    print(f"\n✅ STEP 3 COMPLETE: Trained {len(results)} models")
except Exception as e:
    print(f"\n❌ STEP 3 FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ RETRAINING PIPELINE COMPLETE!")
print("=" * 70)
print(f"\n📁 Generated Files:")
print(f"   ✅ merged_final.csv ({len(merged_df):,} rows)")
print(f"   ✅ soil_ml_ready.csv ({len(ml_ready_df):,} rows)")
print(f"   ✅ model_outputs/ (trained models)")
print(f"\n🎯 Next Steps:")
print(f"   1. Run: streamlit run streamlit_app/app.py")
print(f"   2. Test predictions in the Streamlit app")
print("=" * 70)

