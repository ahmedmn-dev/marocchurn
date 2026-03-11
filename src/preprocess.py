"""
src/preprocess.py
─────────────────
Reusable preprocessing pipeline for MarocChurn.
Imports and uses this module in notebooks and app.py.

Author : Ahmed Mansof · Master Data Science · ENS Tétouan
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ── Constants ─────────────────────────────────────────────────────
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_ratio']
TARGET_COL     = 'Churn'
DROP_COLS      = ['customerID']


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and fix types."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(f"[load_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values."""
    df = df.copy()
    # TotalCharges: fill with median (new customers with 0 tenure)
    median_tc = df['TotalCharges'].median()
    n_missing  = df['TotalCharges'].isnull().sum()
    df['TotalCharges'].fillna(median_tc, inplace=True)
    print(f"[handle_missing] Filled {n_missing} TotalCharges nulls with median={median_tc:.2f}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create new predictive features."""
    df = df.copy()
    # Ratio of monthly charges to tenure (high = new + expensive = risk)
    df['charges_ratio']    = df['MonthlyCharges'] / (df['tenure'] + 1)
    # Binary flags
    df['is_new_customer']  = (df['tenure'] < 6).astype(int)
    df['high_value']       = (df['MonthlyCharges'] > 70).astype(int)
    df['has_online_security'] = (df.get('OnlineSecurity', pd.Series(['No']*len(df))) == 'Yes').astype(int)
    df['has_tech_support']    = (df.get('TechSupport',    pd.Series(['No']*len(df))) == 'Yes').astype(int)
    print(f"[feature_engineering] Added 5 new features. Total: {df.shape[1]} cols")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple:
    """
    Label-encode all object columns.
    Returns (df_encoded, dict_of_encoders)
    """
    df = df.copy()
    encoders = {}
    cat_cols  = df.select_dtypes('object').columns.tolist()
    for col in cat_cols:
        if col == TARGET_COL:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    print(f"[encode_categorical] Encoded {len(cat_cols)} categorical columns")
    return df, encoders


def scale_numerical(df: pd.DataFrame, scaler=None, fit: bool = True) -> tuple:
    """
    MinMax scale numerical columns.
    If fit=True, fit a new scaler; else transform with existing scaler.
    Returns (df_scaled, scaler)
    """
    df = df.copy()
    cols_to_scale = [c for c in NUMERICAL_COLS if c in df.columns]
    if fit:
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"[scale_numerical] Fitted and transformed {cols_to_scale}")
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        print(f"[scale_numerical] Transformed {cols_to_scale} with existing scaler")
    return df, scaler


def full_pipeline(raw_path: str, save_dir: str = None) -> dict:
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    raw_path : str — path to raw CSV
    save_dir : str — optional directory to save processed files

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # 1. Load
    df = load_data(raw_path)

    # 2. Drop unnecessary columns
    df.drop(columns=DROP_COLS, errors='ignore', inplace=True)

    # 3. Handle missing
    df = handle_missing(df)

    # 4. Feature engineering
    df = feature_engineering(df)

    # 5. Encode target
    df[TARGET_COL] = (df[TARGET_COL].astype(str).str.strip() == 'Yes').astype(int)

    # 6. Encode categoricals
    df, encoders = encode_categorical(df)

    # 7. Split X / y
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # 8. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 9. Scale numerical
    X_train, scaler = scale_numerical(X_train, fit=True)
    X_test,  _      = scale_numerical(X_test,  scaler=scaler, fit=False)

    # 10. SMOTE on training only
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] Train: {X_train.shape} → {X_train_res.shape}")

    # 11. Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        X_train_res.to_csv(f'{save_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{save_dir}/X_test.csv', index=False)
        pd.Series(y_train_res).to_csv(f'{save_dir}/y_train.csv', index=False)
        pd.Series(y_test).to_csv(f'{save_dir}/y_test.csv', index=False)
        os.makedirs('../models', exist_ok=True)
        with open('../models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('../models/feature_names.pkl', 'wb') as f:
            pickle.dump(list(X.columns), f)
        print(f"[full_pipeline] Saved to {save_dir}/")

    return {
        'X_train': X_train_res,
        'X_test':  X_test,
        'y_train': y_train_res,
        'y_test':  y_test,
        'scaler':  scaler,
        'feature_names': list(X.columns),
        'encoders': encoders,
    }


if __name__ == '__main__':
    result = full_pipeline(
        raw_path='../data/raw/telco_churn.csv',
        save_dir='../data/processed'
    )
    print("\n✅ Pipeline complete!")
    print(f"   X_train: {result['X_train'].shape}")
    print(f"   X_test:  {result['X_test'].shape}")
    print(f"   Features: {result['feature_names']}")
