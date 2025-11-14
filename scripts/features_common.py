"""
Common utilities for feature extraction and preprocessing across modalities.
"""
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


def save_features(df: pd.DataFrame, output_path: str, modality: str = "tabular"):
    """Save extracted features to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {modality} features to {output_path}")


def load_features(file_path: str) -> pd.DataFrame:
    """Load features from CSV."""
    return pd.read_csv(file_path)


def normalize_features(
    X_train: np.ndarray, 
    X_test: Optional[np.ndarray] = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[StandardScaler]]:
    """Normalize features using StandardScaler."""
    scaler = StandardScaler() if fit_scaler else None
    if fit_scaler:
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train
    
    X_test_scaled = None
    if X_test is not None:
        if fit_scaler:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
    
    return X_train_scaled, X_test_scaled, scaler


def impute_missing(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    strategy: str = 'median'
) -> Tuple[np.ndarray, Optional[np.ndarray], SimpleImputer]:
    """Impute missing values."""
    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test) if X_test is not None else None
    return X_train_imputed, X_test_imputed, imputer


def combine_modality_features(
    feature_files: Dict[str, str],
    output_path: str
) -> pd.DataFrame:
    """Combine features from multiple modalities."""
    dfs = []
    id_cols = ['subject_id', 'patient_id', 'id']
    
    for modality, file_path in feature_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Find the ID column
            common_id = None
            for id_col in id_cols:
                if id_col in df.columns:
                    common_id = id_col
                    break
            
            # Add modality prefix to feature columns (except common IDs)
            feature_cols = [c for c in df.columns if c not in id_cols and c != 'label']
            rename_dict = {c: f"{modality}_{c}" for c in feature_cols}
            df = df.rename(columns=rename_dict)
            
            # Keep label if present (only from first dataframe)
            if 'label' in df.columns and modality != list(feature_files.keys())[0]:
                df = df.drop(columns=['label'])
            
            dfs.append(df)
        else:
            print(f"⚠️ {modality} features file not found: {file_path}")
    
    if not dfs:
        raise ValueError("No feature files found")
    
    # Merge on common ID column
    merged = dfs[0]
    for df in dfs[1:]:
        # Try to find common ID column
        common_id = None
        for id_col in id_cols:
            if id_col in merged.columns and id_col in df.columns:
                common_id = id_col
                break
        
        if common_id:
            merged = pd.merge(merged, df, on=common_id, how='outer')
        else:
            # Concatenate if no common ID (assume same order)
            merged = pd.concat([merged.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    
    merged.to_csv(output_path, index=False)
    print(f"✅ Combined features saved to {output_path}")
    return merged

