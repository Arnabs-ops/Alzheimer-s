"""
Behavior module: track daily routines and detect anomalies using time-series analysis.
Uses Prophet for trend/seasonality detection.
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_behavior_log(csv_path: str) -> pd.DataFrame:
    """
    Load behavior log CSV.
    
    Expected columns:
    - date/datetime column
    - activity/task columns (e.g., 'medication_taken', 'exercise_done')
    - Optional: subject_id
    
    Returns:
        DataFrame with datetime index
    """
    df = pd.read_csv(csv_path)
    
    # Try to find date/datetime column
    date_col = None
    for col in ['date', 'datetime', 'timestamp', 'time']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.sort_index()
    
    return df


def extract_prophet_features(
    behavior_df: pd.DataFrame,
    metric_column: str,
    subject_id: Optional[str] = None
) -> Dict[str, float]:
    """
    Extract time-series features using Prophet.
    
    Args:
        behavior_df: DataFrame with datetime index
        metric_column: Column name to analyze (e.g., 'activity_score', 'routine_completion')
        subject_id: Optional subject ID
    
    Returns:
        Dictionary with Prophet-derived features
    """
    try:
        from prophet import Prophet
        
        if metric_column not in behavior_df.columns:
            print(f"⚠️ Column '{metric_column}' not found. Using first numeric column.")
            numeric_cols = behavior_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {}
            metric_column = numeric_cols[0]
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': behavior_df.index,
            'y': behavior_df[metric_column].values
        })
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 7:  # Need minimum data points
            return {
                'trend_slope': 0.0,
                'trend_change': 0.0,
                'seasonal_strength': 0.0,
                'anomaly_score': 0.0
            }
        
        # Fit Prophet model
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        
        # Predict
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        
        # Extract features
        trend = forecast['trend'].values
        trend_slope = (trend[-1] - trend[0]) / len(trend) if len(trend) > 1 else 0.0
        
        # Trend change: compare first half vs second half
        mid = len(trend) // 2
        first_half_trend = (trend[mid] - trend[0]) / mid if mid > 0 else 0.0
        second_half_trend = (trend[-1] - trend[mid]) / (len(trend) - mid) if len(trend) > mid else 0.0
        trend_change = second_half_trend - first_half_trend
        
        # Seasonal strength: variance of seasonal component
        seasonal = forecast.get('weekly', forecast.get('seasonal', pd.Series([0]*len(forecast))))
        seasonal_strength = float(np.std(seasonal.values)) if len(seasonal) > 0 else 0.0
        
        # Anomaly detection: residuals
        actual = prophet_df['y'].values
        predicted = forecast['yhat'].values[:len(actual)]
        residuals = actual - predicted
        anomaly_score = float(np.mean(np.abs(residuals)) / (np.std(actual) + 1e-6)) if np.std(actual) > 0 else 0.0
        
        return {
            'trend_slope': float(trend_slope),
            'trend_change': float(trend_change),
            'seasonal_strength': float(seasonal_strength),
            'anomaly_score': float(anomaly_score),
            'mean_value': float(np.mean(actual)),
            'std_value': float(np.std(actual))
        }
    except Exception as e:
        print(f"⚠️ Prophet feature extraction failed: {e}")
        return {
            'trend_slope': 0.0,
            'trend_change': 0.0,
            'seasonal_strength': 0.0,
            'anomaly_score': 0.0
        }


def extract_routine_features(behavior_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract routine consistency features.
    
    Args:
        behavior_df: DataFrame with datetime index
    
    Returns:
        Dictionary with routine features
    """
    if len(behavior_df) == 0:
        return {
            'routine_consistency': 0.0,
            'missing_days_ratio': 0.0,
            'activity_variance': 0.0
        }
    
    # Check for missing days (gaps in time series)
    if isinstance(behavior_df.index, pd.DatetimeIndex):
        date_range = pd.date_range(start=behavior_df.index.min(), end=behavior_df.index.max(), freq='D')
        missing_days = len(date_range) - len(behavior_df)
        missing_days_ratio = missing_days / len(date_range) if len(date_range) > 0 else 0.0
    else:
        missing_days_ratio = 0.0
    
    # Activity variance (if numeric columns exist)
    numeric_cols = behavior_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        activity_variance = float(np.mean(behavior_df[numeric_cols].var(axis=0)))
    else:
        activity_variance = 0.0
    
    # Routine consistency: coefficient of variation of daily patterns
    if len(numeric_cols) > 0:
        daily_means = behavior_df[numeric_cols].mean(axis=1)
        consistency = 1.0 / (1.0 + np.std(daily_means)) if len(daily_means) > 0 else 0.0
    else:
        consistency = 0.0
    
    return {
        'routine_consistency': float(consistency),
        'missing_days_ratio': float(missing_days_ratio),
        'activity_variance': float(activity_variance)
    }


def process_behavior_log(
    csv_path: str,
    subject_id: Optional[str] = None,
    metric_column: Optional[str] = None
) -> Dict[str, float]:
    """
    Process a behavior log file and extract all features.
    
    Args:
        csv_path: Path to behavior log CSV
        subject_id: Optional subject ID
        metric_column: Column to analyze with Prophet (if None, uses first numeric)
    
    Returns:
        Dictionary with all extracted features
    """
    print(f"📊 Processing behavior log: {csv_path}")
    
    behavior_df = load_behavior_log(csv_path)
    
    if len(behavior_df) == 0:
        print("⚠️ Empty behavior log")
        return {}
    
    # Extract features
    routine_features = extract_routine_features(behavior_df)
    
    # Prophet features (if numeric data available)
    prophet_features = {}
    numeric_cols = behavior_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        metric_col = metric_column or numeric_cols[0]
        prophet_features = extract_prophet_features(behavior_df, metric_col, subject_id=subject_id)
    
    # Combine all features
    all_features = {
        'log_duration_days': float((behavior_df.index.max() - behavior_df.index.min()).days) if isinstance(behavior_df.index, pd.DatetimeIndex) else 0.0,
        'total_records': float(len(behavior_df)),
        **routine_features,
        **prophet_features
    }
    
    if subject_id:
        all_features['subject_id'] = subject_id
    
    return all_features


def process_behavior_directory(
    behavior_dir: str,
    output_path: str = "data/processed/behavior_features.csv"
) -> pd.DataFrame:
    """
    Process all behavior log CSVs in a directory.
    
    Args:
        behavior_dir: Directory containing CSV files
        output_path: Output CSV path
    
    Returns:
        DataFrame with features for all files
    """
    import glob
    
    csv_files = glob.glob(os.path.join(behavior_dir, "*.csv"))
    
    print(f"📁 Found {len(csv_files)} behavior log files")
    
    all_features = []
    for i, csv_file in enumerate(csv_files):
        print(f"   Processing {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        subject_id = os.path.splitext(os.path.basename(csv_file))[0]
        
        try:
            features = process_behavior_log(csv_file, subject_id=subject_id)
            if features:
                all_features.append(features)
        except Exception as e:
            print(f"   ⚠️ Failed to process {csv_file}: {e}")
    
    if not all_features:
        print("⚠️ No features extracted")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_features)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved behavior features to {output_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        features = process_behavior_log(csv_path)
        print(features)
    else:
        print("Usage: python behavior_input.py <behavior_log_csv_path>")

