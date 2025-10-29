#!/usr/bin/env python3
"""
Quick Hyperparameter Tuning Script
Lightweight version for fast experimentation
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('./src')

from src.hyper_tuning import run_random_search
from src.advanced_model import AdvancedAlzheimerModel, load_real_data

# Set thread limits
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def main():
    """Quick hyperparameter tuning for key models"""
    
    print("⚡ Quick Hyperparameter Tuning")
    print("=" * 40)
    
    # Load data
    print("📊 Loading data...")
    X, y = load_real_data()
    print(f"✅ Data loaded: X {X.shape}, y {y.shape}")
    
    # Clean data
    def clean_data(X):
        X = np.array(X, dtype=np.float64)
        X = np.where(np.isinf(X), np.nan, X)
        X = np.where(np.abs(X) > 1e10, np.nan, X)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        return X
    
    X = clean_data(X)
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Models to tune (quick versions)
    models_to_tune = {
        'Random Forest': {
            'n_estimators': [50, 100, 150],
            'max_depth': [6, 8, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 0.5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
    
    results = {}
    
    for model_name, param_grid in models_to_tune.items():
        print(f"\n🎯 Tuning {model_name}...")
        
        # Create base model
        if model_name == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(random_state=42, n_jobs=1)
        elif model_name == 'XGBoost':
            import xgboost as xgb
            base_model = xgb.XGBClassifier(random_state=42, verbosity=0, n_jobs=1)
        elif model_name == 'LightGBM':
            import lightgbm as lgb
            base_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1)
        
        # Run random search
        try:
            tuned_model, tuning_info = run_random_search(
                model=base_model,
                model_name=model_name,
                X=X_train,
                y=y_train,
                n_iter=15,  # Quick search
                cv_folds=3,
                n_jobs=1,
                random_state=42
            )
            
            # Evaluate
            tuned_model.fit(X_train, y_train)
            train_score = tuned_model.score(X_train, y_train)
            test_score = tuned_model.score(X_test, y_test)
            
            results[model_name] = {
                'model': tuned_model,
                'best_params': tuning_info['best_params'],
                'best_cv_score': tuning_info['best_score'],
                'train_score': train_score,
                'test_score': test_score
            }
            
            print(f"✅ {model_name}:")
            print(f"  CV Score: {tuning_info['best_score']:.4f}")
            print(f"  Train Score: {train_score:.4f}")
            print(f"  Test Score: {test_score:.4f}")
            print(f"  Best Params: {tuning_info['best_params']}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    # Summary
    if results:
        print(f"\n📊 Quick Tuning Summary:")
        print("-" * 40)
        for name, info in results.items():
            print(f"{name:15} | CV: {info['best_cv_score']:.4f} | Test: {info['test_score']:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        summary_data = []
        for name, info in results.items():
            summary_data.append({
                'Model': name,
                'CV_Score': info['best_cv_score'],
                'Train_Score': info['train_score'],
                'Test_Score': info['test_score']
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(results_dir, f'quick_tuning_summary_{timestamp}.csv')
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\n💾 Results saved to: {csv_path}")


if __name__ == '__main__':
    main()
