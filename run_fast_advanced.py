#!/usr/bin/env python3
"""
Fast Advanced AI Model - Quick improvements on NPZ data
Focuses on key improvements without heavy computation
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('./src')

def main():
    print("🚀 Fast Advanced AI Model - NPZ Data")
    print("=" * 50)
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.svm import SVC
        import xgboost as xgb
        import lightgbm as lgb
        import joblib
        from datetime import datetime
        
        # Configuration
        RANDOM_STATE = 42
        RESULTS_DIR = 'results/advanced'
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        print("📊 Loading NPZ data...")
        
        # Load NPZ data
        data = np.load('data/raw/preprocessed_alz_data.npz', allow_pickle=True)
        keys = list(data.keys())
        
        # Find X and y arrays
        X_key = None
        y_key = None
        
        for key in keys:
            if key.lower() in ['x', 'features', 'x_train']:
                X_key = key
            elif key.lower() in ['y', 'labels', 'target', 'y_train']:
                y_key = key
        
        if X_key and y_key:
            X = data[X_key]
            y = data[y_key]
            
            # Handle multi-dimensional y
            if len(y.shape) > 1:
                if y.shape[1] == 1:
                    y = y.ravel()
                else:
                    y = np.argmax(y, axis=1)
            
            print(f"✅ Loaded NPZ data: X {X.shape}, y {y.shape}")
        else:
            print("❌ Could not find X and y in NPZ file")
            return None, None
        
        print(f"📈 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Classes: {len(np.unique(y))} (Distribution: {np.bincount(y)})")
        
        # Quick data cleaning
        print("🧹 Quick data cleaning...")
        X = np.where(np.isinf(X), np.nan, X)
        X = np.where(np.abs(X) > 1e10, np.nan, X)
        
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        print("🔄 Creating train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"📚 Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Advanced preprocessing
        print("⚙️ Advanced preprocessing...")
        
        # Feature scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(50, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        print(f"🔍 Selected {X_train_selected.shape[1]} features from {X_train_scaled.shape[1]}")
        
        # Define models (simplified for speed)
        models = {
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, multi_class='ovr'),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False, verbosity=0),
            'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            'SVM': SVC(random_state=RANDOM_STATE, probability=True)
        }
        
        print("🤖 Training models...")
        results = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='accuracy', n_jobs=-1)
            
            # Train and predict
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'pred': y_pred
            }
            
            print(f"    {name}: {accuracy:.4f} (CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f})")
        
        # Create ensemble
        print("🎯 Creating ensemble...")
        ensemble_models = [
            ('rf', results['Random Forest']['model']),
            ('xgb', results['XGBoost']['model']),
            ('lgb', results['LightGBM']['model'])
        ]
        
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train_selected, y_train)
        ensemble_pred = ensemble.predict(X_test_selected)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        ensemble_cv = cross_val_score(ensemble, X_train_selected, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        
        results['Ensemble'] = {
            'model': ensemble,
            'accuracy': ensemble_accuracy,
            'cv_mean': ensemble_cv.mean(),
            'cv_std': ensemble_cv.std(),
            'pred': ensemble_pred
        }
        
        print(f"    Ensemble: {ensemble_accuracy:.4f} (CV: {ensemble_cv.mean():.4f}±{ensemble_cv.std():.4f})")
        
        # Results summary
        results_data = []
        for name, res in results.items():
            results_data.append({
                'Model': name,
                'Accuracy': res['accuracy'],
                'CV_Mean': res['cv_mean'],
                'CV_Std': res['cv_std']
            })
        
        results_df = pd.DataFrame(results_data).sort_values('Accuracy', ascending=False)
        results_df.reset_index(drop=True, inplace=True)
        
        print("\n🏆 RESULTS SUMMARY:")
        print(results_df.round(4))
        
        # Best model
        best_model_name = results_df.iloc[0]['Model']
        best_result = results[best_model_name]
        
        print(f"\n🥇 BEST MODEL: {best_model_name}")
        print(f"📈 Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
        print(f"📈 CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report for {best_model_name}:")
        print(classification_report(y_test, best_result['pred']))
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(f'{RESULTS_DIR}/fast_advanced_results_{timestamp}.csv', index=False)
        
        best_model_path = f'{RESULTS_DIR}/best_fast_model_{best_model_name.replace(" ", "_")}_{timestamp}.pkl'
        joblib.dump(best_result['model'], best_model_path)
        
        print(f"\n💾 Saved:")
        print(f"  - Results: {RESULTS_DIR}/fast_advanced_results_{timestamp}.csv")
        print(f"  - Best Model: {best_model_path}")
        
        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"  - Previous NPZ Results: 100% accuracy")
        print(f"  - Fast Advanced Model: {best_result['accuracy']*100:.2f}% accuracy")
        
        if best_result['accuracy'] >= 0.99:
            print(f"🎯 EXCELLENT! Maintains near-perfect performance!")
        elif best_result['accuracy'] >= 0.95:
            print(f"🎯 GREAT! Shows excellent performance!")
        else:
            print(f"🎯 GOOD! Shows solid performance!")
        
        print(f"\n✨ Fast Advanced AI Model Complete!")
        
        return results_df, best_result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results_df, best_result = main()
    
    if results_df is not None:
        print("\n🚀 Fast Advanced Model Ready!")
    else:
        print("\n❌ Training failed.")
