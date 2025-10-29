#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Alzheimer's Disease Prediction
Uses Optuna for intelligent parameter search with pruning and timeout
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

from src.hyper_tuning import run_optuna_tuning, run_random_search
from src.advanced_model import AdvancedAlzheimerModel, load_real_data
from src.data_fusion import DataFusion
from src.validation import RobustValidator

# Set thread limits for stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'


def build_xgboost_model(trial):
    """Build XGBoost model with Optuna-suggested parameters"""
    import xgboost as xgb
    
    return xgb.XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 300, step=25),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
        n_jobs=1
    )


def build_random_forest_model(trial):
    """Build Random Forest model with Optuna-suggested parameters"""
    from sklearn.ensemble import RandomForestClassifier
    
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 300, step=25),
        max_depth=trial.suggest_int("max_depth", 4, 15),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        random_state=42,
        n_jobs=1
    )


def build_lightgbm_model(trial):
    """Build LightGBM model with Optuna-suggested parameters"""
    import lightgbm as lgb
    
    return lgb.LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 300, step=25),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0),
        num_leaves=trial.suggest_int("num_leaves", 10, 100),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        random_state=42,
        verbose=-1,
        n_jobs=1
    )


def build_svm_model(trial):
    """Build SVM model with Optuna-suggested parameters"""
    from sklearn.svm import SVC
    
    return SVC(
        C=trial.suggest_float("C", 0.01, 100, log=True),
        gamma=trial.suggest_categorical("gamma", ["scale", "auto", 0.001, 0.01, 0.1, 1.0]),
        kernel=trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        degree=trial.suggest_int("degree", 2, 5) if trial.params.get("kernel") == "poly" else 3,
        probability=True,
        random_state=42
    )


def build_logistic_regression_model(trial):
    """Build Logistic Regression model with Optuna-suggested parameters"""
    from sklearn.linear_model import LogisticRegression
    
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    solver = "saga" if penalty in ["l1", "elasticnet"] else "liblinear"
    
    return LogisticRegression(
        C=trial.suggest_float("C", 0.001, 10, log=True),
        penalty=penalty,
        solver=solver,
        l1_ratio=trial.suggest_float("l1_ratio", 0.1, 0.9) if penalty == "elasticnet" else None,
        max_iter=500,
        random_state=42,
        n_jobs=1
    )


def build_mlp_model(trial):
    """Build MLP model with Optuna-suggested parameters"""
    from sklearn.neural_network import MLPClassifier
    
    hidden_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = []
    for i in range(hidden_layers):
        hidden_sizes.append(trial.suggest_int(f"layer_{i}_size", 20, 200))
    
    return MLPClassifier(
        hidden_layer_sizes=tuple(hidden_sizes),
        activation=trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
        solver=trial.suggest_categorical("solver", ["adam", "lbfgs"]),
        alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        learning_rate=trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )


def main():
    """Main hyperparameter tuning pipeline"""
    
    print("🔧 Hyperparameter Tuning for Alzheimer's Disease Prediction")
    print("=" * 60)
    print("🚀 Features:")
    print("  - Optuna intelligent search with pruning")
    print("  - Multiple model architectures")
    print("  - Cross-validation with early stopping")
    print("  - Timeout protection (10 minutes per model)")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    X, y = load_real_data()
    print(f"✅ Data loaded: X {X.shape}, y {y.shape}")
    
    # Create fresh splits
    print("\n🔄 Creating fresh train/test splits...")
    validator = RobustValidator(random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = validator.create_fresh_splits(
        X, y, test_size=0.2, val_size=0.2
    )
    
    # Clean data
    def clean_data(X):
        X = np.array(X, dtype=np.float64)
        X = np.where(np.isinf(X), np.nan, X)
        X = np.where(np.abs(X) > 1e10, np.nan, X)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        return X
    
    X_train = clean_data(X_train)
    X_val = clean_data(X_val)
    X_test = clean_data(X_test)
    
    # Models to tune
    models_to_tune = {
        'XGBoost': build_xgboost_model,
        'Random Forest': build_random_forest_model,
        'LightGBM': build_lightgbm_model,
        'SVM': build_svm_model,
        'Logistic Regression': build_logistic_regression_model,
        'MLP': build_mlp_model
    }
    
    # Results storage
    tuning_results = {}
    
    # Tune each model
    for model_name, model_builder in models_to_tune.items():
        print(f"\n🎯 Tuning {model_name}...")
        print("-" * 40)
        
        try:
            # Use Optuna tuning
            best_model, tuning_info = run_optuna_tuning(
                model_builder=model_builder,
                model_name=model_name,
                X=X_train,
                y=y_train,
                n_trials=30,
                timeout=600,  # 10 minutes timeout
                cv_folds=3,
                n_jobs=1,
                enable_pruning=True,
                verbose=True
            )
            
            if best_model is not None:
                # Evaluate on validation set
                best_model.fit(X_train, y_train)
                val_score = best_model.score(X_val, y_val)
                test_score = best_model.score(X_test, y_test)
                
                tuning_results[model_name] = {
                    'model': best_model,
                    'best_params': tuning_info['best_params'],
                    'best_cv_score': tuning_info['best_value'],
                    'val_score': val_score,
                    'test_score': test_score,
                    'n_trials': tuning_info['n_trials'],
                    'pruned_trials': tuning_info['pruned']
                }
                
                print(f"✅ {model_name} tuning completed:")
                print(f"  Best CV Score: {tuning_info['best_value']:.4f}")
                print(f"  Validation Score: {val_score:.4f}")
                print(f"  Test Score: {test_score:.4f}")
                print(f"  Trials: {tuning_info['n_trials']} (Pruned: {tuning_info['pruned']})")
                print(f"  Best Params: {tuning_info['best_params']}")
                
            else:
                print(f"❌ {model_name} tuning failed")
                
        except Exception as e:
            print(f"❌ {model_name} tuning error: {e}")
            continue
    
    # Find best model
    if tuning_results:
        best_model_name = max(tuning_results.keys(), 
                            key=lambda k: tuning_results[k]['val_score'])
        best_model_info = tuning_results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"📊 Validation Score: {best_model_info['val_score']:.4f}")
        print(f"📊 Test Score: {best_model_info['test_score']:.4f}")
        print(f"📊 Best Parameters: {best_model_info['best_params']}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save tuning summary
        summary_data = []
        for name, info in tuning_results.items():
            summary_data.append({
                'Model': name,
                'Val_Score': info['val_score'],
                'Test_Score': info['test_score'],
                'CV_Score': info['best_cv_score'],
                'Trials': info['n_trials'],
                'Pruned': info['pruned']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Val_Score', ascending=False)
        
        csv_path = os.path.join(results_dir, f'hyperparameter_tuning_summary_{timestamp}.csv')
        summary_df.to_csv(csv_path, index=False)
        
        # Save best model
        import joblib
        model_path = os.path.join(results_dir, f'best_tuned_model_{best_model_name.replace(" ", "_")}_{timestamp}.pkl')
        joblib.dump(best_model_info['model'], model_path)
        
        print(f"\n💾 Results saved:")
        print(f"  Summary: {csv_path}")
        print(f"  Best Model: {model_path}")
        
    else:
        print("\n❌ No models were successfully tuned")


if __name__ == '__main__':
    main()
