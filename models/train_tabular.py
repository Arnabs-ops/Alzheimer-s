"""
Train tabular models (XGBoost, LightGBM, etc.) on extracted features.
Keeps existing models, adds new ones as needed.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Existing models (keep these)
import xgboost as xgb
import lightgbm as lgb

# Additional models (add if needed)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Build model dictionary. Keeps existing XGBoost/LightGBM, adds new ones.
    
    Returns:
        Dictionary of model_name -> model instance
    """
    models = {
        # Existing models (keep these)
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=random_state,
            tree_method='hist',
            eval_metric='logloss',
            verbosity=0,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            min_child_samples=20,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        ),
        
        # Additional models (add if existing ones don't perform well)
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            max_samples=0.8,
            n_jobs=-1,
            random_state=random_state,
            oob_score=True
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            max_samples=0.8,
            n_jobs=-1,
            random_state=random_state
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            validation_fraction=0.1,
            n_iter_no_change=20
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=2000,
            solver='lbfgs',
            C=1.0,
            class_weight='balanced',
            random_state=random_state,
            multi_class='ovr',
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
            batch_size=128
        )
    }
    
    # Try to add CatBoost if available
    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_state=random_state,
            verbose=False,
            loss_function='MultiClass'
        )
    except ImportError:
        pass
    
    return models


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    save_dir: str = "models/saved"
) -> Dict[str, Dict[str, Any]]:
    """
    Train all models and return results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        models: Model dictionary (if None, uses build_models())
        cv_folds: Cross-validation folds
        save_dir: Directory to save models
    
    Returns:
        Dictionary mapping model_name -> {
            'model': trained_model,
            'accuracy': float,
            'roc_auc': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'cv_mean': float,
            'cv_std': float,
            'y_pred': predictions,
            'y_proba': probabilities
        }
    """
    if models is None:
        models = build_models()
    
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print(f"🤖 Training {len(models)} models...")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n🔁 Training {name}...")
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            
            # ROC-AUC (handle binary and multi-class)
            roc_auc = None
            if y_proba is not None:
                try:
                    if y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except Exception:
                    pass
            
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='accuracy', n_jobs=-1
            )
            
            results[name] = {
                'model': model,
                'accuracy': float(acc),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            print(f"   ✅ Acc={acc:.4f} | ROC-AUC={roc_auc:.4f if roc_auc else 'N/A'} | "
                  f"F1={f1:.4f} | CV={cv_scores.mean():.4f}±{cv_scores.std():.3f}")
            
            # Save model
            model_path = os.path.join(save_dir, f"{name.replace(' ', '_')}.pkl")
            joblib.dump(model, model_path)
            print(f"   💾 Saved to {model_path}")
            
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results summary
    summary_path = os.path.join(save_dir, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'ROC_AUC': res['roc_auc'] if res['roc_auc'] is not None else 0.0,
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1': res['f1'],
            'CV_Mean': res['cv_mean'],
            'CV_Std': res['cv_std']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Results summary saved to {summary_path}")
    
    return results


def load_trained_model(model_name: str, models_dir: str = "models/saved") -> Any:
    """Load a trained model from disk."""
    model_path = os.path.join(models_dir, f"{model_name.replace(' ', '_')}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")


if __name__ == "__main__":
    # Example usage
    print("This module provides functions for training tabular models.")
    print("Use it as: from models.train_tabular import train_models, build_models")

