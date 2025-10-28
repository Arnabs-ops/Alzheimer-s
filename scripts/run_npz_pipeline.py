import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'SVM': SVC(random_state=42, probability=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        accuracy = model.score(X_test, y_test)
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean = None
            cv_std = None
        results[name] = {
            'model': model,
            'accuracy': float(accuracy),
            'auc': None,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    return results


def main():
    npz_path = os.environ.get('NPZ_PATH', 'data/raw/preprocessed_alz_data.npz')
    results_dir = os.environ.get('RESULTS_DIR', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading NPZ from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"Keys: {keys}")

    def pick(keys_list, candidates):
        for cand in candidates:
            for k in keys_list:
                if k.lower() == cand.lower():
                    return k
        return None

    X_key = pick(keys, ["X", "x", "features", "X_train"]) or keys[0]
    y_key = pick(keys, ["y", "labels", "target", "y_train"]) or keys[1 if len(keys) > 1 else keys[0]]

    X = data[X_key]
    y = data[y_key]

    print(f"Using keys X='{X_key}', y='{y_key}' -> X{X.shape}, y{y.shape}")

    # Handle missing values and infinity
    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    
    # Replace infinity and very large values with NaN
    X = np.where(np.isinf(X), np.nan, X)
    X = np.where(np.abs(X) > 1e10, np.nan, X)
    
    # Fill NaN values with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Handle multi-class target
    if len(y.shape) > 1:
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            # Multi-class: convert to single class (argmax)
            y = np.argmax(y, axis=1)
    
    print(f"After preprocessing -> X{X.shape}, y{y.shape}")

    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'AUC': res['auc'],
            'CV_Mean': res['cv_mean'],
            'CV_Std': res['cv_std']
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('Accuracy', ascending=False)
    csv_path = os.path.join(results_dir, f'npz_model_summary_{timestamp}.csv')
    summary_df.to_csv(csv_path, index=False)

    json_path = os.path.join(results_dir, f'npz_all_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({k: {m: res[m] for m in ['accuracy', 'auc', 'cv_mean', 'cv_std']} for k, res in results.items()}, f, indent=2)

    best_name = summary_df.iloc[0]['Model']
    best_model = results[best_name]['model']
    model_path = os.path.join(results_dir, f'npz_best_model_{best_name}_{timestamp}.pkl')
    joblib.dump(best_model, model_path)

    print(f"Saved summary: {csv_path}")
    print(f"Saved results: {json_path}")
    print(f"Saved best model: {model_path}")


if __name__ == '__main__':
    main()


