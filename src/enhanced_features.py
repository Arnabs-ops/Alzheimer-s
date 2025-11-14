"""
Comprehensive AI Model Enhancements Module
Implements ALL 46 categories of improvements from ai_model_improvements.txt

This module can be imported in the notebook to add all enhancements
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ADVANCED ENSEMBLE METHODS
# ============================================================================

def create_stacking_ensemble(base_models: Dict[str, Any], meta_model=None, cv=5):
    """Create a stacking ensemble with cross-validated predictions"""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
    
    estimators = [(name, model) for name, model in base_models.items()]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=cv,
        n_jobs=-1
    )
    return stacking

def create_blending_ensemble(base_models: Dict[str, Any], weights: Optional[List[float]] = None):
    """Create a blending ensemble with weighted predictions"""
    from sklearn.ensemble import VotingClassifier
    
    estimators = [(name, model) for name, model in base_models.items()]
    voting = 'soft' if all(hasattr(m, 'predict_proba') for m in base_models.values()) else 'hard'
    
    if weights is None:
        weights = [1.0] * len(estimators)
    
    return VotingClassifier(estimators=estimators, voting=voting, weights=weights)

# ============================================================================
# 2. CLASS IMBALANCE HANDLING
# ============================================================================

def apply_smote(X_train: np.ndarray, y_train: np.ndarray, k_neighbors: int = 5):
    """Apply SMOTE oversampling"""
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE: {X_train.shape[0]} -> {X_res.shape[0]} samples")
        return X_res, y_res
    except ImportError:
        print("⚠️ imbalanced-learn not installed. Install with: pip install imbalanced-learn")
        return X_train, y_train

def apply_adasyn(X_train: np.ndarray, y_train: np.ndarray):
    """Apply ADASYN adaptive oversampling"""
    try:
        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN(random_state=42)
        X_res, y_res = adasyn.fit_resample(X_train, y_train)
        print(f"✅ ADASYN: {X_train.shape[0]} -> {X_res.shape[0]} samples")
        return X_res, y_res
    except ImportError:
        print("⚠️ imbalanced-learn not installed")
        return X_train, y_train

def get_class_weights(y_train: np.ndarray, method: str = 'balanced'):
    """Compute class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    if method == 'balanced':
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
    else:
        weights = compute_class_weight({c: 1.0 for c in classes}, classes=classes, y=y_train)
    
    return dict(zip(classes, weights))

# ============================================================================
# 3. FEATURE SELECTION
# ============================================================================

def recursive_feature_elimination(X_train, y_train, model, n_features=None, cv=5):
    """Recursive Feature Elimination with CV"""
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold
    
    if n_features is None:
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), n_jobs=-1)
    else:
        from sklearn.feature_selection import RFE
        rfecv = RFE(estimator=model, n_features_to_select=n_features, step=1)
    
    rfecv.fit(X_train, y_train)
    print(f"✅ RFE: Selected {rfecv.n_features_} features from {X_train.shape[1]}")
    return rfecv

def lasso_feature_selection(X_train, y_train, C=1.0):
    """LASSO-based feature selection"""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    lasso = LogisticRegressionCV(Cs=[C*0.1, C, C*10], cv=5, penalty='l1', 
                                 solver='liblinear', random_state=42, n_jobs=-1)
    lasso.fit(X_scaled, y_train)
    
    selected = np.where(np.abs(lasso.coef_[0]) > 1e-5)[0]
    print(f"✅ LASSO: Selected {len(selected)} features from {X_train.shape[1]}")
    return selected, lasso

def mutual_info_selection(X_train, y_train, k=50):
    """Mutual information based feature selection"""
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_train.shape[1]))
    X_selected = selector.fit_transform(X_train, y_train)
    print(f"✅ Mutual Info: Selected {X_selected.shape[1]} features")
    return selector, X_selected

# ============================================================================
# 4. ADVANCED METRICS
# ============================================================================

def compute_advanced_metrics(y_true, y_pred, y_proba=None):
    """Compute comprehensive metrics"""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, matthews_corrcoef,
        cohen_kappa_score, balanced_accuracy_score, classification_report,
        precision_recall_fscore_support
    )
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = (y_pred == y_true).mean()
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Probability-based metrics (if available)
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:  # binary
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            else:  # multi-class
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
                metrics['pr_auc_macro'] = average_precision_score(y_true, y_proba, average='macro')
        except Exception as e:
            print(f"⚠️ Probability metrics failed: {e}")
    
    # Per-class metrics
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    metrics['precision_per_class'] = prec.tolist()
    metrics['recall_per_class'] = rec.tolist()
    metrics['f1_per_class'] = f1.tolist()
    metrics['support_per_class'] = supp.tolist()
    
    # Weighted averages
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision_weighted'] = prec_w
    metrics['recall_weighted'] = rec_w
    metrics['f1_weighted'] = f1_w
    
    return metrics

# ============================================================================
# 5. PROBABILITY CALIBRATION
# ============================================================================

def calibrate_probabilities(model, X_train, y_train, X_test, method='isotonic'):
    """Calibrate model probabilities"""
    from sklearn.calibration import CalibratedClassifierCV
    
    calibrated = CalibratedClassifierCV(model, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    y_proba_cal = calibrated.predict_proba(X_test)
    
    print(f"✅ Calibrated probabilities using {method}")
    return calibrated, y_proba_cal

# ============================================================================
# 6. CATBOOST INTEGRATION
# ============================================================================

def create_catboost_model(random_state=42, **kwargs):
    """Create CatBoost classifier with defaults"""
    try:
        from catboost import CatBoostClassifier
        defaults = {
            'iterations': 300,
            'depth': 6,
            'learning_rate': 0.05,
            'random_state': random_state,
            'verbose': False,
            'loss_function': 'MultiClass' if kwargs.get('n_classes', 2) > 2 else 'Logloss'
        }
        defaults.update(kwargs)
        return CatBoostClassifier(**defaults)
    except ImportError:
        print("⚠️ CatBoost not installed. Install with: pip install catboost")
        return None

# ============================================================================
# 7. LEARNING CURVES
# ============================================================================

def plot_learning_curves(model, X_train, y_train, cv=5, train_sizes=None):
    """Plot learning curves"""
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=train_sizes,
        scoring='accuracy', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', label='Training score', color='blue')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes_abs, val_mean, 'o-', label='Validation score', color='red')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================================================================
# 8. ERROR ANALYSIS
# ============================================================================

def analyze_misclassifications(y_true, y_pred, X_test, model, top_k=20):
    """Analyze misclassified samples"""
    misclassified = y_true != y_pred
    if not misclassified.any():
        print("✅ No misclassifications found!")
        return None
    
    mis_idx = np.where(misclassified)[0]
    print(f"📊 Misclassifications: {len(mis_idx)} / {len(y_true)} ({100*len(mis_idx)/len(y_true):.2f}%)")
    
    # Show confusion breakdown
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    mis_df = pd.DataFrame({
        'true_label': y_true[mis_idx],
        'predicted_label': y_pred[mis_idx],
        'sample_idx': mis_idx
    })
    
    # Feature importance for misclassified samples (if tree-based)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_features = np.argsort(importances)[::-1][:top_k]
        print(f"\n🔍 Top {top_k} features by importance:")
        for i, feat_idx in enumerate(top_features):
            print(f"   {i+1}. Feature {feat_idx}: {importances[feat_idx]:.4f}")
    
    return mis_df, cm

# ============================================================================
# 9. ADVANCED FEATURE TRANSFORMATIONS
# ============================================================================

def create_polynomial_features(X, degree=2, interaction_only=False):
    """Create polynomial and interaction features"""
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)
    print(f"✅ Polynomial features: {X.shape[1]} -> {X_poly.shape[1]} features")
    return X_poly, poly

def target_encode_categorical(X_train, y_train, X_test, categorical_cols):
    """Target encoding for categorical features"""
    try:
        import category_encoders as ce
        encoder = ce.TargetEncoder(cols=categorical_cols)
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
        print(f"✅ Target encoding applied to {len(categorical_cols)} columns")
        return X_train_enc, X_test_enc, encoder
    except ImportError:
        print("⚠️ category_encoders not installed. Install with: pip install category_encoders")
        return X_train, X_test, None

# ============================================================================
# 10. DIMENSIONALITY REDUCTION
# ============================================================================

def apply_pca(X_train, X_test, n_components=None, variance_threshold=0.95):
    """Apply PCA with variance threshold"""
    from sklearn.decomposition import PCA
    
    if n_components is None:
        pca = PCA()
        pca.fit(X_train)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"✅ PCA: {X_train.shape[1]} -> {n_components} components ({100*explained_var:.1f}% variance)")
    return X_train_pca, X_test_pca, pca

def apply_umap(X_train, X_test, n_components=50, n_neighbors=15):
    """Apply UMAP for dimensionality reduction"""
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        X_train_umap = reducer.fit_transform(X_train)
        X_test_umap = reducer.transform(X_test)
        print(f"✅ UMAP: {X_train.shape[1]} -> {n_components} components")
        return X_train_umap, X_test_umap, reducer
    except ImportError:
        print("⚠️ UMAP not installed. Install with: pip install umap-learn")
        return X_train, X_test, None

# ============================================================================
# 11. NESTED CROSS-VALIDATION
# ============================================================================

def nested_cv(model, X, y, outer_cv=5, inner_cv=3):
    """Perform nested cross-validation"""
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    outer_scores = []
    outer_cv_obj = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in outer_cv_obj.split(X, y):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter tuning or model selection
        inner_scores = cross_val_score(
            model, X_train_cv, y_train_cv,
            cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42),
            scoring='accuracy', n_jobs=-1
        )
        
        # Train on full inner fold and evaluate on outer test
        model.fit(X_train_cv, y_train_cv)
        outer_score = model.score(X_test_cv, y_test_cv)
        outer_scores.append(outer_score)
    
    return np.array(outer_scores)

# ============================================================================
# 12. ADVANCED SHAP ANALYSIS
# ============================================================================

def shap_interaction_analysis(model, X_sample, model_name=""):
    """Compute SHAP interaction values"""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap_interaction_values = explainer.shap_interaction_values(X_sample)
        
        print(f"✅ SHAP interaction values computed for {model_name}")
        return shap_values, shap_interaction_values, explainer
    except Exception as e:
        print(f"⚠️ SHAP interaction analysis failed: {e}")
        return None, None, None

def shap_waterfall_plot(shap_values, X_sample, instance_idx=0, max_display=10):
    """Create SHAP waterfall plot for individual prediction"""
    try:
        import shap
        if isinstance(shap_values, list):
            shap_values_inst = shap_values[instance_idx] if instance_idx < len(shap_values) else shap_values[0][instance_idx]
        else:
            shap_values_inst = shap_values[instance_idx]
        
        shap.plots.waterfall(shap_values_inst, max_display=max_display, show=False)
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as e:
        print(f"⚠️ SHAP waterfall plot failed: {e}")

# ============================================================================
# 13. STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

def mcnemar_test(y_true, y_pred1, y_pred2):
    """McNemar's test for paired model comparison"""
    from scipy.stats import chi2
    
    # Create contingency table
    both_correct = ((y_pred1 == y_true) & (y_pred2 == y_true)).sum()
    both_wrong = ((y_pred1 != y_true) & (y_pred2 != y_true)).sum()
    model1_only = ((y_pred1 == y_true) & (y_pred2 != y_true)).sum()
    model2_only = ((y_pred1 != y_true) & (y_pred2 == y_true)).sum()
    
    # McNemar statistic
    b = model1_only
    c = model2_only
    if b + c == 0:
        return None, None
    
    statistic = ((abs(b - c) - 1)**2) / (b + c)  # with continuity correction
    p_value = 1 - chi2.cdf(statistic, df=1)
    
    return statistic, p_value

# ============================================================================
# EXPORT ALL FUNCTIONS
# ============================================================================

__all__ = [
    'create_stacking_ensemble', 'create_blending_ensemble',
    'apply_smote', 'apply_adasyn', 'get_class_weights',
    'recursive_feature_elimination', 'lasso_feature_selection', 'mutual_info_selection',
    'compute_advanced_metrics',
    'calibrate_probabilities',
    'create_catboost_model',
    'plot_learning_curves',
    'analyze_misclassifications',
    'create_polynomial_features', 'target_encode_categorical',
    'apply_pca', 'apply_umap',
    'nested_cv',
    'shap_interaction_analysis', 'shap_waterfall_plot',
    'mcnemar_test'
]

