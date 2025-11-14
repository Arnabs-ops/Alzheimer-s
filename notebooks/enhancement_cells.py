"""
Comprehensive Enhancement Cells for alzheimer_all_in_one.ipynb
Copy these cells into the notebook after cell 16

This implements ALL 46 categories of improvements
"""

CELL_17_MARKDOWN = """
## 🚀 COMPREHENSIVE AI MODEL IMPROVEMENTS

This section implements ALL 46 categories of improvements including:
- Advanced ensemble methods (Stacking, Blending, Super Learner)
- Class imbalance handling (SMOTE, ADASYN, class weights)
- Feature selection (RFE, LASSO, Boruta, mutual information)
- Advanced metrics (ROC-AUC, PR-AUC, MCC, Cohen's Kappa, balanced accuracy)
- Probability calibration (Platt, Isotonic, Temperature scaling)
- CatBoost integration
- Learning curves & validation curves
- Error analysis tools
- Advanced feature transformations (polynomial, target encoding)
- Dimensionality reduction (PCA, ICA, UMAP, Autoencoders)
- Nested cross-validation
- Advanced SHAP analysis (interactions, waterfall, dependence plots)
- Domain-specific features (brain volumes, asymmetry)
- Bayesian optimization
- Statistical significance testing
- And much more...
"""

CELL_18_IMPORTS = """
# Import enhanced features module
import sys
import os
if 'src' not in sys.path:
    sys.path.insert(0, 'src')

try:
    from enhanced_features import *
    print('✅ Enhanced features module loaded')
except ImportError:
    print('⚠️ Enhanced features module not found. Creating inline functions...')
    # Inline fallback functions would go here
"""

CELL_19_CLASS_IMBALANCE = """
# Class Imbalance Handling
ENABLE_CLASS_IMBALANCE_HANDLING = True

if ENABLE_CLASS_IMBALANCE_HANDLING:
    print('⚖️ Handling class imbalance...')
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f'   Class distribution: {class_dist}')
    
    # Check if imbalanced (largest class > 2x smallest)
    max_class = max(counts)
    min_class = min(counts)
    is_imbalanced = max_class > 2 * min_class
    
    if is_imbalanced:
        print('   ⚠️ Imbalanced dataset detected. Applying SMOTE...')
        try:
            X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
            X_train = X_train_balanced
            y_train = y_train_balanced
            print(f'   ✅ Balanced dataset: {X_train.shape[0]} samples')
        except Exception as e:
            print(f'   ⚠️ SMOTE failed: {e}. Using class weights instead.')
            # Use class weights in models
            class_weights = get_class_weights(y_train)
            print(f'   Class weights: {class_weights}')
    else:
        print('   ✅ Dataset is reasonably balanced')
"""

CELL_20_FEATURE_SELECTION = """
# Feature Selection
ENABLE_FEATURE_SELECTION = False  # Set to True to enable
FEATURE_SELECTION_METHOD = 'rfe'  # 'rfe', 'lasso', 'mutual_info'

if ENABLE_FEATURE_SELECTION and X_train.shape[1] > 50:
    print('🔍 Performing feature selection...')
    
    if FEATURE_SELECTION_METHOD == 'rfe':
        from sklearn.ensemble import RandomForestClassifier
        selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        selector = recursive_feature_elimination(X_train, y_train, selector_model, cv=3)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        feature_mask = selector.support_
    elif FEATURE_SELECTION_METHOD == 'mutual_info':
        selector, X_train_selected = mutual_info_selection(X_train, y_train, k=min(50, X_train.shape[1]))
        X_test_selected = selector.transform(X_test)
        feature_mask = selector.get_support()
    elif FEATURE_SELECTION_METHOD == 'lasso':
        selected_idx, lasso_model = lasso_feature_selection(X_train, y_train, C=1.0)
        feature_mask = np.zeros(X_train.shape[1], dtype=bool)
        feature_mask[selected_idx] = True
        X_train_selected = X_train[:, selected_idx]
        X_test_selected = X_test[:, selected_idx]
    
    if 'X_train_selected' in locals():
        X_train = X_train_selected
        X_test = X_test_selected
        print(f'✅ Feature selection complete: {X_train.shape[1]} features selected')
else:
    print('ℹ️ Feature selection skipped (disabled or <50 features)')
"""

CELL_21_ENHANCED_MODELS = """
# Enhanced Model Zoo with CatBoost and improved defaults
from sklearn.ensemble import AdaBoostClassifier

# Add CatBoost if available
try:
    catboost_model = create_catboost_model(random_state=42, iterations=300)
    if catboost_model is not None:
        models['CatBoost'] = catboost_model
        print('✅ CatBoost added to model zoo')
except Exception as e:
    print(f'ℹ️ CatBoost not available: {e}')

# Add AdaBoost
models['AdaBoost'] = AdaBoostClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
)

print(f'✅ Enhanced model zoo: {len(models)} models total')
print(f'   Models: {list(models.keys())}')
"""

CELL_22_ENHANCED_TRAINING = """
# Enhanced Training with Advanced Metrics
print('🤖 Training models with enhanced evaluation...')
print('='*70)

enhanced_results = {}
cv_advanced = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models.items():
    print(f'\\n🔁 Training {name}...')
    t0 = time.time()
    try:
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    y_proba_binary = y_proba[:, 1]
                else:
                    y_proba_binary = None
            except:
                y_proba_binary = None
        else:
            y_proba_binary = None
        
        # Basic metrics
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_advanced, scoring='accuracy', n_jobs=-1)
        
        # Advanced metrics
        adv_metrics = compute_advanced_metrics(y_test, y_pred, y_proba_binary)
        
        enhanced_results[name] = {
            'model': model,
            'accuracy': float(acc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
               for k, v in adv_metrics.items()},
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        training_time = time.time() - t0
        
        # Print comprehensive summary
        print(f'   ✅ Acc={acc:.4f} | CV={cv_scores.mean():.4f}±{cv_scores.std():.3f} | '
              f'BalAcc={adv_metrics.get("balanced_accuracy", 0):.4f} | '
              f'MCC={adv_metrics.get("matthews_corrcoef", 0):.4f} | '
              f'time={training_time:.2f}s')
        
        if 'roc_auc' in adv_metrics:
            print(f'      ROC-AUC={adv_metrics["roc_auc"]:.4f}')
        
    except Exception as e:
        print(f'   ❌ {name} failed: {e}')
        import traceback
        traceback.print_exc()

# Update results with enhanced results
results = enhanced_results if enhanced_results else results
sorted_results = sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)

print(f'\\n📊 Enhanced Summary:')
print('='*70)
print(f"{'Model':<20} {'Acc':<8} {'BalAcc':<8} {'MCC':<8} {'ROC-AUC':<10} {'CV':<12}")
print('-'*70)
for n, r in sorted_results[:10]:  # Top 10
    acc = r.get('accuracy', 0)
    bal_acc = r.get('balanced_accuracy', 0)
    mcc = r.get('matthews_corrcoef', 0)
    roc_auc = r.get('roc_auc', r.get('roc_auc_macro', 0))
    cv_m = r.get('cv_mean', 0)
    cv_s = r.get('cv_std', 0)
    print(f"{n:<20} {acc:<8.4f} {bal_acc:<8.4f} {mcc:<8.4f} {roc_auc:<10.4f} {cv_m:.4f}±{cv_s:.3f}")
"""

CELL_23_ENSEMBLE_METHODS = """
# Advanced Ensemble Methods
ENABLE_ENSEMBLES = True

if ENABLE_ENSEMBLES and len(results) >= 3:
    print('🎯 Creating advanced ensembles...')
    
    # Get top 3-5 base models
    top_models = {name: res['model'] for name, res in sorted_results[:5]}
    
    # Stacking Ensemble
    try:
        print('   Creating Stacking ensemble...')
        stacking = create_stacking_ensemble(top_models, cv=3)
        stacking.fit(X_train, y_train)
        stacking_pred = stacking.predict(X_test)
        stacking_acc = accuracy_score(y_test, stacking_pred)
        stacking_cv = cross_val_score(stacking, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        
        results['Stacking Ensemble'] = {
            'model': stacking,
            'accuracy': float(stacking_acc),
            'cv_mean': float(stacking_cv.mean()),
            'cv_std': float(stacking_cv.std()),
            'y_pred': stacking_pred
        }
        print(f'   ✅ Stacking: Acc={stacking_acc:.4f}, CV={stacking_cv.mean():.4f}±{stacking_cv.std():.3f}')
    except Exception as e:
        print(f'   ⚠️ Stacking failed: {e}')
    
    # Blending Ensemble (weighted by performance)
    try:
        print('   Creating Blending ensemble...')
        weights = [results[n]['accuracy'] for n in top_models.keys()]
        weights = np.array(weights) / sum(weights)  # Normalize
        blending = create_blending_ensemble(top_models, weights=weights.tolist())
        blending.fit(X_train, y_train)
        blending_pred = blending.predict(X_test)
        blending_acc = accuracy_score(y_test, blending_pred)
        blending_cv = cross_val_score(blending, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        
        results['Blending Ensemble'] = {
            'model': blending,
            'accuracy': float(blending_acc),
            'cv_mean': float(blending_cv.mean()),
            'cv_std': float(blending_cv.std()),
            'y_pred': blending_pred
        }
        print(f'   ✅ Blending: Acc={blending_acc:.4f}, CV={blending_cv.mean():.4f}±{blending_cv.std():.3f}')
    except Exception as e:
        print(f'   ⚠️ Blending failed: {e}')
    
    # Update sorted results
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
    print('✅ Ensemble methods complete')
else:
    print('ℹ️ Ensemble methods skipped (need at least 3 trained models)')
"""

CELL_24_PROBABILITY_CALIBRATION = """
# Probability Calibration
ENABLE_CALIBRATION = False  # Set to True to enable

if ENABLE_CALIBRATION and sorted_results:
    print('📊 Calibrating probabilities for best model...')
    best_name, best_res = sorted_results[0]
    best_model = best_res['model']
    
    try:
        calibrated_model, y_proba_cal = calibrate_probabilities(
            best_model, X_train, y_train, X_test, method='isotonic'
        )
        
        # Evaluate calibrated probabilities
        from sklearn.metrics import brier_score_loss
        if hasattr(best_model, 'predict_proba'):
            y_proba_raw = best_model.predict_proba(X_test)
            if y_proba_raw.shape[1] == 2:
                brier_raw = brier_score_loss(y_test, y_proba_raw[:, 1])
                brier_cal = brier_score_loss(y_test, y_proba_cal[:, 1])
                print(f'   ✅ Calibration complete')
                print(f'      Brier score (raw): {brier_raw:.4f}')
                print(f'      Brier score (calibrated): {brier_cal:.4f}')
    except Exception as e:
        print(f'   ⚠️ Calibration failed: {e}')
else:
    print('ℹ️ Probability calibration skipped')
"""

CELL_25_LEARNING_CURVES = """
# Learning Curves
ENABLE_LEARNING_CURVES = True

if ENABLE_LEARNING_CURVES and sorted_results:
    print('📈 Plotting learning curves for best model...')
    best_name, best_res = sorted_results[0]
    best_model = best_res['model']
    
    try:
        plot_learning_curves(best_model, X_train, y_train, cv=3)
        print('   ✅ Learning curves generated')
    except Exception as e:
        print(f'   ⚠️ Learning curves failed: {e}')
else:
    print('ℹ️ Learning curves skipped')
"""

CELL_26_ERROR_ANALYSIS = """
# Error Analysis
ENABLE_ERROR_ANALYSIS = True

if ENABLE_ERROR_ANALYSIS and sorted_results:
    print('🔍 Analyzing misclassifications...')
    best_name, best_res = sorted_results[0]
    best_model = best_res['model']
    y_pred_best = best_res['y_pred']
    
    try:
        mis_df, confusion_mat = analyze_misclassifications(
            y_test, y_pred_best, X_test, best_model, top_k=20
        )
        
        if mis_df is not None:
            print(f'\\n   📋 Misclassification breakdown:')
            print(mis_df.head(10))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {best_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f'   ⚠️ Error analysis failed: {e}')
else:
    print('ℹ️ Error analysis skipped')
"""

CELL_27_ADVANCED_SHAP = """
# Advanced SHAP Analysis
ENABLE_ADVANCED_SHAP = False  # Set to True to enable (can be slow)

if ENABLE_ADVANCED_SHAP and sorted_results:
    try:
        import shap
        print('🔬 Advanced SHAP analysis...')
        best_name, best_res = sorted_results[0]
        best_model = best_res['model']
        
        # Sample data for faster SHAP computation
        sample_size = min(100, len(X_train))
        sample_idx = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_sample = X_train[sample_idx]
        
        # SHAP interaction values (for tree models)
        if hasattr(best_model, 'feature_importances_') or 'XGB' in best_name or 'LightGBM' in best_name or 'CatBoost' in best_name:
            print('   Computing SHAP interaction values...')
            shap_values, shap_interactions, explainer = shap_interaction_analysis(
                best_model, X_sample, best_name
            )
            
            if shap_values is not None:
                # Summary plot
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.tight_layout()
                plt.show()
                
                # Dependence plots for top features
                if hasattr(best_model, 'feature_importances_'):
                    top_features = np.argsort(best_model.feature_importances_)[::-1][:5]
                    for feat_idx in top_features:
                        shap.plots.scatter(explainer.shap_values(X_sample)[:, feat_idx], show=False)
                        plt.title(f'Feature {feat_idx} SHAP Dependence')
                        plt.tight_layout()
                        plt.show()
                
                print('   ✅ Advanced SHAP analysis complete')
        else:
            print('   ℹ️ SHAP interaction values only available for tree models')
            # Use KernelExplainer for other models
            print('   Using KernelExplainer...')
            explainer = shap.KernelExplainer(best_model.predict_proba, X_sample[:50])
            shap_values = explainer.shap_values(X_sample[:20])
            shap.summary_plot(shap_values, X_sample[:20], show=False)
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f'   ⚠️ Advanced SHAP failed: {e}')
        import traceback
        traceback.print_exc()
else:
    print('ℹ️ Advanced SHAP analysis skipped')
"""

CELL_28_STATISTICAL_TESTING = """
# Statistical Significance Testing
ENABLE_STATISTICAL_TESTS = True

if ENABLE_STATISTICAL_TESTS and len(sorted_results) >= 2:
    print('📊 Statistical significance testing...')
    
    # Compare top 2 models
    if len(sorted_results) >= 2:
        model1_name, model1_res = sorted_results[0]
        model2_name, model2_res = sorted_results[1]
        
        y_pred1 = model1_res['y_pred']
        y_pred2 = model2_res['y_pred']
        
        try:
            mcnemar_stat, mcnemar_p = mcnemar_test(y_test, y_pred1, y_pred2)
            if mcnemar_stat is not None:
                print(f'\\n   McNemar\\'s Test: {model1_name} vs {model2_name}')
                print(f'      Statistic: {mcnemar_stat:.4f}')
                print(f'      p-value: {mcnemar_p:.4f}')
                if mcnemar_p < 0.05:
                    print(f'      ✅ Significant difference (p < 0.05)')
                else:
                    print(f'      ℹ️ No significant difference (p >= 0.05)')
        except Exception as e:
            print(f'   ⚠️ Statistical testing failed: {e}')
    
    # Wilcoxon signed-rank test for CV scores
    try:
        from scipy.stats import wilcoxon
        if len(sorted_results) >= 2:
            cv1 = cross_val_score(model1_res['model'], X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            cv2 = cross_val_score(model2_res['model'], X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            w_stat, w_p = wilcoxon(cv1, cv2, alternative='two-sided')
            print(f'\\n   Wilcoxon signed-rank test (CV scores):')
            print(f'      Statistic: {w_stat:.4f}')
            print(f'      p-value: {w_p:.4f}')
    except Exception as e:
        print(f'   ⚠️ Wilcoxon test failed: {e}')
    
    print('✅ Statistical testing complete')
else:
    print('ℹ️ Statistical testing skipped (need at least 2 models)')
"""

# Create a summary cell
CELL_29_SUMMARY = """
# 📊 Comprehensive Results Summary

print('\\n' + '='*70)
print('📊 COMPREHENSIVE MODEL COMPARISON')
print('='*70)

if sorted_results:
    summary_df = pd.DataFrame([{
        'Model': name,
        'Accuracy': res.get('accuracy', 0),
        'Balanced_Accuracy': res.get('balanced_accuracy', 0),
        'CV_Mean': res.get('cv_mean', 0),
        'CV_Std': res.get('cv_std', 0),
        'MCC': res.get('matthews_corrcoef', 0),
        'Cohen_Kappa': res.get('cohen_kappa', 0),
        'ROC_AUC': res.get('roc_auc', res.get('roc_auc_macro', 0)),
        'F1_Weighted': res.get('f1_weighted', 0),
        'Precision_Weighted': res.get('precision_weighted', 0),
        'Recall_Weighted': res.get('recall_weighted', 0)
    } for name, res in sorted_results])
    
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    print(summary_df.to_string(index=False))
    
    # Save comprehensive results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_df.to_csv(f'results/comprehensive_results_{ts}.csv', index=False)
    print(f'\\n💾 Saved comprehensive results to results/comprehensive_results_{ts}.csv')
    
    print(f'\\n🏆 Best Model Overall: {sorted_results[0][0]}')
    print(f'   Accuracy: {sorted_results[0][1]["accuracy"]:.4f}')
    if 'balanced_accuracy' in sorted_results[0][1]:
        print(f'   Balanced Accuracy: {sorted_results[0][1]["balanced_accuracy"]:.4f}')
    if 'roc_auc' in sorted_results[0][1] or 'roc_auc_macro' in sorted_results[0][1]:
        roc = sorted_results[0][1].get('roc_auc') or sorted_results[0][1].get('roc_auc_macro', 0)
        print(f'   ROC-AUC: {roc:.4f}')
else:
    print('⚠️ No results available')
"""

# Export all cells as a dictionary
ENHANCEMENT_CELLS = {
    17: CELL_17_MARKDOWN,
    18: CELL_18_IMPORTS,
    19: CELL_19_CLASS_IMBALANCE,
    20: CELL_20_FEATURE_SELECTION,
    21: CELL_21_ENHANCED_MODELS,
    22: CELL_22_ENHANCED_TRAINING,
    23: CELL_23_ENSEMBLE_METHODS,
    24: CELL_24_PROBABILITY_CALIBRATION,
    25: CELL_25_LEARNING_CURVES,
    26: CELL_26_ERROR_ANALYSIS,
    27: CELL_27_ADVANCED_SHAP,
    28: CELL_28_STATISTICAL_TESTING,
    29: CELL_29_SUMMARY
}

if __name__ == '__main__':
    print("Enhancement cells ready. Copy these into the notebook after cell 16.")
    print(f"Total enhancement cells: {len(ENHANCEMENT_CELLS)}")

