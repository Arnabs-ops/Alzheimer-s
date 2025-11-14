"""
Evaluation module: compute metrics and generate SHAP visualizations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, classification_report, confusion_matrix,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }
    
    # ROC-AUC
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics['pr_auc'] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))
                metrics['pr_auc'] = float(average_precision_score(y_true, y_proba, average='weighted'))
        except Exception as e:
            print(f"⚠️ ROC-AUC computation failed: {e}")
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    
    return metrics


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluate a trained model and return comprehensive results.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_train: Training features (for CV)
        y_train: Training labels (for CV)
        model_name: Name of model
    
    Returns:
        Dictionary with predictions, probabilities, metrics, and CV scores
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    
    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    # Cross-validation (if train data provided)
    cv_scores = None
    if X_train is not None and y_train is not None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        metrics['cv_mean'] = float(cv_scores.mean())
        metrics['cv_std'] = float(cv_scores.std())
    
    return {
        'model_name': model_name,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': metrics,
        'cv_scores': cv_scores
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """Plot ROC curve."""
    if y_proba is None or y_proba.shape[1] < 2:
        print("⚠️ Cannot plot ROC curve: probabilities not available")
        return
    
    if y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        # Multi-class: use one-vs-rest
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        n_classes = y_proba.shape[1]
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all classes
        plt.figure(figsize=(8, 6))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} (Multi-class)')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """Plot Precision-Recall curve."""
    if y_proba is None or y_proba.shape[1] < 2:
        print("⚠️ Cannot plot PR curve: probabilities not available")
        return
    
    if y_proba.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        pr_auc = average_precision_score(y_true, y_proba[:, 1])
    else:
        # Multi-class: average PR curve
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(y_proba.shape[1]))
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
        pr_auc = average_precision_score(y_bin.ravel(), y_proba.ravel())
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_shap_summary(
    model: Any,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_name: str = "Model",
    max_display: int = 20,
    save_path: Optional[str] = None
):
    """
    Generate SHAP summary plots for model interpretability.
    
    Args:
        model: Trained model
        X_sample: Sample of features (for speed, use subset)
        feature_names: Optional feature names
        model_name: Model name
        max_display: Max features to display
        save_path: Optional path to save plot
    """
    try:
        import shap
        
        # Select appropriate explainer
        if hasattr(model, 'feature_importances_') or \
           'XGB' in model_name or 'LightGBM' in model_name or 'CatBoost' in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif 'Linear' in model_name or 'Logistic' in model_name:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Kernel explainer as fallback
            explainer = shap.KernelExplainer(model.predict_proba, X_sample[:min(50, len(X_sample))])
            shap_values = explainer.shap_values(X_sample[:min(100, len(X_sample))])
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=max_display, show=False)
        plt.title(f'SHAP Summary - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ SHAP analysis completed for {model_name}")
        
    except ImportError:
        print("⚠️ SHAP not installed. Install with: pip install shap")
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")


def generate_evaluation_report(
    results: Dict[str, Any],
    output_dir: str = "results"
):
    """
    Generate comprehensive evaluation report with all visualizations.
    
    Args:
        results: Dictionary from evaluate_model() or train_models()
        output_dir: Directory to save reports
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = results.get('model_name', 'Model')
    
    # Metrics summary
    metrics = results.get('metrics', {})
    print(f"\n📊 Evaluation Report - {model_name}")
    print("="*70)
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")
    if metrics.get('pr_auc'):
        print(f"PR-AUC:   {metrics.get('pr_auc', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:  {metrics.get('f1', 0):.4f}")
    if 'cv_mean' in metrics:
        print(f"CV Score:  {metrics['cv_mean']:.4f} ± {metrics.get('cv_std', 0):.3f}")
    
    # Classification report
    if 'y_pred' in results and 'y_test' in results:
        print("\n📋 Classification Report:")
        print(classification_report(results['y_test'], results['y_pred']))
    
    # Visualizations
    if 'y_proba' in results and results['y_proba'] is not None:
        plot_roc_curve(
            results.get('y_test', results.get('y_true')),
            results['y_proba'],
            model_name=model_name,
            save_path=os.path.join(output_dir, f"{model_name}_roc_curve.png")
        )
        
        plot_precision_recall_curve(
            results.get('y_test', results.get('y_true')),
            results['y_proba'],
            model_name=model_name,
            save_path=os.path.join(output_dir, f"{model_name}_pr_curve.png")
        )
    
    if 'y_pred' in results:
        plot_confusion_matrix(
            results.get('y_test', results.get('y_true')),
            results['y_pred'],
            model_name=model_name,
            save_path=os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
        )


if __name__ == "__main__":
    print("This module provides evaluation functions and visualizations.")
    print("Use it as: from models.evaluate import evaluate_model, generate_evaluation_report")

