"""
Multimodal fusion: combine predictions from speech, text, and behavior modules.
Implements late fusion (weighted average) with meta-learner.
Includes stub for attention-based fusion if needed.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


def late_fusion_weighted_average(
    modality_predictions: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Simple weighted average fusion.
    
    Args:
        modality_predictions: Dict mapping modality_name -> prediction probabilities
        weights: Optional weights for each modality (if None, equal weights)
    
    Returns:
        Fused prediction probabilities
    """
    if not modality_predictions:
        raise ValueError("No modality predictions provided")
    
    predictions_list = list(modality_predictions.values())
    
    # Ensure all predictions have same shape
    shapes = [p.shape for p in predictions_list]
    if len(set(shapes)) > 1:
        print("⚠️ Predictions have different shapes. Taking average of class indices.")
        # Convert probabilities to class predictions if needed
        predictions_list = [np.argmax(p, axis=1) if p.ndim > 1 else p for p in predictions_list]
        # Convert back to one-hot for averaging
        n_classes = max([int(np.max(p)) + 1 for p in predictions_list])
        predictions_list = [
            np.eye(n_classes)[p.astype(int)] if p.ndim == 1 else p
            for p in predictions_list
        ]
    
    # Set weights
    if weights is None:
        weights = {mod: 1.0 / len(modality_predictions) for mod in modality_predictions.keys()}
    
    # Weighted average
    weighted_sum = np.zeros_like(predictions_list[0])
    for mod, pred in modality_predictions.items():
        if pred.shape != weighted_sum.shape:
            pred = np.eye(weighted_sum.shape[1])[np.argmax(pred, axis=1)] if pred.ndim > 1 else pred
        weighted_sum += weights[mod] * pred
    
    return weighted_sum


def late_fusion_meta_learner(
    modality_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    cv_folds: int = 5
) -> Tuple[np.ndarray, LogisticRegression]:
    """
    Late fusion using meta-learner (LogisticRegression).
    
    Args:
        modality_predictions: Dict mapping modality_name -> prediction probabilities
        y_true: True labels for training meta-learner
        cv_folds: Cross-validation folds
    
    Returns:
        Fused predictions and trained meta-learner
    """
    # Stack predictions as features
    features_list = []
    for mod, pred in modality_predictions.items():
        if pred.ndim == 1:
            # Convert to one-hot if needed
            n_classes = len(np.unique(y_true))
            pred = np.eye(n_classes)[pred.astype(int)]
        features_list.append(pred)
    
    X_fusion = np.hstack(features_list)
    
    # Train meta-learner
    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='ovr',
        class_weight='balanced'
    )
    meta_learner.fit(X_fusion, y_true)
    
    # Predict
    y_pred_fused = meta_learner.predict(X_fusion)
    y_proba_fused = meta_learner.predict_proba(X_fusion)
    
    return y_proba_fused, meta_learner


def attention_fusion_stub(
    modality_features: Dict[str, np.ndarray],
    modality_predictions: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Any]:
    """
    Stub for attention-based fusion (placeholder for future implementation).
    
    This would use:
    - Transformer-style attention to learn weights dynamically
    - Cross-modal attention mechanisms
    
    For now, returns simple weighted average.
    
    Args:
        modality_features: Raw features from each modality
        modality_predictions: Predictions from each modality
    
    Returns:
        Fused predictions and attention weights (placeholder)
    """
    print("ℹ️ Attention fusion stub - using weighted average for now")
    
    # Placeholder: use weighted average
    fused = late_fusion_weighted_average(modality_predictions)
    
    # Placeholder attention weights (equal weights)
    attention_weights = {mod: 1.0 / len(modality_predictions) for mod in modality_predictions.keys()}
    
    return fused, attention_weights


def fuse_modality_predictions(
    speech_results: Optional[Dict[str, Any]] = None,
    text_results: Optional[Dict[str, Any]] = None,
    behavior_results: Optional[Dict[str, Any]] = None,
    y_test: Optional[np.ndarray] = None,
    fusion_method: str = 'meta_learner'
) -> Dict[str, Any]:
    """
    Main fusion function that combines all modality predictions.
    
    Args:
        speech_results: Results dict from speech model (must have 'y_proba')
        text_results: Results dict from text model (must have 'y_proba')
        behavior_results: Results dict from behavior model (must have 'y_proba')
        y_test: True labels (required for meta-learner)
        fusion_method: 'weighted_average', 'meta_learner', or 'attention'
    
    Returns:
        Dictionary with fused predictions and metrics
    """
    modality_predictions = {}
    
    if speech_results and 'y_proba' in speech_results:
        modality_predictions['speech'] = speech_results['y_proba']
    
    if text_results and 'y_proba' in text_results:
        modality_predictions['text'] = text_results['y_proba']
    
    if behavior_results and 'y_proba' in behavior_results:
        modality_predictions['behavior'] = behavior_results['y_proba']
    
    if not modality_predictions:
        raise ValueError("At least one modality prediction required")
    
    print(f"🔀 Fusing {len(modality_predictions)} modalities using {fusion_method}...")
    
    # Perform fusion
    if fusion_method == 'weighted_average':
        y_proba_fused = late_fusion_weighted_average(modality_predictions)
        meta_model = None
    
    elif fusion_method == 'meta_learner':
        if y_test is None:
            print("⚠️ y_test required for meta-learner. Falling back to weighted average.")
            y_proba_fused = late_fusion_weighted_average(modality_predictions)
            meta_model = None
        else:
            y_proba_fused, meta_model = late_fusion_meta_learner(
                modality_predictions, y_test
            )
    
    elif fusion_method == 'attention':
        # Use attention stub
        modality_features = {}  # Placeholder - would need raw features
        y_proba_fused, attention_weights = attention_fusion_stub(
            modality_features, modality_predictions
        )
        meta_model = {'attention_weights': attention_weights}
    
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Convert probabilities to predictions
    y_pred_fused = np.argmax(y_proba_fused, axis=1)
    
    # Compute metrics (if y_test provided)
    metrics = {}
    if y_test is not None:
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
        
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred_fused))
        
        try:
            if y_proba_fused.shape[1] == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba_fused[:, 1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba_fused, multi_class='ovr', average='weighted'))
        except Exception:
            metrics['roc_auc'] = None
        
        metrics['precision'] = float(precision_score(y_test, y_pred_fused, average='weighted', zero_division=0))
        metrics['recall'] = float(recall_score(y_test, y_pred_fused, average='weighted', zero_division=0))
        metrics['f1'] = float(f1_score(y_test, y_pred_fused, average='weighted', zero_division=0))
        
        print(f"   ✅ Fused Accuracy: {metrics['accuracy']:.4f}")
        if metrics['roc_auc']:
            print(f"   ✅ Fused ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return {
        'y_pred': y_pred_fused,
        'y_proba': y_proba_fused,
        'meta_model': meta_model,
        'metrics': metrics,
        'modality_predictions': modality_predictions
    }


if __name__ == "__main__":
    print("This module provides fusion functions for multimodal predictions.")
    print("Use it as: from models.fusion import fuse_modality_predictions")

