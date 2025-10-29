"""
Robust Validation Strategies for Alzheimer's Disease Prediction
Addresses overfitting and data leakage issues
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    validation_curve, learning_curve, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class RobustValidator:
    """Robust validation strategies to detect and prevent overfitting"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.validation_results = {}
        
    def create_fresh_splits(self, X: np.ndarray, y: np.ndarray, 
                          test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """Create fresh train/validation/test splits to avoid data leakage"""
        
        print("🔄 Creating fresh train/validation/test splits...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"📊 Fresh splits created:")
        print(f"  Train: {X_train.shape} (Target: {np.bincount(y_train)})")
        print(f"  Validation: {X_val.shape} (Target: {np.bincount(y_val)})")
        print(f"  Test: {X_test.shape} (Target: {np.bincount(y_test)})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def stratified_kfold_cv(self, model, X: np.ndarray, y: np.ndarray, 
                          cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """Perform stratified K-fold cross-validation"""
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
    
    def nested_cv_hyperparameter_tuning(self, model, param_grid: Dict, 
                                      X: np.ndarray, y: np.ndarray,
                                      inner_cv: int = 3, outer_cv: int = 5) -> Dict[str, Any]:
        """Perform nested cross-validation for hyperparameter tuning"""
        
        print(f"🔍 Performing nested CV ({outer_cv}-fold outer, {inner_cv}-fold inner)...")
        
        outer_scores = []
        best_params_list = []
        
        skf_outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf_outer.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Inner CV for hyperparameter tuning
            inner_search = GridSearchCV(
                model, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1
            )
            inner_search.fit(X_train_fold, y_train_fold)
            
            # Evaluate best model on outer fold
            best_model = inner_search.best_estimator_
            val_score = best_model.score(X_val_fold, y_val_fold)
            
            outer_scores.append(val_score)
            best_params_list.append(inner_search.best_params_)
            
            print(f"  Fold {fold+1}: Score={val_score:.4f}, Params={inner_search.best_params_}")
        
        return {
            'nested_scores': outer_scores,
            'nested_mean': np.mean(outer_scores),
            'nested_std': np.std(outer_scores),
            'best_params': best_params_list
        }
    
    def generate_learning_curves(self, model, X: np.ndarray, y: np.ndarray,
                               train_sizes: List[float] = None, cv_folds: int = 5) -> Dict[str, Any]:
        """Generate learning curves to detect overfitting"""
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        print("📈 Generating learning curves...")
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv_folds, 
            scoring='accuracy', n_jobs=-1, random_state=self.random_state
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        return {
            'train_sizes': train_sizes_abs,
            'train_mean': train_mean,
            'train_std': train_std,
            'val_mean': val_mean,
            'val_std': val_std,
            'overfitting_gap': train_mean - val_mean
        }
    
    def generate_validation_curves(self, model, X: np.ndarray, y: np.ndarray,
                                 param_name: str, param_range: List, cv_folds: int = 5) -> Dict[str, Any]:
        """Generate validation curves for hyperparameter analysis"""
        
        print(f"📊 Generating validation curves for {param_name}...")
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv_folds, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        return {
            'param_range': param_range,
            'train_mean': train_mean,
            'train_std': train_std,
            'val_mean': val_mean,
            'val_std': val_std,
            'best_param': param_range[np.argmax(val_mean)]
        }
    
    def plot_learning_curves(self, learning_curve_data: Dict[str, Any], 
                            model_name: str = "Model") -> None:
        """Plot learning curves"""
        
        plt.figure(figsize=(10, 6))
        
        train_sizes = learning_curve_data['train_sizes']
        train_mean = learning_curve_data['train_mean']
        train_std = learning_curve_data['train_std']
        val_mean = learning_curve_data['val_mean']
        val_std = learning_curve_data['val_std']
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add overfitting detection
        gap = learning_curve_data['overfitting_gap']
        max_gap = np.max(gap)
        if max_gap > 0.1:
            plt.text(0.5, 0.1, f'⚠️ Potential Overfitting (Max Gap: {max_gap:.3f})', 
                    transform=plt.gca().transAxes, fontsize=12, color='red')
        else:
            plt.text(0.5, 0.1, f'✅ Good Generalization (Max Gap: {max_gap:.3f})', 
                    transform=plt.gca().transAxes, fontsize=12, color='green')
        
        plt.tight_layout()
        plt.show()
    
    def plot_validation_curves(self, val_curve_data: Dict[str, Any], 
                             param_name: str, model_name: str = "Model") -> None:
        """Plot validation curves"""
        
        plt.figure(figsize=(10, 6))
        
        param_range = val_curve_data['param_range']
        train_mean = val_curve_data['train_mean']
        train_std = val_curve_data['train_std']
        val_mean = val_curve_data['val_mean']
        val_std = val_curve_data['val_std']
        
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('Accuracy Score')
        plt.title(f'Validation Curves - {model_name} ({param_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark best parameter
        best_param = val_curve_data['best_param']
        plt.axvline(x=best_param, color='green', linestyle='--', 
                   label=f'Best {param_name}: {best_param}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def detect_overfitting(self, train_score: float, val_score: float, 
                          threshold: float = 0.05) -> Dict[str, Any]:
        """Detect overfitting based on train-validation gap"""
        
        gap = train_score - val_score
        
        if gap > threshold:
            severity = "High" if gap > 0.15 else "Medium"
            recommendation = "Increase regularization, reduce model complexity, or get more data"
        else:
            severity = "Low"
            recommendation = "Model shows good generalization"
        
        return {
            'gap': gap,
            'severity': severity,
            'recommendation': recommendation,
            'is_overfitting': gap > threshold
        }
    
    def bootstrap_confidence_intervals(self, model, X: np.ndarray, y: np.ndarray,
                                     n_bootstrap: int = 20, test_size: float = 0.2) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for model performance"""
        
        print(f"🔄 Computing bootstrap confidence intervals ({n_bootstrap} iterations)...")
        
        scores = []

        try:
            for i in range(n_bootstrap):
                if i % max(1, n_bootstrap // 5) == 0:
                    print(f"  Bootstrap iteration {i}/{n_bootstrap}")

                # Bootstrap sample
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X[indices]
                y_boot = y[indices]

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_boot, y_boot, test_size=test_size, random_state=self.random_state, stratify=y_boot
                )

                # Train and evaluate
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)
        except KeyboardInterrupt:
            print(f"\n⏹️ Bootstrap interrupted early. Using {len(scores)} samples for CI.")
        
        scores = np.array(scores)
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'ci_95_lower': np.percentile(scores, 2.5),
            'ci_95_upper': np.percentile(scores, 97.5),
            'ci_99_lower': np.percentile(scores, 0.5),
            'ci_99_upper': np.percentile(scores, 99.5)
        }
    
    def comprehensive_validation_report(self, model, X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray,
                                       X_test: np.ndarray, y_test: np.ndarray,
                                       model_name: str = "Model") -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        print(f"\n📋 Comprehensive Validation Report for {model_name}")
        print("=" * 60)
        
        # Clean data - handle infinity and NaN values
        def clean_data(X):
            X = np.array(X, dtype=np.float64)
            # Replace infinity and very large values with NaN
            X = np.where(np.isinf(X), np.nan, X)
            X = np.where(np.abs(X) > 1e10, np.nan, X)
            # Fill NaN with median
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            return X
        
        # Clean all datasets
        X_train = clean_data(X_train)
        X_val = clean_data(X_val)
        X_test = clean_data(X_test)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate scores
        train_score = accuracy_score(y_train, train_pred)
        val_score = accuracy_score(y_val, val_pred)
        test_score = accuracy_score(y_test, test_pred)
        
        # Cross-validation
        cv_results = self.stratified_kfold_cv(model, X_train, y_train)
        
        # Overfitting detection
        overfitting_info = self.detect_overfitting(train_score, val_score)
        
        # Bootstrap confidence intervals
        bootstrap_ci = self.bootstrap_confidence_intervals(model, X_train, y_train)
        
        report = {
            'model_name': model_name,
            'train_score': train_score,
            'val_score': val_score,
            'test_score': test_score,
            'cv_mean': cv_results['cv_mean'],
            'cv_std': cv_results['cv_std'],
            'overfitting': overfitting_info,
            'bootstrap_ci': bootstrap_ci,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred
        }
        
        # Print summary
        print(f"📊 Performance Summary:")
        print(f"  Train Accuracy: {train_score:.4f}")
        print(f"  Validation Accuracy: {val_score:.4f}")
        print(f"  Test Accuracy: {test_score:.4f}")
        print(f"  CV Mean ± Std: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        print(f"  Bootstrap Mean ± Std: {bootstrap_ci['mean']:.4f} ± {bootstrap_ci['std']:.4f}")
        
        print(f"\n🔍 Overfitting Analysis:")
        print(f"  Train-Val Gap: {overfitting_info['gap']:.4f}")
        print(f"  Severity: {overfitting_info['severity']}")
        print(f"  Recommendation: {overfitting_info['recommendation']}")
        
        print(f"\n📈 Confidence Intervals (95%):")
        print(f"  [{bootstrap_ci['ci_95_lower']:.4f}, {bootstrap_ci['ci_95_upper']:.4f}]")
        
        return report
