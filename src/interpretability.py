"""
Model Interpretability and Feature Analysis
SHAP, LIME, and feature importance analysis for Alzheimer's Disease prediction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not available. Install with: pip install lime")


class ModelInterpreter:
    """Comprehensive model interpretability analysis"""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
        
    def analyze_feature_importance(self, model, X: np.ndarray, y: np.ndarray, 
                                 model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive feature importance analysis"""
        
        print(f"🔍 Analyzing feature importance for {model_name}...")
        
        results = {}
        
        # 1. Built-in feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            try:
                model.fit(X, y)
                builtin_importance = model.feature_importances_
                results['builtin_importance'] = builtin_importance
                print(f"  ✅ Built-in feature importance available")
            except Exception as e:
                print(f"  ⚠️ Built-in feature importance failed: {e}")
        
        # 2. Permutation importance
        try:
            perm_importance = permutation_importance(
                model, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            results['permutation_importance'] = {
                'mean': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }
            print(f"  ✅ Permutation importance completed")
        except Exception as e:
            print(f"  ⚠️ Permutation importance failed: {e}")
        
        # 3. Coefficient analysis (for linear models)
        if hasattr(model, 'coef_'):
            try:
                model.fit(X, y)
                if hasattr(model.coef_, 'shape') and len(model.coef_.shape) > 1:
                    # Multi-class case
                    coef_importance = np.mean(np.abs(model.coef_), axis=0)
                else:
                    coef_importance = np.abs(model.coef_)
                results['coefficient_importance'] = coef_importance
                print(f"  ✅ Coefficient analysis completed")
            except Exception as e:
                print(f"  ⚠️ Coefficient analysis failed: {e}")
        
        return results
    
    def plot_feature_importance(self, importance_data: Dict[str, Any], 
                              model_name: str = "Model", top_k: int = 20) -> None:
        """Plot feature importance from different methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Built-in importance
        if 'builtin_importance' in importance_data:
            importance = importance_data['builtin_importance']
            if self.feature_names:
                feature_names = self.feature_names
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Get top K features
            top_indices = np.argsort(importance)[-top_k:]
            top_importance = importance[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            axes[plot_idx].barh(range(len(top_importance)), top_importance)
            axes[plot_idx].set_yticks(range(len(top_importance)))
            axes[plot_idx].set_yticklabels(top_names)
            axes[plot_idx].set_title(f'{model_name} - Built-in Importance (Top {top_k})')
            axes[plot_idx].set_xlabel('Importance')
            plot_idx += 1
        
        # Permutation importance
        if 'permutation_importance' in importance_data:
            perm_data = importance_data['permutation_importance']
            importance = perm_data['mean']
            std = perm_data['std']
            
            if self.feature_names:
                feature_names = self.feature_names
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Get top K features
            top_indices = np.argsort(importance)[-top_k:]
            top_importance = importance[top_indices]
            top_std = std[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            axes[plot_idx].barh(range(len(top_importance)), top_importance, xerr=top_std)
            axes[plot_idx].set_yticks(range(len(top_importance)))
            axes[plot_idx].set_yticklabels(top_names)
            axes[plot_idx].set_title(f'{model_name} - Permutation Importance (Top {top_k})')
            axes[plot_idx].set_xlabel('Importance')
            plot_idx += 1
        
        # Coefficient importance
        if 'coefficient_importance' in importance_data:
            importance = importance_data['coefficient_importance']
            if self.feature_names:
                feature_names = self.feature_names
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Get top K features
            top_indices = np.argsort(importance)[-top_k:]
            top_importance = importance[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            axes[plot_idx].barh(range(len(top_importance)), top_importance)
            axes[plot_idx].set_yticks(range(len(top_importance)))
            axes[plot_idx].set_yticklabels(top_names)
            axes[plot_idx].set_title(f'{model_name} - Coefficient Importance (Top {top_k})')
            axes[plot_idx].set_xlabel('|Coefficient|')
            plot_idx += 1
        
        # Feature correlation heatmap (if we have multiple importance measures)
        if len(importance_data) > 1:
            importance_matrix = []
            method_names = []
            
            for method, data in importance_data.items():
                if method == 'builtin_importance':
                    importance_matrix.append(data)
                    method_names.append('Built-in')
                elif method == 'permutation_importance':
                    importance_matrix.append(data['mean'])
                    method_names.append('Permutation')
                elif method == 'coefficient_importance':
                    importance_matrix.append(data)
                    method_names.append('Coefficient')
            
            if len(importance_matrix) > 1:
                importance_matrix = np.array(importance_matrix)
                correlation_matrix = np.corrcoef(importance_matrix)
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           xticklabels=method_names, yticklabels=method_names,
                           ax=axes[plot_idx])
                axes[plot_idx].set_title('Feature Importance Correlation')
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def shap_analysis(self, model, X_train: np.ndarray, X_test: np.ndarray, 
                     model_name: str = "Model") -> Optional[Dict[str, Any]]:
        """Perform SHAP analysis if available"""
        
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available. Skipping SHAP analysis.")
            return None
        
        print(f"🔍 Performing SHAP analysis for {model_name}...")
        
        try:
            # Train model
            model.fit(X_train, np.zeros(len(X_train)))  # Dummy y for now
            
            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test[:100])  # Sample for speed
                    explainer_type = "TreeExplainer"
                except:
                    # Fallback to KernelExplainer
                    explainer = shap.KernelExplainer(model.predict_proba, X_train[:50])
                    shap_values = explainer.shap_values(X_test[:20])
                    explainer_type = "KernelExplainer"
            else:
                # Linear models
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test[:100])
                explainer_type = "LinearExplainer"
            
            self.shap_explainer = explainer
            
            print(f"  ✅ SHAP analysis completed using {explainer_type}")
            
            return {
                'explainer': explainer,
                'shap_values': shap_values,
                'explainer_type': explainer_type
            }
            
        except Exception as e:
            print(f"  ❌ SHAP analysis failed: {e}")
            return None
    
    def plot_shap_summary(self, shap_data: Dict[str, Any], model_name: str = "Model") -> None:
        """Plot SHAP summary plots"""
        
        if not SHAP_AVAILABLE or shap_data is None:
            print("⚠️ SHAP data not available for plotting")
            return
        
        print(f"📊 Creating SHAP summary plots for {model_name}...")
        
        try:
            shap_values = shap_data['shap_values']
            explainer_type = shap_data['explainer_type']
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            if explainer_type == "TreeExplainer" and isinstance(shap_values, list):
                # Multi-class case
                shap.summary_plot(shap_values[0], show=False)  # Plot first class
            else:
                shap.summary_plot(shap_values, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()
            
            # Bar plot
            plt.figure(figsize=(10, 8))
            if explainer_type == "TreeExplainer" and isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], plot_type='bar', show=False)
            else:
                shap.summary_plot(shap_values, plot_type='bar', show=False)
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ SHAP plotting failed: {e}")
    
    def lime_analysis(self, model, X_train: np.ndarray, X_test: np.ndarray, 
                     feature_names: Optional[List[str]] = None,
                     model_name: str = "Model") -> Optional[Dict[str, Any]]:
        """Perform LIME analysis if available"""
        
        if not LIME_AVAILABLE:
            print("⚠️ LIME not available. Skipping LIME analysis.")
            return None
        
        print(f"🔍 Performing LIME analysis for {model_name}...")
        
        try:
            # Train model
            model.fit(X_train, np.zeros(len(X_train)))  # Dummy y for now
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train, feature_names=feature_names, mode='classification'
            )
            
            # Explain a few instances
            explanations = []
            for i in range(min(5, len(X_test))):
                exp = explainer.explain_instance(X_test[i], model.predict_proba)
                explanations.append(exp)
            
            self.lime_explainer = explainer
            
            print(f"  ✅ LIME analysis completed for {len(explanations)} instances")
            
            return {
                'explainer': explainer,
                'explanations': explanations
            }
            
        except Exception as e:
            print(f"  ❌ LIME analysis failed: {e}")
            return None
    
    def plot_lime_explanations(self, lime_data: Dict[str, Any], 
                              model_name: str = "Model") -> None:
        """Plot LIME explanations"""
        
        if not LIME_AVAILABLE or lime_data is None:
            print("⚠️ LIME data not available for plotting")
            return
        
        print(f"📊 Creating LIME explanation plots for {model_name}...")
        
        try:
            explanations = lime_data['explanations']
            
            # Plot explanations for first few instances
            for i, exp in enumerate(explanations[:3]):
                plt.figure(figsize=(10, 6))
                exp.as_pyplot_figure()
                plt.title(f'LIME Explanation - {model_name} (Instance {i+1})')
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"❌ LIME plotting failed: {e}")
    
    def plot_confusion_matrix_detailed(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     class_names: Optional[List[str]] = None,
                                     model_name: str = "Model") -> None:
        """Plot detailed confusion matrix with percentages and counts"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=class_names, yticklabels=class_names)
        ax1.set_title(f'{model_name} - Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Percentage confusion matrix
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                   xticklabels=class_names, yticklabels=class_names)
        ax2.set_title(f'{model_name} - Confusion Matrix (%)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_partial_dependence(self, model, X: np.ndarray, feature_indices: List[int],
                               feature_names: Optional[List[str]] = None,
                               model_name: str = "Model") -> None:
        """Plot partial dependence plots for selected features"""
        
        print(f"📊 Creating partial dependence plots for {model_name}...")
        
        n_features = len(feature_indices)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature_idx in enumerate(feature_indices):
            row = i // n_cols
            col = i % n_cols
            
            try:
                # Calculate partial dependence
                pd_result = partial_dependence(model, X, [feature_idx])
                
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.plot(pd_result['values'][0], pd_result['average'][0])
                ax.set_xlabel(f'Feature {feature_idx}' if not feature_names else feature_names[feature_idx])
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'Partial Dependence - Feature {feature_idx}')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"  ⚠️ Partial dependence failed for feature {feature_idx}: {e}")
        
        # Hide unused subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(f'Partial Dependence Plots - {model_name}')
        plt.tight_layout()
        plt.show()
    
    def comprehensive_interpretability_report(self, model, X_train: np.ndarray, y_train: np.ndarray,
                                            X_test: np.ndarray, y_test: np.ndarray,
                                            model_name: str = "Model") -> Dict[str, Any]:
        """Generate comprehensive interpretability report"""
        
        print(f"\n🔍 Comprehensive Interpretability Report for {model_name}")
        print("=" * 60)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Feature importance analysis
        importance_data = self.analyze_feature_importance(model, X_train, y_train, model_name)
        
        # SHAP analysis
        shap_data = self.shap_analysis(model, X_train, X_test, model_name)
        
        # LIME analysis
        lime_data = self.lime_analysis(model, X_train, X_test, self.feature_names, model_name)
        
        # Generate plots
        self.plot_feature_importance(importance_data, model_name)
        
        if shap_data:
            self.plot_shap_summary(shap_data, model_name)
        
        if lime_data:
            self.plot_lime_explanations(lime_data, model_name)
        
        # Confusion matrix
        class_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
        self.plot_confusion_matrix_detailed(y_test, y_pred, class_names, model_name)
        
        # Partial dependence for top features
        if importance_data:
            if 'permutation_importance' in importance_data:
                top_features = np.argsort(importance_data['permutation_importance']['mean'])[-5:]
                self.plot_partial_dependence(model, X_test, top_features.tolist(), 
                                           self.feature_names, model_name)
        
        report = {
            'model_name': model_name,
            'importance_data': importance_data,
            'shap_data': shap_data,
            'lime_data': lime_data,
            'predictions': y_pred
        }
        
        print(f"\n✅ Interpretability analysis completed for {model_name}")
        
        return report
