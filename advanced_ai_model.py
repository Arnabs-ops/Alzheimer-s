# Advanced Alzheimer's Disease AI Model

This notebook implements a state-of-the-art AI model with advanced preprocessing, feature selection, hyperparameter tuning, and ensemble methods.

## Environment Setup
```python
import os, sys, warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('./src')

# Import advanced model
from advanced_model import AdvancedAlzheimerModel, load_real_data

# Additional imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import joblib
from datetime import datetime

# Configuration
RANDOM_STATE = 42
RESULTS_DIR = 'results/advanced'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("🚀 Advanced AI Model Environment Ready!")
```

## Load Real Alzheimer's Data
```python
# Load real data
X, y = load_real_data()

print(f"📈 Dataset Shape: {X.shape}")
print(f"🎯 Target Distribution: {np.bincount(y)}")
print(f"📊 Target Balance: {np.bincount(y) / len(y)}")
print(f"🔢 Data Types: X={X.dtype}, y={y.dtype}")

# Check for missing values
missing_X = np.isnan(X).sum()
missing_y = np.isnan(y).sum()
print(f"❓ Missing Values: X={missing_X}, y={missing_y}")

# Check for infinite values
inf_X = np.isinf(X).sum()
inf_y = np.isinf(y).sum()
print(f"♾️ Infinite Values: X={inf_X}, y={inf_y}")
```

## Train-Test Split
```python
# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"📚 Training Set: {X_train.shape} (Target: {np.bincount(y_train)})")
print(f"🧪 Test Set: {X_test.shape} (Target: {np.bincount(y_test)})")
print(f"⚖️ Class Balance - Train: {np.bincount(y_train) / len(y_train)}")
print(f"⚖️ Class Balance - Test: {np.bincount(y_test) / len(y_test)}")
```

## Advanced Model Training
```python
# Initialize advanced model
advanced_model = AdvancedAlzheimerModel(random_state=RANDOM_STATE)

# Train with all improvements
results = advanced_model.train_and_evaluate_advanced(
    X_train, y_train, X_test, y_test,
    use_feature_selection=True,
    use_hyperparameter_tuning=True,
    use_ensemble=True
)

print("\n🎉 Advanced Training Complete!")
```

## Performance Analysis
```python
# Create comprehensive results DataFrame
results_data = []
for name, res in results.items():
    results_data.append({
        'Model': name,
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1-Score': res['f1'],
        'AUC': res['auc'] if res['auc'] is not None else 0,
        'CV_Mean': res['cv_mean'],
        'CV_Std': res['cv_std']
    })

results_df = pd.DataFrame(results_data).sort_values('Accuracy', ascending=False)
results_df.reset_index(drop=True, inplace=True)

print("🏆 Model Performance Summary:")
print(results_df.round(4))

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_df.to_csv(f'{RESULTS_DIR}/advanced_results_{timestamp}.csv', index=False)
print(f"\n💾 Results saved to: {RESULTS_DIR}/advanced_results_{timestamp}.csv")
```

## Performance Visualization
```python
# Create comprehensive performance plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy comparison
sns.barplot(data=results_df, x='Model', y='Accuracy', ax=axes[0,0])
axes[0,0].set_title('Model Accuracy Comparison')
axes[0,0].tick_params(axis='x', rotation=45)

# Cross-validation scores
sns.barplot(data=results_df, x='Model', y='CV_Mean', ax=axes[0,1])
axes[0,1].errorbar(range(len(results_df)), results_df['CV_Mean'], 
                   yerr=results_df['CV_Std'], fmt='none', color='black')
axes[0,1].set_title('Cross-Validation Performance')
axes[0,1].tick_params(axis='x', rotation=45)

# F1-Score comparison
sns.barplot(data=results_df, x='Model', y='F1-Score', ax=axes[1,0])
axes[1,0].set_title('F1-Score Comparison')
axes[1,0].tick_params(axis='x', rotation=45)

# AUC comparison
sns.barplot(data=results_df, x='Model', y='AUC', ax=axes[1,1])
axes[1,1].set_title('AUC Comparison')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Save plot
plt.savefig(f'{RESULTS_DIR}/performance_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
print(f"📊 Performance plots saved to: {RESULTS_DIR}/performance_comparison_{timestamp}.png")
```

## Best Model Analysis
```python
# Get best model
best_model_name = results_df.iloc[0]['Model']
best_result = results[best_model_name]

print(f"🥇 Best Model: {best_model_name}")
print(f"📊 Accuracy: {best_result['accuracy']:.4f}")
print(f"📊 Precision: {best_result['precision']:.4f}")
print(f"📊 Recall: {best_result['recall']:.4f}")
print(f"📊 F1-Score: {best_result['f1']:.4f}")
print(f"📊 AUC: {best_result['auc']:.4f}" if best_result['auc'] else "📊 AUC: N/A")
print(f"📊 CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, best_result['pred']))

# Save best model
best_model_path = f'{RESULTS_DIR}/best_advanced_model_{best_model_name.replace(" ", "_")}_{timestamp}.pkl'
joblib.dump(best_result['model'], best_model_path)
print(f"\n💾 Best model saved to: {best_model_path}")
```

## Model Interpretability - SHAP Analysis
```python
# SHAP analysis for tree-based models
try:
    # Get the best model
    best_model = best_result['model']
    
    # Check if it's a tree-based model
    tree_models = ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier', 'ExtraTreesClassifier']
    model_type = type(best_model).__name__
    
    if model_type in tree_models:
        print(f"🌳 Performing SHAP analysis for {model_type}...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        
        # Calculate SHAP values for a sample
        sample_size = min(100, len(X_test))
        shap_values = explainer.shap_values(X_test[:sample_size])
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test[:sample_size], show=False)
        plt.title(f'SHAP Summary Plot - {best_model_name}')
        plt.tight_layout()
        plt.show()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test[:sample_size], plot_type='bar', show=False)
        plt.title(f'SHAP Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.show()
        
        print("✅ SHAP analysis completed!")
        
    else:
        print(f"⚠️ SHAP TreeExplainer not supported for {model_type}")
        print("💡 Try using KernelExplainer for non-tree models")
        
        # Try KernelExplainer for non-tree models
        try:
            print("🔄 Attempting KernelExplainer...")
            explainer = shap.KernelExplainer(best_model.predict_proba, X_train[:50])
            shap_values = explainer.shap_values(X_test[:20])
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test[:20], show=False)
            plt.title(f'SHAP Summary Plot - {best_model_name} (KernelExplainer)')
            plt.tight_layout()
            plt.show()
            
            print("✅ KernelExplainer SHAP analysis completed!")
            
        except Exception as e:
            print(f"❌ KernelExplainer failed: {e}")
            
except Exception as e:
    print(f"❌ SHAP analysis failed: {e}")
    print("💡 This is normal for some model types or data configurations")
```

## ROC Curves and Confusion Matrix
```python
# ROC Curves for all models
plt.figure(figsize=(12, 8))

for name, res in results.items():
    if res['proba'] is not None:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, res['proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Advanced Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Confusion Matrix for best model
from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_result['pred'], 
    display_labels=['No Alzheimer\'s', 'Alzheimer\'s'],
    cmap='Blues',
    normalize='true'
)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.show()

# Save plots
plt.savefig(f'{RESULTS_DIR}/roc_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
print(f"📊 ROC curves saved to: {RESULTS_DIR}/roc_curves_{timestamp}.png")
```

## Model Performance Summary
```python
# Final summary
print("🎉 ADVANCED AI MODEL TRAINING COMPLETE!")
print("=" * 50)

print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"🎯 Target: {len(np.unique(y))} classes")
print(f"⚖️ Class Balance: {np.bincount(y) / len(y)}")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"📈 Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"📈 F1-Score: {best_result['f1']:.4f}")
print(f"📈 AUC: {best_result['auc']:.4f}" if best_result['auc'] else "📈 AUC: N/A")
print(f"📈 CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")

print(f"\n💾 Artifacts Saved:")
print(f"  - Best Model: {best_model_path}")
print(f"  - Results CSV: {RESULTS_DIR}/advanced_results_{timestamp}.csv")
print(f"  - Performance Plots: {RESULTS_DIR}/performance_comparison_{timestamp}.png")
print(f"  - ROC Curves: {RESULTS_DIR}/roc_curves_{timestamp}.png")

print(f"\n🚀 Next Steps:")
print(f"  1. Load the best model for inference")
print(f"  2. Test on new/unseen data")
print(f"  3. Deploy to production")
print(f"  4. Monitor performance over time")

print("\n✨ Advanced AI Model Ready for Production! ✨")
```
