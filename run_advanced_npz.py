#!/usr/bin/env python3
"""
Simple Advanced AI Model Runner - Focus on NPZ data
This runs the advanced model on the NPZ data that already achieved 100% accuracy
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('./src')

def main():
    print("🧠 Advanced Alzheimer's Disease AI Model - NPZ Focus")
    print("=" * 60)
    
    try:
        # Import advanced model
        from advanced_model import AdvancedAlzheimerModel
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        import joblib
        from datetime import datetime
        
        # Configuration
        RANDOM_STATE = 42
        RESULTS_DIR = 'results/advanced'
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        print("🚀 Loading NPZ data (already achieved 100% accuracy)...")
        
        # Load NPZ data directly
        data = np.load('data/raw/preprocessed_alz_data.npz', allow_pickle=True)
        keys = list(data.keys())
        
        # Find X and y arrays
        X_key = None
        y_key = None
        
        for key in keys:
            if key.lower() in ['x', 'features', 'x_train']:
                X_key = key
            elif key.lower() in ['y', 'labels', 'target', 'y_train']:
                y_key = key
        
        if X_key and y_key:
            X = data[X_key]
            y = data[y_key]
            
            # Handle multi-dimensional y
            if len(y.shape) > 1:
                if y.shape[1] == 1:
                    y = y.ravel()
                else:
                    y = np.argmax(y, axis=1)
            
            print(f"✅ Loaded NPZ data: X {X.shape}, y {y.shape}")
        else:
            print("❌ Could not find X and y in NPZ file")
            return None, None
        
        print(f"📈 Dataset Shape: {X.shape}")
        print(f"🎯 Target Distribution: {np.bincount(y)}")
        print(f"📊 Target Balance: {np.bincount(y) / len(y)}")
        print(f"🔢 Unique Classes: {len(np.unique(y))}")
        
        # Check for missing values
        try:
            missing_X = np.isnan(X).sum() if X.dtype in [np.float32, np.float64] else 0
            missing_y = np.isnan(y).sum() if y.dtype in [np.float32, np.float64] else 0
            print(f"❓ Missing Values: X={missing_X}, y={missing_y}")
        except:
            print("❓ Missing Values: Unable to check")
        
        # Check for infinite values
        try:
            inf_X = np.isinf(X).sum() if X.dtype in [np.float32, np.float64] else 0
            inf_y = np.isinf(y).sum() if y.dtype in [np.float32, np.float64] else 0
            print(f"♾️ Infinite Values: X={inf_X}, y={inf_y}")
        except:
            print("♾️ Infinite Values: Unable to check")
        
        print("\n🔄 Creating train-test split...")
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"📚 Training Set: {X_train.shape} (Target: {np.bincount(y_train)})")
        print(f"🧪 Test Set: {X_test.shape} (Target: {np.bincount(y_test)})")
        
        print("\n🤖 Training advanced AI model...")
        
        # Initialize advanced model
        advanced_model = AdvancedAlzheimerModel(random_state=RANDOM_STATE)
        
        # Train with all improvements
        results = advanced_model.train_and_evaluate_advanced(
            X_train, y_train, X_test, y_test,
            use_feature_selection=True,
            use_hyperparameter_tuning=True,
            use_ensemble=True
        )
        
        print("\n📊 Analyzing results...")
        
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
        
        print("\n🏆 Model Performance Summary:")
        print(results_df.round(4))
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(f'{RESULTS_DIR}/advanced_npz_results_{timestamp}.csv', index=False)
        
        # Get best model
        best_model_name = results_df.iloc[0]['Model']
        best_result = results[best_model_name]
        
        print(f"\n🥇 BEST MODEL: {best_model_name}")
        print(f"📈 Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
        print(f"📈 Precision: {best_result['precision']:.4f}")
        print(f"📈 Recall: {best_result['recall']:.4f}")
        print(f"📈 F1-Score: {best_result['f1']:.4f}")
        print(f"📈 AUC: {best_result['auc']:.4f}" if best_result['auc'] else "📈 AUC: N/A")
        print(f"📈 CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, best_result['pred']))
        
        # Save best model
        best_model_path = f'{RESULTS_DIR}/best_advanced_npz_model_{best_model_name.replace(" ", "_")}_{timestamp}.pkl'
        joblib.dump(best_result['model'], best_model_path)
        
        print(f"\n💾 Artifacts Saved:")
        print(f"  - Best Model: {best_model_path}")
        print(f"  - Results CSV: {RESULTS_DIR}/advanced_npz_results_{timestamp}.csv")
        
        print(f"\n🎉 ADVANCED AI MODEL TRAINING COMPLETE!")
        print(f"✨ Best Model: {best_model_name} with {best_result['accuracy']*100:.2f}% accuracy!")
        
        # Compare with previous results
        print(f"\n📊 Performance Comparison:")
        print(f"  - Previous NPZ Results: 100% accuracy (Random Forest, XGBoost)")
        print(f"  - Advanced Model Best: {best_result['accuracy']*100:.2f}% accuracy ({best_model_name})")
        
        if best_result['accuracy'] >= 0.99:
            print(f"🎯 EXCELLENT! Advanced model maintains near-perfect performance!")
        elif best_result['accuracy'] >= 0.95:
            print(f"🎯 GREAT! Advanced model shows excellent performance!")
        else:
            print(f"🎯 GOOD! Advanced model shows solid performance!")
        
        return results_df, best_result
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you have installed all required packages:")
        print("   pip install scikit-learn xgboost lightgbm shap matplotlib seaborn pandas numpy joblib")
        return None, None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your data files and try again")
        return None, None

if __name__ == "__main__":
    results_df, best_result = main()
    
    if results_df is not None:
        print("\n🚀 Advanced AI Model Ready!")
        print("💡 Next steps:")
        print("   1. Check the results in results/advanced/")
        print("   2. Load the best model for inference")
        print("   3. Test on new data")
        print("   4. Compare with previous 100% accuracy results")
    else:
        print("\n❌ Training failed. Please check the errors above.")
