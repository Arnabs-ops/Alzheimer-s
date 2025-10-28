# Machine Learning Models
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

class AlzheimerModelPipeline:
    """
    Comprehensive ML pipeline for Alzheimer's Disease prediction
    """
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'SVM': SVC(random_state=42, probability=True),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
        }
        self.trained_models = {}
        self.results = {}
    
    def train_models(self, X_train, y_train):
        """
        Train all models in the pipeline
        """
        print("🤖 Training models...")
        
        for name, model in self.models.items():
            print(f"   Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"   ✅ {name} trained successfully")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)}")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        """
        print("📊 Evaluating models...")
        
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                accuracy = model.score(X_test, y_test)
                auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   ✅ {name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f if auc_score else 'N/A'}")
                
            except Exception as e:
                print(f"   ❌ {name} evaluation failed: {str(e)}")
    
    def hyperparameter_tuning(self, X_train, y_train, model_name, param_grid, cv=5):
        """
        Perform hyperparameter tuning for a specific model
        """
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return None
        
        print(f"🔧 Tuning hyperparameters for {model_name}...")
        
        model = self.models[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Create ensemble model from best performing models
        """
        print("🎯 Creating ensemble model...")
        
        # Get top 3 models by accuracy
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top_models = [model for name, (model, _) in sorted_models[:3]]
        
        if len(top_models) < 2:
            print("❌ Need at least 2 models for ensemble")
            return None
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[(f'model_{i}', model) for i, model in enumerate(top_models)],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
        
        accuracy = ensemble.score(X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   ✅ Ensemble: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}")
        
        return ensemble
    
    def get_best_model(self):
        """
        Get the best performing model
        """
        if not self.results:
            print("❌ No models have been evaluated yet")
            return None
        
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        return best_model_name, self.results[best_model_name]['model']
