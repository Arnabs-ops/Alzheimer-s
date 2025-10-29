"""
Advanced AI Model for Alzheimer's Disease Prediction
Includes hyperparameter tuning, ensemble methods, and advanced preprocessing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, 
    StratifiedKFold, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    PowerTransformer, QuantileTransformer
)
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, 
    mutual_info_classif, f_classif
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

# Advanced preprocessing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class AdvancedAlzheimerModel:
    """Advanced AI model with comprehensive improvements for Alzheimer's prediction"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scalers = {}
        self.feature_selectors = {}
        self.models = {}
        self.best_model = None
        self.feature_names = None
        
    def get_regularized_models(self) -> Dict[str, Any]:
        """Get regularized models to prevent overfitting"""
        return {
            # Linear Models with strong regularization
            'Logistic Regression (L1)': LogisticRegression(
                random_state=self.random_state, 
                max_iter=500,
                C=0.1,  # Strong regularization
                penalty='l1',
                solver='saga',
                multi_class='ovr'
            ),
            
            'Logistic Regression (L2)': LogisticRegression(
                random_state=self.random_state, 
                max_iter=500,
                C=0.1,  # Strong regularization
                penalty='l2',
                multi_class='ovr'
            ),
            
            'Ridge Classifier': RidgeClassifier(
                alpha=1.0,  # Strong regularization
                random_state=self.random_state
            ),
            
            # Tree-based Models with reduced complexity
            'Random Forest (Regularized)': RandomForestClassifier(
                n_estimators=50,  # Further reduced for speed
                max_depth=8,       # Reduced from 15
                min_samples_split=10,  # Increased from 5
                min_samples_leaf=5,    # Increased from 2
                max_features=0.5,      # Reduced from 'sqrt'
                random_state=self.random_state,
                n_jobs=1
            ),
            
            'Extra Trees (Regularized)': ExtraTreesClassifier(
                n_estimators=50,  # Further reduced for speed
                max_depth=8,       # Reduced from 20
                min_samples_split=10,  # Increased from 3
                min_samples_leaf=5,    # Increased from 1
                max_features=0.5,      # Reduced from 'sqrt'
                random_state=self.random_state,
                n_jobs=1
            ),
            
            # Gradient Boosting with early stopping
            'XGBoost (Regularized)': xgb.XGBClassifier(
                n_estimators=100,  # Further reduced for speed
                max_depth=4,       # Reduced from 8
                learning_rate=0.1, # Increased from 0.05
                subsample=0.7,     # Reduced from 0.8
                colsample_bytree=0.7,  # Reduced from 0.8
                reg_alpha=1.0,     # Increased regularization
                reg_lambda=1.0,    # Increased regularization
                early_stopping_rounds=10,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                verbosity=0
            ),
            
            'LightGBM (Regularized)': lgb.LGBMClassifier(
                n_estimators=100,  # Further reduced for speed
                max_depth=6,       # Reduced from 10
                learning_rate=0.1, # Increased from 0.05
                subsample=0.7,     # Reduced from 0.8
                colsample_bytree=0.7,  # Reduced from 0.8
                reg_alpha=1.0,     # Increased regularization
                reg_lambda=1.0,    # Increased regularization
                random_state=self.random_state,
                verbose=-1
            ),
            
            'Gradient Boosting (Regularized)': GradientBoostingClassifier(
                n_estimators=50,  # Further reduced for speed
                max_depth=4,       # Reduced from 8
                learning_rate=0.1, # Increased from 0.05
                subsample=0.7,     # Reduced from 0.8
                random_state=self.random_state
            ),
            
            # SVM with regularization
            'SVM (Regularized)': SVC(
                C=0.1,  # Strong regularization
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            # Neural Network with dropout (simulated with regularization)
            'Neural Network (Regularized)': MLPClassifier(
                hidden_layer_sizes=(50, 25),  # Reduced complexity
                activation='relu',
                solver='adam',
                alpha=0.01,  # Increased regularization
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.random_state
            ),
            
            # Naive Bayes (already regularized)
            'Naive Bayes': GaussianNB(),
            
            # KNN with regularization
            'KNN (Regularized)': KNeighborsClassifier(
                n_neighbors=15,  # Increased from 7
                weights='distance',
                metric='minkowski',
                p=2
            )
        }
    
    def get_advanced_models(self) -> Dict[str, Any]:
        """Get advanced models with optimized hyperparameters"""
        return self.get_regularized_models()
    
    def create_preprocessing_pipeline(self, X: np.ndarray) -> Pipeline:
        """Create advanced preprocessing pipeline"""
        
        # Handle missing values, infinity, and outliers
        preprocessing_steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),  # More robust to outliers than StandardScaler
            ('power_transform', PowerTransformer(method='yeo-johnson')),
        ]
        
        return Pipeline(preprocessing_steps)
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess data to handle infinity and extreme values"""
        X = X.copy().astype(np.float64)
        
        # Replace infinity and very large values with NaN
        X = np.where(np.isinf(X), np.nan, X)
        X = np.where(np.abs(X) > 1e10, np.nan, X)
        
        # Fill NaN values with median
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        return X
    
    def feature_selection(self, X: np.ndarray, y: np.ndarray, method: str = 'mutual_info') -> Tuple[np.ndarray, List[int]]:
        """Advanced feature selection"""
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(50, X.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
        elif method == 'rfe':
            # Use Random Forest for RFE
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = RFE(estimator=rf, n_features_to_select=min(50, X.shape[1]))
        else:
            return X, list(range(X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        
        return X_selected, selected_features.tolist()
    
    def hyperparameter_tuning(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                            model_name: str, use_optuna: bool = True) -> Any:
        """Hyperparameter tuning using Optuna or RandomizedSearchCV"""
        
        if use_optuna:
            try:
                from src.hyper_tuning import run_random_search
                tuned_model, tuning_info = run_random_search(
                    model=model,
                    model_name=model_name,
                    X=X_train,
                    y=y_train,
                    n_iter=20,
                    cv_folds=3,
                    n_jobs=1,
                    random_state=self.random_state
                )
                return tuned_model
            except Exception as e:
                print(f"⚠️ Optuna tuning failed for {model_name}: {e}")
                print("🔄 Falling back to RandomizedSearchCV...")
        
        # Fallback to RandomizedSearchCV
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [6, 8, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 0.5, 0.7]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            return model
        
        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            model,
            param_grids[model_name],
            n_iter=15,  # Reduced iterations for speed
            cv=3,  # 3-fold CV for speed
            scoring='accuracy',
            random_state=self.random_state,
            n_jobs=1,  # Reduced parallelism
            verbose=0
        )
        
        search.fit(X_train, y_train)
        return search.best_estimator_
    
    def create_ensemble_model(self, models: Dict[str, Any]) -> VotingClassifier:
        """Create ensemble model using voting"""
        
        # Select best performing models for ensemble
        ensemble_models = []
        for name, model in models.items():
            if name in ['Random Forest', 'XGBoost', 'LightGBM', 'SVM']:
                ensemble_models.append((name, model))
        
        return VotingClassifier(
            estimators=ensemble_models,
            voting='soft',  # Use probabilities
            n_jobs=-1
        )
    
    def train_and_evaluate_advanced(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        use_feature_selection: bool = True,
        use_hyperparameter_tuning: bool = True,
        use_ensemble: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Train and evaluate advanced models with all improvements"""
        
        print("🚀 Starting Advanced AI Model Training...")
        
        # Step 1: Advanced Preprocessing
        print("📊 Applying advanced preprocessing...")
        
        # First, handle infinity and extreme values
        X_train_clean = self.preprocess_data(X_train)
        X_test_clean = self.preprocess_data(X_test)
        
        # Then apply advanced preprocessing pipeline
        preprocessing_pipeline = self.create_preprocessing_pipeline(X_train_clean)
        X_train_processed = preprocessing_pipeline.fit_transform(X_train_clean)
        X_test_processed = preprocessing_pipeline.transform(X_test_clean)
        
        # Step 2: Feature Selection
        if use_feature_selection:
            print("🔍 Performing feature selection...")
            X_train_selected, selected_features = self.feature_selection(
                X_train_processed, y_train, method='mutual_info'
            )
            X_test_selected = X_test_processed[:, selected_features]
            print(f"Selected {len(selected_features)} features from {X_train_processed.shape[1]}")
        else:
            X_train_selected = X_train_processed
            X_test_selected = X_test_processed
            selected_features = list(range(X_train_processed.shape[1]))
        
        # Step 3: Get Models
        models = self.get_advanced_models()
        
        # Step 4: Train Individual Models
        results = {}
        print("🤖 Training individual models...")
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Hyperparameter tuning
            if use_hyperparameter_tuning and name in ['Random Forest (Regularized)', 'XGBoost (Regularized)', 'LightGBM (Regularized)', 'SVM (Regularized)', 'Logistic Regression (L1)', 'Logistic Regression (L2)']:
                # Extract base name for tuning
                base_name = name.replace(' (Regularized)', '').replace(' (L1)', '').replace(' (L2)', '')
                model = self.hyperparameter_tuning(model, X_train_selected, y_train, base_name, use_optuna=False)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_selected, y_train, 
                cv=3, scoring='accuracy', n_jobs=1
            )
            
            # Train and predict
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test_selected)[:, 1]
                except:
                    y_proba = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'pred': y_pred,
                'proba': y_proba,
                'selected_features': selected_features
            }
            
            print(f"    {name}: Accuracy={accuracy:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # Step 5: Create Ensemble
        if use_ensemble:
            print("🎯 Creating ensemble model...")
            ensemble_model = self.create_ensemble_model(models)
            
            # Train ensemble
            ensemble_model.fit(X_train_selected, y_train)
            ensemble_pred = ensemble_model.predict(X_test_selected)
            ensemble_proba = ensemble_model.predict_proba(X_test_selected)[:, 1]
            
            # Evaluate ensemble
            ensemble_cv = cross_val_score(
                ensemble_model, X_train_selected, y_train, 
                cv=3, scoring='accuracy', n_jobs=1
            )
            
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted')
            ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted')
            ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
            ensemble_auc = roc_auc_score(y_test, ensemble_proba)
            
            results['Ensemble'] = {
                'model': ensemble_model,
                'accuracy': ensemble_accuracy,
                'precision': ensemble_precision,
                'recall': ensemble_recall,
                'f1': ensemble_f1,
                'auc': ensemble_auc,
                'cv_mean': ensemble_cv.mean(),
                'cv_std': ensemble_cv.std(),
                'pred': ensemble_pred,
                'proba': ensemble_proba,
                'selected_features': selected_features
            }
            
            print(f"    Ensemble: Accuracy={ensemble_accuracy:.4f}, CV={ensemble_cv.mean():.4f}±{ensemble_cv.std():.4f}")
        
        # Step 6: Select Best Model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
        
        return results


def process_mri_images(df_train, df_test):
    """Process MRI image data from parquet files"""
    try:
        import io
        from PIL import Image
        
        def extract_features_from_image(image_dict):
            """Extract features from image bytes using PIL only"""
            try:
                # Get image bytes
                image_bytes = image_dict['bytes']
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to numpy array
                img_array = np.array(image)
                
                # Handle different image formats (grayscale, RGB, etc.)
                if len(img_array.shape) == 3:
                    # RGB image - convert to grayscale
                    img_array = np.mean(img_array, axis=2)
                
                # Extract basic statistical features
                features = [
                    float(img_array.mean()),      # Mean intensity
                    float(img_array.std()),       # Standard deviation
                    float(img_array.min()),       # Min intensity
                    float(img_array.max()),       # Max intensity
                    float(img_array.shape[0]),    # Height
                    float(img_array.shape[1]),    # Width
                    float(np.median(img_array)),  # Median intensity
                    float(np.percentile(img_array, 25)),  # 25th percentile
                    float(np.percentile(img_array, 75)),  # 75th percentile
                ]
                
                # Add histogram features (10 bins)
                hist, _ = np.histogram(img_array.flatten(), bins=10, range=(0, 255))
                features.extend(hist.astype(float).tolist())
                
                # Add texture features (simple)
                # Calculate local standard deviation as texture measure
                if img_array.shape[0] > 2 and img_array.shape[1] > 2:
                    # Simple edge detection (gradient magnitude)
                    grad_x = np.diff(img_array, axis=1)
                    grad_y = np.diff(img_array, axis=0)
                    gradient_magnitude = np.sqrt(grad_x[:, :-1]**2 + grad_y[:-1, :]**2)
                    features.extend([
                        float(gradient_magnitude.mean()),
                        float(gradient_magnitude.std()),
                        float(gradient_magnitude.max())
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
                
                return features
                
            except Exception as e:
                print(f"    Warning: Image processing failed: {e}")
                # Return zeros if image processing fails
                return [0.0] * 22  # 9 basic + 10 histogram + 3 texture features
        
        print("🖼️ Processing MRI images with PIL...")
        
        # Process train images (sample first 1000 for speed)
        train_features = []
        max_train = min(1000, len(df_train))
        for idx in range(max_train):
            if idx % 200 == 0:
                print(f"  Processing train image {idx}/{max_train}")
            features = extract_features_from_image(df_train.iloc[idx]['image'])
            train_features.append(features)
        
        # Process test images (sample first 500 for speed)
        test_features = []
        max_test = min(500, len(df_test))
        for idx in range(max_test):
            if idx % 100 == 0:
                print(f"  Processing test image {idx}/{max_test}")
            features = extract_features_from_image(df_test.iloc[idx]['image'])
            test_features.append(features)
        
        # Convert to numpy arrays
        X_train = np.array(train_features)
        X_test = np.array(test_features)
        y_train = df_train['label'].iloc[:max_train].values
        y_test = df_test['label'].iloc[:max_test].values
        
        # Combine for full dataset
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        
        print(f"✅ Processed MRI data: X {X.shape}, y {y.shape}")
        print(f"📊 Features per image: {X.shape[1]}")
        return X, y
        
    except ImportError as e:
        print(f"⚠️ Required libraries not available: {e}")
        print("🔄 Using NPZ data instead...")
        return None, None
    except Exception as e:
        print(f"⚠️ Image processing failed: {e}")
        print("🔄 Using NPZ data instead...")
        return None, None


def load_real_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load real Alzheimer's data from available sources"""
    
    # Try parquet files first (MRI image data)
    try:
        import pandas as pd
        df_train = pd.read_parquet('data/raw/train.parquet')
        df_test = pd.read_parquet('data/raw/test.parquet')
        
        print(f"✅ Loaded parquet data: Train {df_train.shape}, Test {df_test.shape}")
        print(f"📊 Columns: {df_train.columns.tolist()}")
        print(f"🎯 Label distribution: {df_train['label'].value_counts().sort_index()}")
        
        # Try to process MRI images
        X, y = process_mri_images(df_train, df_test)
        if X is not None:
            return X, y
        else:
            print("🔄 Using NPZ data for preprocessed features...")
        
    except Exception as e:
        print(f"❌ Parquet loading failed: {e}")
        print("🔄 Falling back to NPZ data...")
    
    # Try NPZ file
    try:
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
            return X, y
            
    except Exception as e:
        print(f"❌ NPZ loading failed: {e}")
    
    # Fallback to sample data
    print("⚠️ Using sample data as fallback")
    np.random.seed(42)
    X = np.random.randn(1000, 50)
    y = np.random.choice([0, 1], 1000)
    
    return X, y
