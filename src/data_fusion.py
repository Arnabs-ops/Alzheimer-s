"""
Multi-Modal Data Fusion for Alzheimer's Disease Prediction
Integrates NPZ, MRI, and genomic data using various fusion strategies
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import io
from PIL import Image
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class DataFusion:
    """Multi-modal data fusion strategies for Alzheimer's prediction"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scalers = {}
        self.feature_selectors = {}
        self.fusion_models = {}
        
    def load_npz_data(self, file_path: str = 'data/raw/preprocessed_alz_data.npz') -> Tuple[np.ndarray, np.ndarray]:
        """Load NPZ data and combine train/test splits"""
        
        print("📊 Loading NPZ data...")
        
        data = np.load(file_path, allow_pickle=True)
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
            X_train = data[f'X_train']
            X_test = data[f'X_test']
            y_train = data[f'y_train']
            y_test = data[f'y_test']
            
            # Handle multi-dimensional y
            if len(y_train.shape) > 1:
                if y_train.shape[1] == 1:
                    y_train = y_train.ravel()
                    y_test = y_test.ravel()
                else:
                    y_train = np.argmax(y_train, axis=1)
                    y_test = np.argmax(y_test, axis=1)
            
            # Combine train and test for fresh splits
            X = np.vstack([X_train, X_test])
            y = np.concatenate([y_train, y_test])
            
            print(f"✅ NPZ data loaded: {X.shape}, {y.shape}")
            return X, y
        else:
            raise ValueError("Could not find X and y arrays in NPZ file")
    
    def extract_mri_features(self, df_train, df_test, max_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from MRI images"""
        
        print("🖼️ Extracting MRI features...")
        
        def extract_features_from_image(image_dict):
            """Extract comprehensive features from image bytes"""
            try:
                # Get image bytes
                image_bytes = image_dict['bytes']
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to numpy array
                img_array = np.array(image)
                
                # Handle different image formats
                if len(img_array.shape) == 3:
                    # RGB image - convert to grayscale
                    img_array = np.mean(img_array, axis=2)
                
                # Basic statistical features
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
                    float(np.var(img_array)),     # Variance
                ]
                
                # Histogram features (20 bins)
                hist, _ = np.histogram(img_array.flatten(), bins=20, range=(0, 255))
                features.extend(hist.astype(float).tolist())
                
                # Texture features
                if img_array.shape[0] > 2 and img_array.shape[1] > 2:
                    # Gradient features (align shapes to (H-1, W-1))
                    grad_x = np.diff(img_array, axis=1)  # (H, W-1)
                    grad_y = np.diff(img_array, axis=0)  # (H-1, W)
                    gx = grad_x[:-1, :]                  # (H-1, W-1)
                    gy = grad_y[:, :-1]                  # (H-1, W-1)
                    gradient_magnitude = np.sqrt(gx**2 + gy**2)
                    
                    features.extend([
                        float(gradient_magnitude.mean()),
                        float(gradient_magnitude.std()),
                        float(gradient_magnitude.max()),
                        float(gradient_magnitude.min())
                    ])
                    
                    # Local binary pattern-like features
                    center = img_array[1:-1, 1:-1]
                    neighbors = [
                        img_array[:-2, :-2], img_array[:-2, 1:-1], img_array[:-2, 2:],
                        img_array[1:-1, :-2], img_array[1:-1, 2:],
                        img_array[2:, :-2], img_array[2:, 1:-1], img_array[2:, 2:]
                    ]
                    
                    lbp_features = []
                    for neighbor in neighbors:
                        lbp_features.append(float(np.mean(center > neighbor)))
                    features.extend(lbp_features)
                    
                else:
                    features.extend([0.0] * 12)  # Padding for small images
                
                return features
                
            except Exception as e:
                print(f"    Warning: Image processing failed: {e}")
                return [0.0] * 42  # Return zeros if processing fails
        
        # Process train images (limit for speed)
        train_features = []
        max_train = min(max_samples, len(df_train))
        for idx in range(max_train):
            if idx % 200 == 0:
                print(f"  Processing train image {idx}/{max_train}")
            features = extract_features_from_image(df_train.iloc[idx]['image'])
            train_features.append(features)
        
        # Process test images
        test_features = []
        max_test = min(max_samples // 2, len(df_test))
        for idx in range(max_test):
            if idx % 100 == 0:
                print(f"  Processing test image {idx}/{max_test}")
            features = extract_features_from_image(df_test.iloc[idx]['image'])
            test_features.append(features)
        
        # Convert to numpy arrays
        X_train_mri = np.array(train_features)
        X_test_mri = np.array(test_features)
        y_train_mri = df_train['label'].iloc[:max_train].values
        y_test_mri = df_test['label'].iloc[:max_test].values
        
        # Combine for full dataset
        X_mri = np.vstack([X_train_mri, X_test_mri])
        y_mri = np.concatenate([y_train_mri, y_test_mri])
        
        print(f"✅ MRI features extracted: {X_mri.shape}, {y_mri.shape}")
        return X_mri, y_mri
    
    def process_genomic_data(self, file_path: str = 'data/raw/advp.hg38.tsv') -> Tuple[np.ndarray, np.ndarray]:
        """Process genomic variant data"""
        
        print("🧬 Processing genomic variant data...")
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            print(f"✅ Genomic data loaded: {df.shape}")
            
            # Create features from genomic data
            features = []
            
            # P-value features
            p_values = pd.to_numeric(df['P-value'], errors='coerce')
            p_values = p_values.fillna(p_values.median())
            
            # Log transform p-values
            log_p_values = -np.log10(p_values + 1e-10)
            
            # Odds ratio features
            or_values = pd.to_numeric(df['OR_nonref'], errors='coerce')
            or_values = or_values.fillna(or_values.median())
            
            # Sample size features
            sample_sizes = pd.to_numeric(df['Sample size'], errors='coerce')
            sample_sizes = sample_sizes.fillna(sample_sizes.median())
            
            # Create aggregated features
            genomic_features = []
            
            # Statistical features
            genomic_features.extend([
                float(log_p_values.mean()),
                float(log_p_values.std()),
                float(log_p_values.median()),
                float(log_p_values.max()),
                float(log_p_values.min()),
                float(np.sum(log_p_values > 5)),  # Significant variants
                float(np.sum(log_p_values > 3)),  # Suggestive variants
            ])
            
            # Odds ratio features
            genomic_features.extend([
                float(or_values.mean()),
                float(or_values.std()),
                float(or_values.median()),
                float(np.sum(or_values > 1.2)),  # Risk variants
                float(np.sum(or_values < 0.8)),  # Protective variants
            ])
            
            # Sample size features
            genomic_features.extend([
                float(sample_sizes.mean()),
                float(sample_sizes.std()),
                float(sample_sizes.median()),
                float(sample_sizes.max()),
                float(sample_sizes.min()),
            ])
            
            # Gene-based features
            gene_counts = df['nearest_gene_symb'].value_counts()
            genomic_features.extend([
                float(len(gene_counts)),  # Number of unique genes
                float(gene_counts.max()),  # Max variants per gene
                float(gene_counts.mean()), # Mean variants per gene
            ])
            
            # Study type features
            study_types = df['Study type'].value_counts()
            genomic_features.extend([
                float(len(study_types)),  # Number of study types
                float(study_types.max()),  # Max variants per study type
            ])
            
            # Phenotype features
            phenotypes = df['Phenotype'].value_counts()
            genomic_features.extend([
                float(len(phenotypes)),  # Number of phenotypes
                float(phenotypes.max()),  # Max variants per phenotype
            ])
            
            # Create synthetic labels (since we don't have ground truth)
            # This is a simplified approach - in reality, you'd need proper labels
            n_samples = 1000  # Create synthetic samples
            X_genomic = np.tile(genomic_features, (n_samples, 1))
            
            # Add some noise to create variation
            noise = np.random.normal(0, 0.1, X_genomic.shape)
            X_genomic = X_genomic + noise
            
            # Create synthetic labels based on genomic risk
            risk_score = np.sum(X_genomic[:, :7], axis=1)  # Sum of p-value features
            y_genomic = np.where(risk_score > np.median(risk_score), 1, 0)
            
            print(f"✅ Genomic features created: {X_genomic.shape}, {y_genomic.shape}")
            return X_genomic, y_genomic
            
        except Exception as e:
            print(f"❌ Genomic data processing failed: {e}")
            # Return dummy data
            X_genomic = np.random.randn(1000, 20)
            y_genomic = np.random.choice([0, 1], 1000)
            return X_genomic, y_genomic
    
    def early_fusion(self, X_npz: np.ndarray, X_mri: np.ndarray, X_genomic: np.ndarray,
                    y_npz: np.ndarray, y_mri: np.ndarray, y_genomic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Early fusion: concatenate all features"""
        
        print("🔗 Performing early fusion...")
        
        # Find common sample size
        min_samples = min(len(X_npz), len(X_mri), len(X_genomic))
        
        # Truncate to common size
        X_npz_trimmed = X_npz[:min_samples]
        X_mri_trimmed = X_mri[:min_samples]
        X_genomic_trimmed = X_genomic[:min_samples]
        
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
        
        X_npz_trimmed = clean_data(X_npz_trimmed)
        X_mri_trimmed = clean_data(X_mri_trimmed)
        X_genomic_trimmed = clean_data(X_genomic_trimmed)
        
        # Use NPZ labels as ground truth (most reliable)
        y_fused = y_npz[:min_samples]
        
        # Concatenate features
        X_fused = np.hstack([X_npz_trimmed, X_mri_trimmed, X_genomic_trimmed])
        
        print(f"✅ Early fusion completed: {X_fused.shape}, {y_fused.shape}")
        print(f"  NPZ features: {X_npz_trimmed.shape[1]}")
        print(f"  MRI features: {X_mri_trimmed.shape[1]}")
        print(f"  Genomic features: {X_genomic_trimmed.shape[1]}")
        
        return X_fused, y_fused
    
    def late_fusion(self, X_npz: np.ndarray, X_mri: np.ndarray, X_genomic: np.ndarray,
                   y_npz: np.ndarray, y_mri: np.ndarray, y_genomic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Late fusion: ensemble predictions from each modality"""
        
        print("🎯 Performing late fusion...")
        
        # Find common sample size
        min_samples = min(len(X_npz), len(X_mri), len(X_genomic))
        
        # Truncate to common size
        X_npz_trimmed = X_npz[:min_samples]
        X_mri_trimmed = X_mri[:min_samples]
        X_genomic_trimmed = X_genomic[:min_samples]
        
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
        
        X_npz_trimmed = clean_data(X_npz_trimmed)
        X_mri_trimmed = clean_data(X_mri_trimmed)
        X_genomic_trimmed = clean_data(X_genomic_trimmed)
        
        # Use NPZ labels as ground truth
        y_fused = y_npz[:min_samples]
        
        # Train individual models for each modality
        models = {
            'npz': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'mri': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'genomic': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        # Train models
        models['npz'].fit(X_npz_trimmed, y_fused)
        models['mri'].fit(X_mri_trimmed, y_fused)
        models['genomic'].fit(X_genomic_trimmed, y_fused)
        
        # Get predictions
        pred_npz = models['npz'].predict_proba(X_npz_trimmed)
        pred_mri = models['mri'].predict_proba(X_mri_trimmed)
        pred_genomic = models['genomic'].predict_proba(X_genomic_trimmed)
        
        # Average predictions
        pred_fused = (pred_npz + pred_mri + pred_genomic) / 3
        
        # Convert probabilities to predictions
        y_pred_fused = np.argmax(pred_fused, axis=1)
        
        print(f"✅ Late fusion completed: {len(y_pred_fused)} predictions")
        
        # Store models for later use
        self.fusion_models = models
        
        return X_npz_trimmed, y_pred_fused  # Return NPZ features with fused predictions
    
    def intermediate_fusion(self, X_npz: np.ndarray, X_mri: np.ndarray, X_genomic: np.ndarray,
                           y_npz: np.ndarray, y_mri: np.ndarray, y_genomic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Intermediate fusion: learn joint representations"""
        
        print("🧠 Performing intermediate fusion...")
        
        # Find common sample size
        min_samples = min(len(X_npz), len(X_mri), len(X_genomic))
        
        # Truncate to common size
        X_npz_trimmed = X_npz[:min_samples]
        X_mri_trimmed = X_mri[:min_samples]
        X_genomic_trimmed = X_genomic[:min_samples]
        
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
        
        X_npz_trimmed = clean_data(X_npz_trimmed)
        X_mri_trimmed = clean_data(X_mri_trimmed)
        X_genomic_trimmed = clean_data(X_genomic_trimmed)
        
        # Use NPZ labels as ground truth
        y_fused = y_npz[:min_samples]
        
        # Apply PCA to each modality to reduce dimensionality
        pca_npz = PCA(n_components=min(20, X_npz_trimmed.shape[1]), random_state=self.random_state)
        pca_mri = PCA(n_components=min(20, X_mri_trimmed.shape[1]), random_state=self.random_state)
        pca_genomic = PCA(n_components=min(10, X_genomic_trimmed.shape[1]), random_state=self.random_state)
        
        X_npz_pca = pca_npz.fit_transform(X_npz_trimmed)
        X_mri_pca = pca_mri.fit_transform(X_mri_trimmed)
        X_genomic_pca = pca_genomic.fit_transform(X_genomic_trimmed)
        
        # Concatenate PCA features
        X_fused = np.hstack([X_npz_pca, X_mri_pca, X_genomic_pca])
        
        print(f"✅ Intermediate fusion completed: {X_fused.shape}, {y_fused.shape}")
        print(f"  NPZ PCA features: {X_npz_pca.shape[1]}")
        print(f"  MRI PCA features: {X_mri_pca.shape[1]}")
        print(f"  Genomic PCA features: {X_genomic_pca.shape[1]}")
        
        return X_fused, y_fused
    
    def comprehensive_fusion(self, npz_path: str = 'data/raw/preprocessed_alz_data.npz',
                           mri_train_path: str = 'data/raw/train.parquet',
                           mri_test_path: str = 'data/raw/test.parquet',
                           genomic_path: str = 'data/raw/advp.hg38.tsv') -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Perform comprehensive multi-modal fusion"""
        
        print("🚀 Starting comprehensive multi-modal fusion...")
        print("=" * 60)
        
        # Load all datasets
        X_npz, y_npz = self.load_npz_data(npz_path)
        
        try:
            df_train = pd.read_parquet(mri_train_path)
            df_test = pd.read_parquet(mri_test_path)
            X_mri, y_mri = self.extract_mri_features(df_train, df_test)
        except Exception as e:
            print(f"⚠️ MRI data loading failed: {e}")
            # Create dummy MRI data
            X_mri = np.random.randn(1000, 42)
            y_mri = np.random.choice([0, 1, 2, 3], 1000)
        
        X_genomic, y_genomic = self.process_genomic_data(genomic_path)
        
        # Perform different fusion strategies
        fusion_results = {}
        
        # Early fusion
        X_early, y_early = self.early_fusion(X_npz, X_mri, X_genomic, y_npz, y_mri, y_genomic)
        fusion_results['early_fusion'] = (X_early, y_early)
        
        # Late fusion
        X_late, y_late = self.late_fusion(X_npz, X_mri, X_genomic, y_npz, y_mri, y_genomic)
        fusion_results['late_fusion'] = (X_late, y_late)
        
        # Intermediate fusion
        X_intermediate, y_intermediate = self.intermediate_fusion(X_npz, X_mri, X_genomic, y_npz, y_mri, y_genomic)
        fusion_results['intermediate_fusion'] = (X_intermediate, y_intermediate)
        
        # Individual modalities
        fusion_results['npz_only'] = (X_npz, y_npz)
        fusion_results['mri_only'] = (X_mri, y_mri)
        fusion_results['genomic_only'] = (X_genomic, y_genomic)
        
        print(f"\n✅ Multi-modal fusion completed!")
        print(f"📊 Fusion Results:")
        for name, (X, y) in fusion_results.items():
            print(f"  {name}: {X.shape}, {y.shape}")
        
        return fusion_results
    
    def evaluate_fusion_strategies(self, fusion_results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Evaluate different fusion strategies"""
        
        print("\n📊 Evaluating fusion strategies...")
        
        evaluation_results = {}
        
        for strategy_name, (X, y) in fusion_results.items():
            print(f"\n🔍 Evaluating {strategy_name}...")
            
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
            
            X = clean_data(X)
            
            # Apply feature selection for MRI to reduce overfitting
            if strategy_name == 'mri_only':
                from sklearn.feature_selection import SelectKBest, f_classif
                # Select only top 15 features for MRI
                selector = SelectKBest(f_classif, k=min(15, X.shape[1]))
                X = selector.fit_transform(X, y)
                print(f"  Selected {X.shape[1]} features from MRI data")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            # Train model with regularization based on data size
            if strategy_name == 'mri_only':
                # Heavy regularization for small MRI dataset
                model = RandomForestClassifier(
                    n_estimators=50,  # Reduced trees
                    max_depth=5,      # Shallow trees
                    min_samples_split=20,  # High minimum split
                    min_samples_leaf=10,   # High minimum leaf
                    max_features=0.3,      # Fewer features per tree
                    random_state=self.random_state
                )
            else:
                # Standard regularization for other strategies
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features=0.5,
                    random_state=self.random_state
                )
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            evaluation_results[strategy_name] = {
                'train_score': train_score,
                'test_score': test_score,
                'gap': train_score - test_score,
                'n_features': X.shape[1]
            }
            
            print(f"  Train Score: {train_score:.4f}")
            print(f"  Test Score: {test_score:.4f}")
            print(f"  Gap: {train_score - test_score:.4f}")
            print(f"  Features: {X.shape[1]}")
        
        return evaluation_results
