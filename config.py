# Configuration file for Alzheimer's Disease Analysis Project

# Data paths
DATA_PATHS = {
    'raw': '../data/raw/',
    'processed': '../data/processed/',
    'external': '../data/external/',
    'results': '../results/'
}

# Model parameters
MODEL_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_estimators': 100,
    'max_iter': 1000
}

# Feature selection
FEATURE_SELECTION = {
    'top_k_features': 20,
    'correlation_threshold': 0.95,
    'variance_threshold': 0.01
}

# Visualization settings
PLOT_SETTINGS = {
    'figsize_large': (15, 10),
    'figsize_medium': (12, 8),
    'figsize_small': (8, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8'
}

# File formats
SUPPORTED_FORMATS = ['csv', 'excel', 'json', 'parquet']

# Target variable (update based on your dataset)
TARGET_COLUMN = 'diagnosis'  # Update this based on your actual target column name

# Common Alzheimer's related features (update based on your dataset)
ALZHEIMER_FEATURES = {
    'demographic': ['age', 'gender', 'education', 'marital_status'],
    'cognitive': ['mmse_score', 'cdr_score', 'cognitive_tests'],
    'biomarker': ['apoe_genotype', 'csf_proteins', 'imaging_markers'],
    'clinical': ['family_history', 'medications', 'comorbidities']
}
