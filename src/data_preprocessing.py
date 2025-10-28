# Data Preprocessing Utilities
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_validate_data(file_path, file_type='csv'):
    """
    Load data from various file formats with validation
    """
    try:
        if file_type.lower() == 'csv':
            df = pd.read_csv(file_path)
        elif file_type.lower() == 'excel':
            df = pd.read_excel(file_path)
        elif file_type.lower() == 'json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        print(f"✅ Data loaded successfully from {file_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def detect_data_types(df):
    """
    Automatically detect and categorize data types
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    print("📊 Data Type Detection:")
    print(f"   Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"   Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"   Datetime columns ({len(datetime_cols)}): {datetime_cols}")
    
    return numerical_cols, categorical_cols, datetime_cols

def create_data_summary(df):
    """
    Create comprehensive data summary
    """
    summary = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.value_counts().to_dict()
    }
    
    return summary
