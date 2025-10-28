# Feature Engineering Utilities
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def create_interaction_features(df, feature_pairs):
    """
    Create interaction features between specified pairs
    """
    df_new = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df_new[interaction_name] = df[feat1] * df[feat2]
    
    return df_new

def create_polynomial_features(df, degree=2, include_bias=False):
    """
    Create polynomial features for numerical columns
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_features = poly.fit_transform(df[numerical_cols])
    
    # Create feature names
    feature_names = poly.get_feature_names_out(numerical_cols)
    
    # Create new DataFrame
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    return poly_df

def create_age_groups(df, age_col, bins=[0, 50, 65, 80, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly']):
    """
    Create age group categories
    """
    df_new = df.copy()
    df_new[f'{age_col}_group'] = pd.cut(df[age_col], bins=bins, labels=labels, include_lowest=True)
    return df_new

def create_bmi_categories(df, height_col, weight_col):
    """
    Create BMI categories from height and weight
    """
    df_new = df.copy()
    
    # Calculate BMI (assuming height in cm, weight in kg)
    df_new['BMI'] = df[weight_col] / ((df[height_col] / 100) ** 2)
    
    # Create BMI categories
    df_new['BMI_category'] = pd.cut(df_new['BMI'], 
                                   bins=[0, 18.5, 25, 30, 100], 
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                                   include_lowest=True)
    
    return df_new

def select_top_features(X, y, k=20, method='f_classif'):
    """
    Select top k features using statistical tests
    """
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("Method must be 'f_classif' or 'mutual_info'")
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X_selected, selected_features, selector
