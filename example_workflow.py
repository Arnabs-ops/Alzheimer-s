#!/usr/bin/env python3
"""
Example script demonstrating how to use the Alzheimer's Disease Analysis project
This script shows the basic workflow for loading data and running analysis
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append('src')

# Import our custom modules
from data_preprocessing import load_and_validate_data, detect_data_types
from models import AlzheimerModelPipeline
from visualization import plot_correlation_heatmap, plot_feature_distributions

def main():
    """
    Main function demonstrating the analysis workflow
    """
    print("🧠 Alzheimer's Disease Analysis - Example Workflow")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📁 Step 1: Loading Data")
    print("-" * 30)
    
    # Example data loading - replace with your actual data path
    data_path = "data/raw/your_dataset.csv"
    
    if os.path.exists(data_path):
        df = load_and_validate_data(data_path)
        if df is not None:
            print(f"✅ Data loaded successfully!")
            print(f"   Shape: {df.shape}")
        else:
            print("❌ Failed to load data")
            return
    else:
        print(f"⚠️  Data file not found: {data_path}")
        print("   Please download your dataset and update the path")
        return
    
    # Step 2: Data exploration
    print("\n🔍 Step 2: Data Exploration")
    print("-" * 30)
    
    numerical_cols, categorical_cols, datetime_cols = detect_data_types(df)
    
    # Basic statistics
    print(f"\n📊 Basic Statistics:")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicate rows: {df.duplicated().sum()}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Step 3: Visualization
    print("\n📈 Step 3: Data Visualization")
    print("-" * 30)
    
    if len(numerical_cols) > 1:
        print("   Creating correlation heatmap...")
        plot_correlation_heatmap(df)
    
    if len(numerical_cols) > 0:
        print("   Creating feature distribution plots...")
        plot_feature_distributions(df, numerical_cols[:6])
    
    # Step 4: Model training (example)
    print("\n🤖 Step 4: Model Training")
    print("-" * 30)
    
    # This is just an example - you'll need to implement actual preprocessing
    print("   ⚠️  Model training requires proper data preprocessing")
    print("   Please use the Jupyter notebook for complete analysis")
    
    print("\n✅ Example workflow completed!")
    print("\n📝 Next steps:")
    print("   1. Open notebooks/alzheimer_analysis.ipynb")
    print("   2. Update data loading paths with your actual dataset")
    print("   3. Run the notebook cells sequentially")
    print("   4. Customize the analysis based on your data")

if __name__ == "__main__":
    main()
