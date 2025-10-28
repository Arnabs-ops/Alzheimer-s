# Visualization Utilities
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_correlation_heatmap(df, figsize=(12, 10)):
    """
    Create correlation heatmap for numerical features
    """
    numerical_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=figsize)
    correlation_matrix = numerical_df.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df, columns=None, figsize=(15, 10)):
    """
    Plot distributions of specified features
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns[:6]  # First 6 numerical columns
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(columns):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def create_interactive_dashboard(df, target_col):
    """
    Create interactive dashboard using Plotly
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Target Distribution', 'Age Distribution', 
                       'Gender Distribution', 'Education Distribution'),
        specs=[[{"type": "pie"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Target distribution
    target_counts = df[target_col].value_counts()
    fig.add_trace(
        go.Pie(labels=target_counts.index, values=target_counts.values, name="Target"),
        row=1, col=1
    )
    
    # Age distribution (if age column exists)
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    if age_cols:
        fig.add_trace(
            go.Histogram(x=df[age_cols[0]], name="Age"),
            row=1, col=2
        )
    
    # Gender distribution (if gender column exists)
    gender_cols = [col for col in df.columns if 'gender' in col.lower() or 'sex' in col.lower()]
    if gender_cols:
        gender_counts = df[gender_cols[0]].value_counts()
        fig.add_trace(
            go.Bar(x=gender_counts.index, y=gender_counts.values, name="Gender"),
            row=2, col=1
        )
    
    # Education distribution (if education column exists)
    edu_cols = [col for col in df.columns if 'education' in col.lower() or 'edu' in col.lower()]
    if edu_cols:
        edu_counts = df[edu_cols[0]].value_counts()
        fig.add_trace(
            go.Bar(x=edu_counts.index, y=edu_counts.values, name="Education"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Alzheimer's Dataset Dashboard")
    fig.show()

def plot_model_performance_comparison(results):
    """
    Create comprehensive model performance comparison plots
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'auc', 'cv_mean']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names if results[name][metric] is not None]
        names = [name for name in model_names if results[name][metric] is not None]
        
        if values:
            bars = axes[i].bar(names, values, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
