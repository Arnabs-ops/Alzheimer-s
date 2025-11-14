"""
Generate synthetic data for testing the multimodal pipeline.
Creates realistic audio, text, and behavior logs.
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_speech_data(n_samples: int = 20, output_dir: str = "data/raw/synthetic"):
    """Generate synthetic speech features (simulated without actual audio)."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # Simulate features for AD vs control
    data = []
    for i in range(n_samples):
        label = 1 if i < n_samples // 2 else 0  # AD vs Control
        
        if label == 1:  # AD characteristics
            pause_duration = np.random.normal(5.0, 1.5)
            repetition_ratio = np.random.beta(3, 2)  # Higher repetition
            pitch_variation = np.random.normal(0.25, 0.05)  # Lower variation
        else:  # Control
            pause_duration = np.random.normal(2.5, 0.8)
            repetition_ratio = np.random.beta(2, 3)  # Lower repetition
            pitch_variation = np.random.normal(0.35, 0.08)  # Higher variation
        
        row = {
            'subject_id': f'patient_{i+1:03d}',
            'total_pause_duration': max(0, pause_duration),
            'mean_pause_duration': max(0, pause_duration * np.random.uniform(0.8, 1.2)),
            'max_pause_duration': max(0, pause_duration * np.random.uniform(1.5, 2.5)),
            'pause_count': int(np.random.poisson(10)),
            'pause_ratio': np.random.uniform(0.1, 0.4),
            'total_duration': np.random.uniform(30, 120),
            'repetition_ratio': repetition_ratio,
            'unique_word_ratio': 1.0 - repetition_ratio,
            'word_count': int(np.random.uniform(50, 200)),
            'avg_word_length': np.random.uniform(4, 6),
            'immediate_repetition_ratio': repetition_ratio * 0.3,
            'mean_pitch': np.random.uniform(100, 300),
            'std_pitch': np.random.uniform(20, 60),
            'pitch_range': np.random.uniform(50, 150),
            'pitch_variation_coef': pitch_variation,
            'transcription_length': int(np.random.uniform(200, 800)),
            'label': label
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, "speech_features_synthetic.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} synthetic speech samples to {output_path}")
    return df


def generate_synthetic_text_data(n_samples: int = 20, output_dir: str = "data/raw/synthetic"):
    """Generate synthetic text features."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        label = 1 if i < n_samples // 2 else 0
        
        if label == 1:  # AD characteristics
            avg_sentence_length = np.random.normal(8, 2)  # Shorter sentences
            type_token_ratio = np.random.normal(0.5, 0.1)  # Less vocabulary diversity
            semantic_coherence = np.random.normal(0.6, 0.1)  # Lower coherence
        else:  # Control
            avg_sentence_length = np.random.normal(15, 3)
            type_token_ratio = np.random.normal(0.7, 0.1)
            semantic_coherence = np.random.normal(0.85, 0.1)
        
        row = {
            'subject_id': f'patient_{i+1:03d}',
            'text_length': int(np.random.uniform(500, 2000)),
            'avg_sentence_length': max(1, avg_sentence_length),
            'max_sentence_length': int(avg_sentence_length * np.random.uniform(1.5, 2.5)),
            'avg_dependency_depth': np.random.uniform(3, 8),
            'avg_words_per_sentence': max(1, avg_sentence_length * 0.7),
            'sentence_count': int(np.random.uniform(10, 50)),
            'type_token_ratio': np.clip(type_token_ratio, 0, 1),
            'unique_words': int(np.random.uniform(50, 200)),
            'total_words': int(np.random.uniform(100, 400)),
            'avg_word_length': np.random.uniform(4, 6),
            'lexical_diversity': np.random.uniform(40, 80),
            'embedding_similarity': semantic_coherence,
            'semantic_coherence': semantic_coherence,
            'label': label
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, "text_features_synthetic.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} synthetic text samples to {output_path}")
    return df


def generate_synthetic_behavior_data(n_samples: int = 20, output_dir: str = "data/raw/synthetic"):
    """Generate synthetic behavior logs (time-series data)."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    all_logs = []
    
    for i in range(n_samples):
        subject_id = f'patient_{i+1:03d}'
        label = 1 if i < n_samples // 2 else 0
        
        # Generate 30 days of data
        start_date = datetime.now() - timedelta(days=30)
        dates = [start_date + timedelta(days=d) for d in range(30)]
        
        # Simulate activity score (0-1)
        if label == 1:  # AD: declining trend
            baseline = 0.7
            trend = -0.02  # Decline
            noise = 0.1
        else:  # Control: stable
            baseline = 0.8
            trend = 0.0
            noise = 0.05
        
        activity_scores = []
        for day, date in enumerate(dates):
            score = baseline + trend * day + np.random.normal(0, noise)
            score = np.clip(score, 0, 1)
            activity_scores.append(score)
        
        # Create log DataFrame
        log_data = {
            'date': dates,
            'activity_score': activity_scores,
            'medication_taken': np.random.binomial(1, 0.9, 30),
            'exercise_done': np.random.binomial(1, 0.6, 30),
            'sleep_hours': np.random.normal(7, 1, 30)
        }
        log_df = pd.DataFrame(log_data)
        
        # Save individual log
        log_path = os.path.join(output_dir, f"behavior_log_{subject_id}.csv")
        log_df.to_csv(log_path, index=False)
        
        # Extract features (simplified)
        features = {
            'subject_id': subject_id,
            'log_duration_days': 30,
            'total_records': 30,
            'routine_consistency': 1.0 / (1.0 + np.std(activity_scores)),
            'missing_days_ratio': 0.0,
            'activity_variance': float(np.var(activity_scores)),
            'trend_slope': float(trend),
            'trend_change': float(trend * 0.1),
            'seasonal_strength': float(np.std([activity_scores[i] for i in range(0, 30, 7)])),
            'anomaly_score': float(np.mean(np.abs(np.diff(activity_scores)))),
            'mean_value': float(np.mean(activity_scores)),
            'std_value': float(np.std(activity_scores)),
            'label': label
        }
        all_logs.append(features)
    
    # Save aggregated features
    features_df = pd.DataFrame(all_logs)
    features_path = os.path.join(output_dir, "behavior_features_synthetic.csv")
    features_df.to_csv(features_path, index=False)
    print(f"✅ Generated {len(features_df)} synthetic behavior logs to {output_dir}")
    print(f"✅ Aggregated features saved to {features_path}")
    return features_df


def generate_all_synthetic_data(n_samples: int = 20, output_dir: str = "data/raw/synthetic"):
    """Generate all synthetic data for testing."""
    print("🎲 Generating synthetic multimodal data...")
    
    speech_df = generate_synthetic_speech_data(n_samples, output_dir)
    text_df = generate_synthetic_text_data(n_samples, output_dir)
    behavior_df = generate_synthetic_behavior_data(n_samples, output_dir)
    
    # Combine features
    from scripts.features_common import combine_modality_features
    
    feature_files = {
        'speech': os.path.join(output_dir, 'speech_features_synthetic.csv'),
        'text': os.path.join(output_dir, 'text_features_synthetic.csv'),
        'behavior': os.path.join(output_dir, 'behavior_features_synthetic.csv')
    }
    
    combined_path = os.path.join(output_dir, 'combined_features_synthetic.csv')
    combined_df = combine_modality_features(feature_files, combined_path)
    
    print(f"\n✅ All synthetic data generated!")
    print(f"   - Speech: {len(speech_df)} samples")
    print(f"   - Text: {len(text_df)} samples")
    print(f"   - Behavior: {len(behavior_df)} samples")
    print(f"   - Combined: {len(combined_df)} samples")
    
    return combined_df


if __name__ == "__main__":
    # Generate synthetic data
    generate_all_synthetic_data(n_samples=50)

