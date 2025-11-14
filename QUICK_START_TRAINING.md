# Quick Start: Training Data

## Option 1: Use Synthetic Data (Recommended for Testing)

```powershell
# Generate 50 synthetic samples
python scripts/generate_synthetic_data.py
```

This creates:
- `data/raw/synthetic/speech_features_synthetic.csv` (50 samples)
- `data/raw/synthetic/text_features_synthetic.csv` (50 samples)
- `data/raw/synthetic/behavior_features_synthetic.csv` (50 samples)
- `data/raw/synthetic/combined_features_synthetic.csv` (50 combined samples)

## Option 2: Use Your Real Data

### Speech Data Format
Save as `data/raw/speech_features.csv` with columns:
```
subject_id, total_pause_duration, mean_pause_duration, max_pause_duration, 
pause_count, pause_ratio, total_duration, repetition_ratio, 
unique_word_ratio, word_count, avg_word_length, immediate_repetition_ratio,
mean_pitch, std_pitch, pitch_range, pitch_variation_coef, transcription_length, label
```

### Text Data Format
Save as `data/raw/text_features.csv` with columns:
```
subject_id, text_length, avg_sentence_length, max_sentence_length,
avg_dependency_depth, avg_words_per_sentence, sentence_count,
type_token_ratio, unique_words, total_words, avg_word_length,
lexical_diversity, embedding_similarity, semantic_coherence, label
```

### Behavior Data Format
Save as `data/raw/behavior_logs/patient_XXX.csv` (one file per patient) with columns:
```
date, activity_score, medication_taken, exercise_done, sleep_hours
```

## Option 3: Use Example Templates

Example files are in `data/raw/`:
- `example_speech_data.csv`
- `example_text_data.csv`
- `example_behavior_data.csv`

## Train Models

```powershell
python -c "
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.train_tabular import build_models, train_models

# Load combined features (or use synthetic)
df = pd.read_csv('data/raw/synthetic/combined_features_synthetic.csv')

# Prepare
X = df.drop(columns=['label', 'subject_id'], errors='ignore').select_dtypes(include=[np.number]).values
y = df['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
models = build_models()
results = train_models(X_train, y_train, X_test, y_test, models=models)
print('✅ Training complete!')
"
```

## Labels
- `label = 1`: Alzheimer's Disease / At Risk
- `label = 0`: Control / Normal

