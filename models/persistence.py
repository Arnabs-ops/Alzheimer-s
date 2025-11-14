"""
Model persistence utilities: save/load models consistently.
"""
import os
import joblib
import json
from typing import Any, Dict, Optional
from datetime import datetime


def save_model(
    model: Any,
    model_name: str,
    save_dir: str = "models/saved",
    metadata: Optional[Dict] = None
):
    """
    Save a trained model with metadata.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        save_dir: Directory to save model
        metadata: Optional metadata dict (metrics, training date, etc.)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved model to {model_path}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_metadata.json")
        metadata['saved_at'] = datetime.now().isoformat()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata to {metadata_path}")


def load_model(
    model_name: str,
    models_dir: str = "models/saved"
) -> Any:
    """
    Load a saved model.
    
    Args:
        model_name: Name of the model
        models_dir: Directory containing saved models
    
    Returns:
        Loaded model object
    """
    model_path = os.path.join(models_dir, f"{model_name.replace(' ', '_')}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"✅ Loaded model from {model_path}")
    return model


def load_model_metadata(
    model_name: str,
    models_dir: str = "models/saved"
) -> Optional[Dict]:
    """
    Load model metadata if available.
    
    Args:
        model_name: Name of the model
        models_dir: Directory containing saved models
    
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = os.path.join(models_dir, f"{model_name.replace(' ', '_')}_metadata.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def list_saved_models(models_dir: str = "models/saved") -> list:
    """List all saved models in directory."""
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.pkl'):
            model_name = file.replace('.pkl', '').replace('_', ' ')
            models.append(model_name)
    
    return sorted(models)

