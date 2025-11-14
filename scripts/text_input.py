"""
Writing/text input module: analyze grammar, coherence, and vocabulary.
Features: sentence complexity, vocabulary richness, coherence.
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_nlp_models():
    """Load spaCy and optional transformer models."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")
        nlp = None
    
    transformer_model = None
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        transformer_model = (tokenizer, model)
    except Exception as e:
        print(f"⚠️ Transformer model not loaded: {e}")
    
    return nlp, transformer_model


def extract_complexity_features(text: str, nlp) -> Dict[str, float]:
    """
    Extract sentence complexity features using spaCy.
    
    Args:
        text: Input text
        nlp: spaCy language model
    
    Returns:
        Dictionary with complexity features
    """
    if not nlp or not text:
        return {
            'avg_sentence_length': 0.0,
            'max_sentence_length': 0.0,
            'avg_dependency_depth': 0.0,
            'avg_words_per_sentence': 0.0,
            'sentence_count': 0.0
        }
    
    doc = nlp(text)
    sentences = list(doc.sents)
    
    if len(sentences) == 0:
        return {
            'avg_sentence_length': 0.0,
            'max_sentence_length': 0.0,
            'avg_dependency_depth': 0.0,
            'avg_words_per_sentence': 0.0,
            'sentence_count': 0.0
        }
    
    sentence_lengths = [len(sent) for sent in sentences]
    words_per_sentence = [len([t for t in sent if not t.is_punct]) for sent in sentences]
    
    # Compute dependency depth for each sentence
    depths = []
    for sent in sentences:
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 100:  # Safety limit
                    break
            depths.append(depth)
    
    avg_depth = np.mean(depths) if depths else 0.0
    
    return {
        'avg_sentence_length': float(np.mean(sentence_lengths)),
        'max_sentence_length': float(np.max(sentence_lengths)) if sentence_lengths else 0.0,
        'avg_dependency_depth': float(avg_depth),
        'avg_words_per_sentence': float(np.mean(words_per_sentence)) if words_per_sentence else 0.0,
        'sentence_count': float(len(sentences))
    }


def extract_vocabulary_features(text: str, nlp) -> Dict[str, float]:
    """
    Extract vocabulary richness features.
    
    Args:
        text: Input text
        nlp: spaCy language model
    
    Returns:
        Dictionary with vocabulary features
    """
    if not nlp or not text:
        return {
            'type_token_ratio': 0.0,
            'unique_words': 0.0,
            'total_words': 0.0,
            'avg_word_length': 0.0,
            'lexical_diversity': 0.0
        }
    
    doc = nlp(text)
    # Filter out punctuation and stop words for vocabulary analysis
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    
    if len(words) == 0:
        return {
            'type_token_ratio': 0.0,
            'unique_words': 0.0,
            'total_words': 0.0,
            'avg_word_length': 0.0,
            'lexical_diversity': 0.0
        }
    
    unique_words = len(set(words))
    total_words = len(words)
    type_token_ratio = unique_words / total_words if total_words > 0 else 0.0
    
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    
    # Lexical diversity: unique words per 100 words
    lexical_diversity = (unique_words / total_words * 100) if total_words > 0 else 0.0
    
    return {
        'type_token_ratio': float(type_token_ratio),
        'unique_words': float(unique_words),
        'total_words': float(total_words),
        'avg_word_length': float(avg_word_length),
        'lexical_diversity': float(lexical_diversity)
    }


def extract_coherence_features(text: str, transformer_model) -> Dict[str, float]:
    """
    Extract coherence features using transformer embeddings.
    
    Args:
        text: Input text
        transformer_model: Tuple of (tokenizer, model) or None
    
    Returns:
        Dictionary with coherence features
    """
    if not transformer_model or not text:
        return {
            'embedding_similarity': 0.0,
            'semantic_coherence': 0.0
        }
    
    try:
        tokenizer, model = transformer_model
        import torch
        
        # Split into sentences
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) < 2:
            return {
                'embedding_similarity': 0.0,
                'semantic_coherence': 0.0
            }
        
        # Get embeddings for each sentence
        embeddings = []
        for sent in sentences[:10]:  # Limit to 10 sentences
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        if len(embeddings) < 2:
            return {
                'embedding_similarity': 0.0,
                'semantic_coherence': 0.0
            }
        
        # Compute pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Average similarity (excluding diagonal)
        mask = ~np.eye(len(similarity_matrix), dtype=bool)
        avg_similarity = np.mean(similarity_matrix[mask])
        
        # Semantic coherence: consistency across sentences
        semantic_coherence = float(avg_similarity)
        
        return {
            'embedding_similarity': float(avg_similarity),
            'semantic_coherence': float(semantic_coherence)
        }
    except Exception as e:
        print(f"⚠️ Coherence extraction failed: {e}")
        return {
            'embedding_similarity': 0.0,
            'semantic_coherence': 0.0
        }


def process_text(
    text: str,
    subject_id: Optional[str] = None,
    use_transformer: bool = True
) -> Dict[str, float]:
    """
    Process a single text sample and extract all features.
    
    Args:
        text: Input text
        subject_id: Optional subject/patient ID
        use_transformer: Use transformer for coherence features
    
    Returns:
        Dictionary with all extracted features
    """
    nlp, transformer_model = load_nlp_models()
    
    if not nlp:
        print("⚠️ spaCy model not available. Returning minimal features.")
        return {
            'text_length': float(len(text)),
            'word_count': float(len(text.split())) if text else 0.0
        }
    
    # Extract features
    complexity_features = extract_complexity_features(text, nlp)
    vocab_features = extract_vocabulary_features(text, nlp)
    
    coherence_features = {}
    if use_transformer and transformer_model:
        coherence_features = extract_coherence_features(text, transformer_model)
    else:
        coherence_features = {
            'embedding_similarity': 0.0,
            'semantic_coherence': 0.0
        }
    
    # Combine all features
    all_features = {
        'text_length': float(len(text)),
        **complexity_features,
        **vocab_features,
        **coherence_features
    }
    
    if subject_id:
        all_features['subject_id'] = subject_id
    
    return all_features


def process_text_file(
    text_file_path: str,
    subject_id: Optional[str] = None
) -> Dict[str, float]:
    """
    Process a text file.
    
    Args:
        text_file_path: Path to text file
        subject_id: Optional subject ID
    
    Returns:
        Dictionary with features
    """
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return process_text(text, subject_id=subject_id)
    except Exception as e:
        print(f"⚠️ Failed to read text file: {e}")
        return {}


def process_text_directory(
    text_dir: str,
    output_path: str = "data/processed/text_features.csv"
) -> pd.DataFrame:
    """
    Process all text files in a directory.
    
    Args:
        text_dir: Directory containing text files
        output_path: Output CSV path
    
    Returns:
        DataFrame with features for all files
    """
    import glob
    
    text_files = glob.glob(os.path.join(text_dir, "*.txt")) + \
                 glob.glob(os.path.join(text_dir, "*.text"))
    
    print(f"📁 Found {len(text_files)} text files")
    
    all_features = []
    for i, text_file in enumerate(text_files):
        print(f"   Processing {i+1}/{len(text_files)}: {os.path.basename(text_file)}")
        subject_id = os.path.splitext(os.path.basename(text_file))[0]
        
        try:
            features = process_text_file(text_file, subject_id=subject_id)
            if features:
                all_features.append(features)
        except Exception as e:
            print(f"   ⚠️ Failed to process {text_file}: {e}")
    
    if not all_features:
        print("⚠️ No features extracted")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_features)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved text features to {output_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        text_file = sys.argv[1]
        features = process_text_file(text_file)
        print(features)
    else:
        print("Usage: python text_input.py <text_file_path>")

