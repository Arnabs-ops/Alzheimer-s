"""
Speech input module: record, transcribe, and extract audio features.
Features: pause duration, word repetition, pitch variation.
"""
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from typing import Optional, Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def transcribe_audio(audio_path: str, use_whisper: bool = True) -> str:
    """
    Transcribe audio using Whisper or Wav2Vec2.
    
    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        use_whisper: If True, use OpenAI Whisper; else Wav2Vec2
    
    Returns:
        Transcribed text
    """
    try:
        if use_whisper:
            import whisper
            model = whisper.load_model("base")  # base, small, medium, large
            result = model.transcribe(audio_path)
            return result["text"]
        else:
            # Fallback: Wav2Vec2 (requires transformers)
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch
            
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            return transcription
    except Exception as e:
        print(f"⚠️ Transcription failed: {e}")
        return ""


def extract_pause_features(audio_path: str, silence_threshold_db: float = -40.0) -> Dict[str, float]:
    """
    Extract pause duration features.
    
    Args:
        audio_path: Path to audio file
        silence_threshold_db: Threshold for silence detection (dB)
    
    Returns:
        Dictionary with pause features
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        if len(y) == 0 or duration == 0:
            print(f"   ⚠️ Empty audio file or zero duration")
            return {
                'total_pause_duration': 0.0,
                'mean_pause_duration': 0.0,
                'max_pause_duration': 0.0,
                'pause_count': 0.0,
                'pause_ratio': 0.0,
                'total_duration': 0.0
            }
        
        # Detect silence (non-speech segments)
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.power_to_db(rms**2, ref=np.max)
        silence_mask = rms_db < silence_threshold_db
        
        # Find silence segments
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                silence_start = i
                in_silence = True
            elif not is_silent and in_silence:
                silence_segments.append((silence_start, i))
                in_silence = False
        
        if in_silence:
            silence_segments.append((silence_start, len(silence_mask)))
        
        # Compute pause statistics
        pause_durations = []
        if silence_segments:
            frame_duration = len(y) / len(rms) / sr
            pause_durations = [(end - start) * frame_duration for start, end in silence_segments]
        
        return {
            'total_pause_duration': float(np.sum(pause_durations)) if pause_durations else 0.0,
            'mean_pause_duration': float(np.mean(pause_durations)) if pause_durations else 0.0,
            'max_pause_duration': float(np.max(pause_durations)) if pause_durations else 0.0,
            'pause_count': float(len(pause_durations)),
            'pause_ratio': float(np.sum(pause_durations) / duration) if duration > 0 else 0.0,
            'total_duration': float(duration)
        }
    except Exception as e:
        print(f"⚠️ Pause feature extraction failed: {e}")
        return {
            'total_pause_duration': 0.0,
            'mean_pause_duration': 0.0,
            'max_pause_duration': 0.0,
            'pause_count': 0.0,
            'pause_ratio': 0.0,
            'total_duration': 0.0
        }


def extract_repetition_features(transcription: str) -> Dict[str, float]:
    """
    Extract word repetition features from transcription.
    
    Args:
        transcription: Transcribed text
    
    Returns:
        Dictionary with repetition features
    """
    if not transcription:
        return {
            'repetition_ratio': 0.0,
            'unique_word_ratio': 0.0,
            'word_count': 0.0,
            'avg_word_length': 0.0
        }
    
    words = transcription.lower().split()
    word_count = len(words)
    
    if word_count == 0:
        return {
            'repetition_ratio': 0.0,
            'unique_word_ratio': 0.0,
            'word_count': 0.0,
            'avg_word_length': 0.0
        }
    
    unique_words = len(set(words))
    repetition_ratio = 1.0 - (unique_words / word_count) if word_count > 0 else 0.0
    
    # Check for immediate repetitions (same word twice in a row)
    immediate_repetitions = sum(1 for i in range(len(words) - 1) if words[i] == words[i+1])
    immediate_repetition_ratio = immediate_repetitions / max(word_count - 1, 1)
    
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    
    return {
        'repetition_ratio': float(repetition_ratio),
        'unique_word_ratio': float(unique_words / word_count),
        'word_count': float(word_count),
        'avg_word_length': float(avg_word_length),
        'immediate_repetition_ratio': float(immediate_repetition_ratio)
    }


def extract_pitch_features(audio_path: str) -> Dict[str, float]:
    """
    Extract pitch variation features.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with pitch features
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        if len(y) == 0:
            print(f"   ⚠️ Empty audio for pitch extraction")
            return {
                'mean_pitch': 0.0,
                'std_pitch': 0.0,
                'pitch_range': 0.0,
                'pitch_variation_coef': 0.0
            }
        
        # Extract fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Filter out unvoiced segments
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) == 0:
            return {
                'mean_pitch': 0.0,
                'std_pitch': 0.0,
                'pitch_range': 0.0,
                'pitch_variation_coef': 0.0
            }
        
        mean_pitch = np.mean(f0_voiced)
        std_pitch = np.std(f0_voiced)
        pitch_range = np.max(f0_voiced) - np.min(f0_voiced)
        variation_coef = std_pitch / mean_pitch if mean_pitch > 0 else 0.0
        
        return {
            'mean_pitch': float(mean_pitch),
            'std_pitch': float(std_pitch),
            'pitch_range': float(pitch_range),
            'pitch_variation_coef': float(variation_coef)
        }
    except Exception as e:
        print(f"⚠️ Pitch feature extraction failed: {e}")
        return {
            'mean_pitch': 0.0,
            'std_pitch': 0.0,
            'pitch_range': 0.0,
            'pitch_variation_coef': 0.0
        }


def process_audio_file(
    audio_path: str,
    subject_id: Optional[str] = None,
    use_whisper: bool = True
) -> Dict[str, float]:
    """
    Process a single audio file and extract all speech features.
    
    Args:
        audio_path: Path to audio file
        subject_id: Optional subject/patient ID
        use_whisper: Use Whisper for transcription
    
    Returns:
        Dictionary with all extracted features
    """
    print(f"🎤 Processing audio: {audio_path}")
    
    # Check if file exists and has content
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    print(f"   File size: {file_size} bytes")
    
    # Try to load and verify audio
    audio_loaded = False
    conversion_attempted = False
    
    try:
        y_test, sr_test = librosa.load(audio_path, sr=None, duration=1)
        print(f"   Audio loaded successfully: {len(y_test)} samples at {sr_test}Hz")
        if len(y_test) == 0:
            raise ValueError("Audio file has no audio data")
        audio_loaded = True
    except Exception as e:
        print(f"   ⚠️ Audio load failed (format issue?): {e}")
        
        # Try conversion using ffmpeg or pydub
        converted_path = None
        
        # Method 1: Try pydub (if available and ffmpeg is installed)
        try:
            from pydub import AudioSegment
            import subprocess
            
            # Check if ffmpeg is available (pydub needs it for WebM/Opus)
            ffmpeg_check = subprocess.run(['ffmpeg', '-version'], 
                                        capture_output=True, text=True, 
                                        timeout=2)
            has_ffmpeg = ffmpeg_check.returncode == 0
            
            if not has_ffmpeg:
                print("   ℹ️ pydub requires ffmpeg. Trying alternative methods...")
                raise ImportError("ffmpeg not found")
            
            print("   🔄 Converting audio using pydub (with ffmpeg)...")
            
            # Detect format from extension or file content
            file_ext = os.path.splitext(audio_path)[1].lower()
            
            # Try to detect actual format by reading file header
            actual_format = None
            try:
                with open(audio_path, 'rb') as f:
                    header = f.read(12)
                    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                        actual_format = 'wav'
                    elif header[:4] == b'fLaC':
                        actual_format = 'flac'
                    elif header[:4] == b'OggS':
                        actual_format = 'ogg'
                    elif b'webm' in header[:12].lower() or header[:4] == b'\x1a\x45\xdf\xa3':
                        actual_format = 'webm'
                    elif header[:3] == b'ID3' or header[:4] == b'\xff\xfb' or header[:4] == b'\xff\xf3':
                        actual_format = 'mp3'
            except:
                pass
            
            # Use detected format or fall back to extension
            if actual_format:
                print(f"   📋 Detected format: {actual_format}")
                try:
                    audio = AudioSegment.from_file(audio_path, format=actual_format)
                except:
                    # Try common alternatives
                    for fmt in ['webm', 'ogg', 'opus', 'wav']:
                        try:
                            audio = AudioSegment.from_file(audio_path, format=fmt)
                            break
                        except:
                            continue
            elif file_ext == '.wav':
                # Might be WebM masquerading as WAV, try WebM first
                try:
                    audio = AudioSegment.from_file(audio_path, format="webm")
                    print("   📋 File is actually WebM format")
                except:
                    try:
                        audio = AudioSegment.from_file(audio_path, format="ogg")
                        print("   📋 File is actually OGG format")
                    except:
                        audio = AudioSegment.from_file(audio_path, format="wav")
            else:
                audio = AudioSegment.from_file(audio_path)
            
            # Convert to WAV
            converted_path = audio_path.replace(file_ext, '_converted.wav')
            if not converted_path.endswith('.wav'):
                converted_path = audio_path + '_converted.wav'
            
            audio.export(converted_path, format="wav")
            
            if os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
                audio_path = converted_path
                print("   ✅ Audio converted with pydub")
                conversion_attempted = True
                
                # Try loading again
                y_test, sr_test = librosa.load(audio_path, sr=None, duration=1)
                audio_loaded = True
        except ImportError as ie:
            print(f"   ℹ️ pydub not available: {ie}")
        except FileNotFoundError:
            print("   ℹ️ ffmpeg not found in PATH. pydub requires ffmpeg for WebM/Opus conversion.")
        except Exception as e2:
            print(f"   ⚠️ pydub conversion failed: {e2}")
        
        # Method 2: Try ffmpeg if pydub didn't work
        if not audio_loaded:
            try:
                import subprocess
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("   🔄 Converting audio using ffmpeg...")
                    if converted_path is None:
                        converted_path = audio_path.replace('.wav', '_converted.wav')
                        # Also try different input formats
                        for fmt in ['webm', 'ogg', 'opus']:
                            try:
                                subprocess.run([
                                    'ffmpeg', '-i', audio_path, '-ar', '22050', '-ac', '1',
                                    '-f', 'wav', '-y', converted_path
                                ], capture_output=True, check=True)
                                break
                            except:
                                continue
                    
                    if os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
                        audio_path = converted_path
                        print("   ✅ Audio converted with ffmpeg")
                        conversion_attempted = True
                        
                        # Try loading again
                        y_test, sr_test = librosa.load(audio_path, sr=None, duration=1)
                        audio_loaded = True
                else:
                    raise FileNotFoundError("ffmpeg not found")
            except FileNotFoundError:
                print("   ⚠️ ffmpeg not available")
            except Exception as e2:
                print(f"   ⚠️ ffmpeg conversion failed: {e2}")
        
        if not audio_loaded:
            # Provide detailed installation instructions
            install_msg = (
                f"Could not load audio file. The browser recorded in WebM/Opus format, "
                f"which requires ffmpeg for conversion.\n\n"
                f"💡 **Solutions (choose one):**\n\n"
                f"**Option 1: Install ffmpeg (recommended)**\n"
                f"   1. Download ffmpeg from: https://www.gyan.dev/ffmpeg/builds/\n"
                f"   2. Extract and add to PATH, OR\n"
                f"   3. Use conda: conda install -c conda-forge ffmpeg\n\n"
                f"**Option 2: Use pre-recorded audio**\n"
                f"   Upload an existing WAV, MP3, or M4A file instead of recording.\n\n"
                f"**Option 3: Record in different browser**\n"
                f"   Some browsers may record in different formats.\n\n"
                f"Original error: {e}"
            )
            raise ValueError(install_msg)
    
    # Transcribe
    print("   Transcribing audio...")
    transcription = transcribe_audio(audio_path, use_whisper=use_whisper)
    print(f"   Transcription length: {len(transcription)} characters")
    
    # Extract features
    print("   Extracting pause features...")
    pause_features = extract_pause_features(audio_path)
    
    print("   Extracting repetition features...")
    repetition_features = extract_repetition_features(transcription)
    
    print("   Extracting pitch features...")
    pitch_features = extract_pitch_features(audio_path)
    
    # Combine all features
    all_features = {
        'transcription_length': float(len(transcription)),
        **pause_features,
        **repetition_features,
        **pitch_features
    }
    
    if subject_id:
        all_features['subject_id'] = subject_id
    
    print(f"   ✅ Features extracted: {len([k for k in all_features.keys() if k != 'subject_id'])} features")
    
    return all_features


def process_audio_directory(
    audio_dir: str,
    output_path: str = "data/processed/speech_features.csv",
    file_pattern: Optional[str] = None
) -> pd.DataFrame:
    """
    Process all audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files
        output_path: Output CSV path
        file_pattern: Optional pattern to match files (e.g., "*.wav")
    
    Returns:
        DataFrame with features for all files
    """
    import glob
    
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    files = []
    
    for ext in audio_extensions:
        pattern = os.path.join(audio_dir, f"**/*{ext}")
        files.extend(glob.glob(pattern, recursive=True))
    
    if file_pattern:
        files = [f for f in files if file_pattern in f]
    
    print(f"📁 Found {len(files)} audio files")
    
    all_features = []
    for i, audio_path in enumerate(files):
        print(f"   Processing {i+1}/{len(files)}: {os.path.basename(audio_path)}")
        subject_id = os.path.splitext(os.path.basename(audio_path))[0]
        
        try:
            features = process_audio_file(audio_path, subject_id=subject_id)
            all_features.append(features)
        except Exception as e:
            print(f"   ⚠️ Failed to process {audio_path}: {e}")
    
    if not all_features:
        print("⚠️ No features extracted")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_features)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved speech features to {output_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        features = process_audio_file(audio_path)
        print(features)
    else:
        print("Usage: python speech_input.py <audio_file_path>")

