"""
Streamlit dashboard: end-to-end multimodal Alzheimer's risk assessment.
Allows upload/simulation of inputs, shows predictions, risk scores, and SHAP visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.speech_input import process_audio_file
from scripts.text_input import process_text
from scripts.behavior_input import process_behavior_log
from scripts.features_common import combine_modality_features, normalize_features
from models.train_tabular import build_models, load_trained_model, train_models
from models.fusion import fuse_modality_predictions
from models.evaluate import evaluate_model, plot_roc_curve, plot_confusion_matrix, plot_shap_summary
from models.persistence import list_saved_models, load_model_metadata

# Page config
st.set_page_config(
    page_title="Alzheimer's Early Warning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'speech_features' not in st.session_state:
    st.session_state.speech_features = None
if 'text_features' not in st.session_state:
    st.session_state.text_features = None
if 'behavior_features' not in st.session_state:
    st.session_state.behavior_features = None


def main():
    # Sidebar with improved navigation
    st.sidebar.title("🧠 Navigation")
    st.sidebar.markdown("---")
    
    # Quick status indicators
    st.sidebar.markdown("### Status")
    status_col1, status_col2 = st.sidebar.columns(2)
    with status_col1:
        speech_status = "✅" if st.session_state.speech_features else "❌"
        st.markdown(f"Speech: {speech_status}")
    with status_col2:
        text_status = "✅" if st.session_state.text_features else "❌"
        st.markdown(f"Text: {text_status}")
    
    behavior_status = "✅" if st.session_state.behavior_features else "❌"
    st.sidebar.markdown(f"Behavior: {behavior_status}")
    
    st.sidebar.markdown("---")
    
    # Navigation buttons (better UX than selectbox)
    page = st.sidebar.radio(
        "Go to:",
        ["🏠 Home", "📊 Data Input", "🤖 Train Models", "🔀 Predictions", "📈 Results"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    if st.sidebar.button("🔄 Clear All Data"):
        st.session_state.speech_features = None
        st.session_state.text_features = None
        st.session_state.behavior_features = None
        st.session_state.predictions = {}
        st.success("✅ Data cleared!")
        st.rerun()
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Input":
        show_data_input()
    elif page == "🤖 Train Models":
        show_model_training()
    elif page == "🔀 Predictions":
        show_fusion()
    elif page == "📈 Results":
        show_results()


def show_home():
    """Home page with overview."""
    st.title("🧠 Alzheimer's Disease Early Warning System")
    st.markdown("Multimodal AI pipeline: Speech + Writing + Behavior analysis")
    st.markdown("---")
    st.header("Overview")
    st.markdown("""
    This system analyzes three modalities to detect early signs of Alzheimer's:
    
    ### 🗣️ Speech Module
    - Transcribes audio using Whisper
    - Extracts: pause duration, word repetition, pitch variation
    
    ### ✍️ Writing Module
    - Analyzes text using spaCy and transformers
    - Extracts: sentence complexity, vocabulary richness, coherence
    
    ### 🧠 Behavior Module
    - Tracks routines using time-series analysis (Prophet)
    - Extracts: trend changes, anomalies, routine consistency
    
    ### 🔀 Multimodal Fusion
    - Combines predictions using late fusion (weighted average or meta-learner)
    - Generates final risk score
    """)
    
    st.header("Quick Start")
    st.markdown("""
    1. **Data Input**: Upload or simulate speech/text/behavior data
    2. **Model Training**: Train models on your data (or load pre-trained)
    3. **Fusion**: Combine modality predictions
    4. **Results**: View predictions, risk scores, and SHAP explanations
    """)


def show_data_input():
    """Data input page with improved UI."""
    st.header("📊 Data Input")
    st.markdown("Record or upload your data for each modality below.")
    
    # Show status at top
    col1, col2, col3 = st.columns(3)
    with col1:
        speech_status = "✅ Complete" if st.session_state.speech_features else "⏳ Pending"
        st.metric("Speech", speech_status)
    with col2:
        text_status = "✅ Complete" if st.session_state.text_features else "⏳ Pending"
        st.metric("Text", text_status)
    with col3:
        behavior_status = "✅ Complete" if st.session_state.behavior_features else "⏳ Pending"
        st.metric("Behavior", behavior_status)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🗣️ Speech", "✍️ Writing", "🧠 Behavior"])
    
    with tab1:
        st.subheader("🗣️ Speech Audio Input")
        st.markdown("Record audio using your microphone or upload an existing file.")
        
        # Two columns: Record vs Upload
        col_record, col_upload = st.columns(2)
        
        with col_record:
            st.markdown("### 🎤 Record Audio")
            st.markdown("Record audio using your browser's microphone:")
            
            # Custom audio recorder component
            try:
                from dashboard.components.audio_recorder import audio_recorder_component
                audio_recorder_component()
                
                st.markdown("---")
                st.markdown("**After recording:**")
                st.markdown("1. Click 'Download Audio' in the recorder above")
                st.markdown("2. Then upload the downloaded file using the upload option on the right")
                st.info("💡 **Tip:** You can also use the upload option directly if you have a pre-recorded file.")
                
            except ImportError:
                # Simplified HTML5 audio recorder
                st.markdown("**Click to start recording:**")
                
                audio_html = """
                <div style="padding: 15px; border: 2px solid #6aa36f; border-radius: 8px;">
                    <button id="startBtn" style="padding: 12px 24px; font-size: 16px; background-color: #ff4b4b; color: white; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                        🎤 Start Recording
                    </button>
                    <button id="stopBtn" style="padding: 12px 24px; font-size: 16px; background-color: #6aa36f; color: white; border: none; border-radius: 5px; cursor: pointer;" disabled>
                        ⏹️ Stop
                    </button>
                    <div id="status" style="margin-top: 15px; font-weight: bold;"></div>
                    <div id="audioContainer" style="margin-top: 15px;"></div>
                    <div id="downloadContainer" style="margin-top: 15px;"></div>
                </div>
                <script>
                    let mediaRecorder;
                    let audioChunks = [];
                    
                           document.getElementById('startBtn').addEventListener('click', async () => {
                               try {
                                   const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                                   
                                   // Try to use WAV format if supported, otherwise fall back to default
                                   let mimeType = 'audio/wav';
                                   if (!MediaRecorder.isTypeSupported('audio/wav')) {
                                       // Try other formats
                                       if (MediaRecorder.isTypeSupported('audio/webm')) {
                                           mimeType = 'audio/webm';
                                       } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
                                           mimeType = 'audio/ogg';
                                       } else {
                                           mimeType = ''; // Use browser default
                                       }
                                   }
                                   
                                   mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
                                   audioChunks = [];
                            
                            mediaRecorder.ondataavailable = event => {
                                audioChunks.push(event.data);
                            };
                            
                            mediaRecorder.onstop = () => {
                                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                                const audioUrl = URL.createObjectURL(audioBlob);
                                const audio = document.createElement('audio');
                                audio.src = audioUrl;
                                audio.controls = true;
                                audio.style.width = '100%';
                                
                                document.getElementById('audioContainer').innerHTML = '';
                                document.getElementById('audioContainer').appendChild(audio);
                                document.getElementById('status').innerHTML = '<span style="color: green;">✅ Recording saved! Download below, then upload it.</span>';
                                
                                const downloadLink = document.createElement('a');
                                downloadLink.href = audioUrl;
                                downloadLink.download = 'recording.wav';
                                downloadLink.textContent = '📥 Download Recording';
                                downloadLink.style.display = 'inline-block';
                                downloadLink.style.padding = '10px 20px';
                                downloadLink.style.backgroundColor = '#1f77b4';
                                downloadLink.style.color = 'white';
                                downloadLink.style.textDecoration = 'none';
                                downloadLink.style.borderRadius = '5px';
                                downloadLink.style.marginTop = '10px';
                                document.getElementById('downloadContainer').innerHTML = '';
                                document.getElementById('downloadContainer').appendChild(downloadLink);
                            };
                            
                            mediaRecorder.start();
                            document.getElementById('startBtn').disabled = true;
                            document.getElementById('stopBtn').disabled = false;
                            document.getElementById('status').innerHTML = '<span style="color: red;">🔴 Recording... Click Stop when done.</span>';
                        } catch (err) {
                            document.getElementById('status').innerHTML = '<span style="color: red;">❌ Error: ' + err.message + '</span>';
                            alert('Could not access microphone. Please allow microphone access.');
                        }
                    });
                    
                    document.getElementById('stopBtn').addEventListener('click', () => {
                        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                            mediaRecorder.stop();
                            mediaRecorder.stream.getTracks().forEach(track => track.stop());
                            document.getElementById('startBtn').disabled = false;
                            document.getElementById('stopBtn').disabled = true;
                        }
                    });
                </script>
                """
                st.components.v1.html(audio_html, height=250)
                
                st.info("💡 **After recording:** Download the audio file, then upload it using the upload section on the right.")
        
        with col_upload:
            st.markdown("### 📁 Upload Audio File")
            audio_file = st.file_uploader(
                "Upload audio file (WAV, MP3, M4A, etc.)", 
                type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
                key="audio_upload"
            )
            
            if audio_file:
                st.audio(audio_file, format=f"audio/{os.path.splitext(audio_file.name)[1][1:]}")
                
                subject_id_speech = st.text_input("Subject ID (optional)", key="speech_id_upload")
                
                if st.button("Process Uploaded File", key="process_uploaded"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    
                    try:
                        with st.spinner("Processing audio..."):
                            # Show file info
                            file_size = os.path.getsize(tmp_path)
                            st.info(f"📁 File: {audio_file.name} ({file_size:,} bytes)")
                            
                            # Process audio
                            features = process_audio_file(tmp_path, subject_id=subject_id_speech)
                            
                            # Validate features (check if all zeros - indicates processing failure)
                            feature_values = [v for k, v in features.items() if k != 'subject_id' and isinstance(v, (int, float))]
                            if all(v == 0 for v in feature_values):
                                st.warning("⚠️ All features are zero. Audio file might be empty or invalid. Try recording again or use a different file.")
                                st.info("💡 **Tips:**\n- Ensure microphone access is granted\n- Record for at least 3-5 seconds\n- Speak clearly during recording")
                            else:
                                st.session_state.speech_features = features
                                st.success("✅ Speech features extracted!")
                                
                                # Display features summary
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Transcription Length", f"{features.get('transcription_length', 0):.0f} chars")
                                    st.metric("Word Count", f"{features.get('word_count', 0):.0f}")
                                    st.metric("Pause Count", f"{features.get('pause_count', 0):.0f}")
                                with col2:
                                    st.metric("Total Duration", f"{features.get('total_duration', 0):.2f}s")
                                    st.metric("Repetition Ratio", f"{features.get('repetition_ratio', 0):.3f}")
                                    st.metric("Mean Pitch", f"{features.get('mean_pitch', 0):.2f} Hz")
                                
                                # Full features (expandable)
                                with st.expander("📋 View All Features"):
                                    st.json(features)
                                
                                # Download
                                df = pd.DataFrame([features])
                                st.download_button(
                                    label="📥 Download Features CSV",
                                    data=df.to_csv(index=False),
                                    file_name="speech_features.csv",
                                    mime="text/csv",
                                    key="download_speech_upload"
                                )
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Error processing audio: {error_msg}")
                        
                        # Provide specific solutions
                        if "format" in error_msg.lower() or "not recognised" in error_msg.lower() or "ffmpeg" in error_msg.lower():
                            st.warning("🎯 **Audio format issue detected**")
                            st.markdown("""
                            **The browser recorded in WebM/Opus format, which requires ffmpeg for conversion.**
                            
                            **Solutions (choose one):**
                            
                            **Option 1: Install ffmpeg (recommended)**
                            - **Windows**: Download from https://www.gyan.dev/ffmpeg/builds/
                              - Extract the zip file
                              - Add `ffmpeg.exe` to your system PATH, OR
                              - Place `ffmpeg.exe` in the same folder as this script
                            - **Anaconda/Conda**: 
                              ```bash
                              conda install -c conda-forge ffmpeg
                              ```
                            - **After installing**, restart this dashboard
                            
                            **Option 2: Use pre-recorded audio**
                            - Upload an existing WAV, MP3, or M4A file instead
                            - These formats work without ffmpeg
                            
                            **Option 3: Try a different browser**
                            - Some browsers may record in different formats
                            """)
                        else:
                            st.info("💡 **Possible issues:**\n- Audio file is corrupted or empty\n- File format not supported\n- Try recording again or use a different audio file")
                        
                        import traceback
                        with st.expander("🔍 Technical Details"):
                            st.code(traceback.format_exc())
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
    with tab2:
        st.subheader("Writing/Text Input")
        text_input = st.text_area("Enter text", height=200)
        subject_id_text = st.text_input("Subject ID (optional)", key="text_id")
        
        if st.button("Process Text", key="process_text"):
            if text_input:
                try:
                    features = process_text(text_input, subject_id=subject_id_text)
                    st.session_state['text_features'] = features
                    st.success("✅ Text features extracted!")
                    st.json(features)
                    
                    # Download
                    df = pd.DataFrame([features])
                    st.download_button(
                        label="Download Features CSV",
                        data=df.to_csv(index=False),
                        file_name="text_features.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please enter some text")
    
    with tab3:
        st.subheader("Behavior Log Input")
        behavior_file = st.file_uploader("Upload behavior log CSV", type=['csv'])
        subject_id_behavior = st.text_input("Subject ID (optional)", key="behavior_id")
        
        if st.button("Process Behavior Log", key="process_behavior"):
            if behavior_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp:
                    tmp.write(behavior_file.read().decode('utf-8'))
                    tmp_path = tmp.name
                
                try:
                    features = process_behavior_log(tmp_path, subject_id=subject_id_behavior)
                    st.session_state['behavior_features'] = features
                    st.success("✅ Behavior features extracted!")
                    st.json(features)
                    
                    # Download
                    df = pd.DataFrame([features])
                    st.download_button(
                        label="Download Features CSV",
                        data=df.to_csv(index=False),
                        file_name="behavior_features.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                finally:
                    os.unlink(tmp_path)
            else:
                st.warning("⚠️ Please upload a CSV file")


def show_model_training():
    """Model training page."""
    st.header("🤖 Model Training")
    
    # Check for existing models
    saved_models = list_saved_models()
    
    if saved_models:
        st.success(f"✅ Found {len(saved_models)} saved model(s)")
        st.write("Saved models:", ", ".join(saved_models))
    
    # Training options
    st.subheader("Train New Models")
    
    feature_file = st.file_uploader("Upload combined features CSV", type=['csv'])
    
    if feature_file:
        df = pd.read_csv(feature_file)
        st.write(f"📊 Loaded {len(df)} samples with {len(df.columns)} features")
        st.dataframe(df.head())
        
        # Select target column
        target_col = st.selectbox("Select target column", df.columns)
        
        if st.button("Train Models"):
            try:
                # Prepare data
                X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
                y = df[target_col].values
                
                # Split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                )
                
                # Train
                with st.spinner("Training models..."):
                    models = build_models()
                    results = train_models(X_train, y_train, X_test, y_test, models=models)
                
                st.session_state['training_results'] = results
                st.session_state['models_trained'] = True
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.success("✅ Models trained successfully!")
                
                # Show results
                results_df = pd.DataFrame({
                    'Model': [r['metrics']['model_name'] if 'metrics' in r else name for name, r in results.items()],
                    'Accuracy': [r.get('metrics', {}).get('accuracy', 0) for r in results.values()],
                    'ROC-AUC': [r.get('metrics', {}).get('roc_auc', 0) for r in results.values()],
                    'F1': [r.get('metrics', {}).get('f1', 0) for r in results.values()]
                })
                st.dataframe(results_df.sort_values('Accuracy', ascending=False))
                
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def show_fusion():
    """Fusion and predictions page."""
    st.header("🔀 Multimodal Fusion & Predictions")
    
    # Check available modalities
    modalities = {}
    if 'speech_features' in st.session_state:
        modalities['speech'] = st.session_state['speech_features']
    if 'text_features' in st.session_state:
        modalities['text'] = st.session_state['text_features']
    if 'behavior_features' in st.session_state:
        modalities['behavior'] = st.session_state['behavior_features']
    
    if not modalities:
        st.warning("⚠️ No features available. Please input data first.")
        return
    
    st.success(f"✅ {len(modalities)} modality(ies) available: {', '.join(modalities.keys())}")
    
    # Fusion method
    fusion_method = st.selectbox("Fusion Method", ['weighted_average', 'meta_learner'])
    
    if st.button("Generate Risk Score"):
        st.info("ℹ️ Fusion requires trained models. Use Model Training page first.")
        # Placeholder: would load models and perform fusion


def show_results():
    """Results and explainability page."""
    st.header("📈 Results & Explainability")
    
    if 'training_results' not in st.session_state:
        st.warning("⚠️ No training results available. Train models first.")
        return
    
    results = st.session_state['training_results']
    
    # Model comparison
    st.subheader("Model Comparison")
    results_data = []
    for name, res in results.items():
        results_data.append({
            'Model': name,
            'Accuracy': res.get('metrics', {}).get('accuracy', 0),
            'ROC-AUC': res.get('metrics', {}).get('roc_auc', 0),
            'F1': res.get('metrics', {}).get('f1', 0)
        })
    
    results_df = pd.DataFrame(results_data).sort_values('Accuracy', ascending=False)
    st.dataframe(results_df, use_container_width=True)
    
    # Visualizations
    if 'X_test' in st.session_state and 'y_test' in st.session_state:
        selected_model = st.selectbox("Select model for analysis", list(results.keys()))
        
        if selected_model:
            model_results = results[selected_model]
            model = model_results.get('model')
            
            if model:
                # SHAP plot
                if st.checkbox("Show SHAP Analysis"):
                    try:
                        X_sample = st.session_state['X_test'][:min(50, len(st.session_state['X_test']))]
                        plot_shap_summary(
                            model, X_sample,
                            model_name=selected_model
                        )
                    except Exception as e:
                        st.error(f"SHAP analysis failed: {e}")


if __name__ == "__main__":
    main()

