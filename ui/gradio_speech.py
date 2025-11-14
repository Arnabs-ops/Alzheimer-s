"""
Gradio UI for speech input: record audio and extract features.
"""
import gradio as gr
import os
import tempfile
from scripts.speech_input import process_audio_file


def process_speech_interface(audio_file, subject_id):
    """Process uploaded audio file and return features."""
    if audio_file is None:
        return "⚠️ Please upload an audio file", None
    
    try:
        # Process audio
        features = process_audio_file(audio_file.name, subject_id=subject_id)
        
        # Format output
        output_text = "✅ Speech Features Extracted:\n\n"
        for key, value in features.items():
            if key != 'subject_id':
                output_text += f"{key}: {value:.4f}\n"
        
        # Create feature dict for download
        import pandas as pd
        df = pd.DataFrame([features])
        
        return output_text, df.to_csv(index=False)
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def create_speech_interface():
    """Create Gradio interface for speech input."""
    with gr.Blocks(title="Speech Feature Extraction") as demo:
        gr.Markdown("# 🎤 Speech Input & Feature Extraction")
        gr.Markdown("Upload an audio file to extract speech features: pause duration, word repetition, pitch variation.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Upload Audio or Record"
                )
                subject_id_input = gr.Textbox(
                    label="Subject ID (optional)",
                    placeholder="patient_001"
                )
                process_btn = gr.Button("Process Audio", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Extracted Features",
                    lines=15,
                    interactive=False
                )
                download_csv = gr.File(label="Download Features CSV")
        
        process_btn.click(
            fn=process_speech_interface,
            inputs=[audio_input, subject_id_input],
            outputs=[output_text, download_csv]
        )
        
        gr.Markdown("### Features Extracted:")
        gr.Markdown("""
        - **Pause Duration**: Total, mean, max pause times, pause ratio
        - **Word Repetition**: Repetition ratio, unique word ratio, immediate repetitions
        - **Pitch Variation**: Mean pitch, standard deviation, pitch range, variation coefficient
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_speech_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

