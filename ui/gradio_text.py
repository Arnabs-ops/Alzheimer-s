"""
Gradio UI for text/writing input: analyze text and extract features.
"""
import gradio as gr
import pandas as pd
from scripts.text_input import process_text


def process_text_interface(text_input, subject_id):
    """Process text input and return features."""
    if not text_input or len(text_input.strip()) == 0:
        return "⚠️ Please enter some text", None
    
    try:
        # Process text
        features = process_text(text_input, subject_id=subject_id)
        
        # Format output
        output_text = "✅ Text Features Extracted:\n\n"
        for key, value in features.items():
            if key != 'subject_id':
                if isinstance(value, float):
                    output_text += f"{key}: {value:.4f}\n"
                else:
                    output_text += f"{key}: {value}\n"
        
        # Create feature dict for download
        df = pd.DataFrame([features])
        
        return output_text, df.to_csv(index=False)
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def create_text_interface():
    """Create Gradio interface for text input."""
    with gr.Blocks(title="Text Feature Extraction") as demo:
        gr.Markdown("# ✍️ Writing/Text Input & Feature Extraction")
        gr.Markdown("Enter text to extract NLP features: sentence complexity, vocabulary richness, coherence.")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter Text",
                    placeholder="Type or paste text here...",
                    lines=10
                )
                subject_id_input = gr.Textbox(
                    label="Subject ID (optional)",
                    placeholder="patient_001"
                )
                process_btn = gr.Button("Process Text", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Extracted Features",
                    lines=15,
                    interactive=False
                )
                download_csv = gr.File(label="Download Features CSV")
        
        process_btn.click(
            fn=process_text_interface,
            inputs=[text_input, subject_id_input],
            outputs=[output_text, download_csv]
        )
        
        gr.Markdown("### Features Extracted:")
        gr.Markdown("""
        - **Sentence Complexity**: Average sentence length, dependency depth, words per sentence
        - **Vocabulary Richness**: Type-token ratio, unique words, lexical diversity
        - **Coherence**: Embedding similarity, semantic coherence (if transformers available)
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_text_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)

